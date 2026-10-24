"""GGUF header reader.

Only the header is parsed - tensor data is never read. The goal is to
answer two questions without loading a model:

1. What is the architecture, and therefore how many layers carry a KV
   cache?
2. How many bytes per token does the KV cache occupy?

The second question is where naive implementations go wrong. For a hybrid
architecture (Qwen3.5, Qwen3-Next) only every N-th layer carries a KV
cache; the remaining layers are SSM/Mamba with constant-size recurrent
state. Using all layers overestimates the cache by exactly the interval
factor.

For architectures where the closed form does not hold (BERT-style
encoders use a shared, pooled cache; sliding-window and shared-KV models
have their own rules), the profile is marked as not formula-reliable and
the planner has to calibrate instead.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

# GGUF v3 metadata value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)

_FIXED = {
    _U8: ("<B", 1), _I8: ("<b", 1),
    _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4), _F32: ("<f", 4),
    _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

# Aliases for the numeric type ids, so a reader does not have to remember
# which id means which width.
_I8, _I16, _I32, _I64 = 1, 2, 5, 11
_U8, _U16, _U32, _U64 = 0, 3, 4, 10
_F32, _F64 = 6, 12
_BOOL, _STRING, _ARRAY = 7, 8, 9

# Bytes per element per KV cache type.
# f16 is the reference; q8_0 and q4_0 are llama.cpp's quantised cache types.
CACHE_ELEMENT_BYTES = {
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}


class GGUFError(Exception):
    """Raised when a file is not a readable GGUF header."""


@dataclass(frozen=True)
class ModelProfile:
    """Everything the planner needs to know about one GGUF file."""

    path: Path
    architecture: str
    model_name: str
    quantisation: str | None
    layers: int
    kv_layers: int
    full_attention_interval: int
    n_kv_heads: int
    key_length: int
    value_length: int
    embedding_length: int
    max_context: int
    file_size_bytes: int
    is_embedding_model: bool
    has_sliding_window: bool
    has_shared_kv_layers: bool
    # SSM geometry. Hybrid models (Qwen3.5) mix attention layers with
    # state-space layers. The state of the SSM layers lives in its own
    # buffer, separate from the KV cache.
    ssm_state_size: int | None
    ssm_inner_size: int | None
    ssm_group_count: int | None
    ssm_conv_kernel: int | None
    ssm_time_step_rank: int | None

    @property
    def file_size_mib(self) -> float:
        return self.file_size_bytes / (1024 * 1024)

    @property
    def is_hybrid(self) -> bool:
        """Whether the model mixes attention with state-space layers.

        A hybrid model keeps a recurrent state next to the KV cache.
        That buffer has its own size, and it does not scale with the
        context.
        """
        return self.ssm_state_size is not None

    @property
    def ssm_layers(self) -> int:
        """The number of state-space layers."""
        if not self.is_hybrid:
            return 0
        return self.layers - self.kv_layers

    @property
    def formula_reliable(self) -> bool:
        """Whether the closed-form KV calculation applies to this model.

        False for BERT-style encoders (pooled, shared cache) and for models
        with sliding-window or shared-KV layers. Those need calibration.
        """
        return not (self.is_embedding_model or self.has_sliding_window
                    or self.has_shared_kv_layers)

    @property
    def kv_elements_per_token(self) -> int:
        """K and V elements stored per token, summed over KV-carrying layers."""
        return self.kv_layers * (
            self.n_kv_heads * self.key_length + self.n_kv_heads * self.value_length
        )

    def kv_bytes_per_token(self, cache_type: str = "f16") -> int:
        """KV cache bytes per token for the given cache type."""
        if cache_type not in CACHE_ELEMENT_BYTES:
            raise ValueError(
                f"unknown cache type {cache_type!r}; "
                f"expected one of {sorted(CACHE_ELEMENT_BYTES)}"
            )
        return int(self.kv_elements_per_token * CACHE_ELEMENT_BYTES[cache_type])


def _read_exact(fh, size: int) -> bytes:
    data = fh.read(size)
    if len(data) != size:
        raise GGUFError("unexpected end of file")
    return data


def _read_string(fh) -> str:
    (length,) = struct.unpack("<Q", _read_exact(fh, 8))
    return _read_exact(fh, length).decode("utf-8", "replace")


def _read_value(fh, value_type: int):
    if value_type == _STRING:
        return _read_string(fh)
    if value_type in _FIXED:
        fmt, size = _FIXED[value_type]
        return struct.unpack(fmt, _read_exact(fh, size))[0]
    if value_type == _ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(fh, 4))
        (count,) = struct.unpack("<Q", _read_exact(fh, 8))
        if elem_type == _STRING:
            for _ in range(count):
                _read_string(fh)
        elif elem_type in _FIXED:
            _, size = _FIXED[elem_type]
            fh.seek(count * size, 1)
        else:
            raise GGUFError(f"unsupported array element type {elem_type}")
        return None
    raise GGUFError(f"unsupported metadata value type {value_type}")


def _skip_value(fh, value_type: int) -> None:
    """Skip a value of the given type without decoding it.

    Used for metadata the planner does not need. A GGUF header may
    contain types this reader does not model; refusing to parse the file
    because of an unused key would make the tool useless on real models.
    """
    if value_type == _STRING:
        _read_string(fh)
    elif value_type in _FIXED:
        _, size = _FIXED[value_type]
        fh.seek(size, 1)
    elif value_type == _ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(fh, 4))
        (count,) = struct.unpack("<Q", _read_exact(fh, 8))
        if elem_type == _STRING:
            for _ in range(count):
                _read_string(fh)
        elif elem_type in _FIXED:
            _, size = _FIXED[elem_type]
            fh.seek(count * size, 1)
        else:
            raise GGUFError(f"unsupported array element type {elem_type}")
    else:
        raise GGUFError(f"unsupported metadata value type {value_type}")


def _read_metadata(fh) -> dict:
    magic = _read_exact(fh, 4)
    if magic != b"GGUF":
        raise GGUFError("not a GGUF file (bad magic)")
    (version,) = struct.unpack("<I", _read_exact(fh, 4))
    if version not in (2, 3):
        raise GGUFError(f"unsupported GGUF version {version}")
    _read_exact(fh, 8)  # tensor count - not needed for the header
    (kv_count,) = struct.unpack("<Q", _read_exact(fh, 8))
    metadata: dict = {}
    for _ in range(kv_count):
        key = _read_string(fh)
        (value_type,) = struct.unpack("<I", _read_exact(fh, 4))
        metadata[key] = _read_value(fh, value_type)
    return metadata


def _quantisation_from_name(name: str) -> str | None:
    """Best-effort quantisation tag from a filename.

    Only used as a label; the authoritative size comes from the file.
    """
    for tag in ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0",
                "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_XXS",
                "IQ3_XS", "IQ3_XX", "IQ2_XXS", "F16", "BF16"):
        if tag in name:
            return tag
    return None


def read_profile(path: str | Path) -> ModelProfile:
    """Read the GGUF header of ``path`` and derive the planner's inputs."""
    path = Path(path)
    if not path.is_file():
        raise GGUFError(f"not a file: {path}")
    with path.open("rb") as fh:
        meta = _read_metadata(fh)

    architecture = meta.get("general.architecture", "unknown")
    prefix = f"{architecture}."

    def arch(key, default=None):
        value = meta.get(prefix + key)
        return default if value is None else value

    layers = int(arch("block_count", 0) or 0)
    n_heads = int(arch("attention.head_count", 0) or 0)
    n_kv_heads = int(arch("attention.head_count_kv", n_heads) or n_heads)
    embedding_length = int(arch("embedding_length", 0) or 0)
    key_length = arch("attention.key_length")
    value_length = arch("attention.value_length")
    if key_length is None:
        key_length = embedding_length // n_heads if n_heads else 0
    if value_length is None:
        value_length = key_length
    interval = int(arch("full_attention_interval", 1) or 1)

    # Hybrid architectures store a KV cache only on every interval-th layer.
    kv_layers = max(1, layers // interval) if interval > 1 else layers

    is_embedding = bool(arch("attention.causal") is False
                         and architecture.endswith("bert"))
    has_swa = bool(arch("attention.sliding_window"))
    has_shared_kv = bool(arch("attention.shared_kv_layers"))

    model_name = meta.get("general.name") or path.stem
    quant = meta.get("general.file_type")
    quant_label = _quantisation_from_name(path.name)

    # Hybrid architectures keep a recurrent state next to the KV cache.
    # The keys are named after the architecture:
    #   qwen35.ssm.state_size, qwen35.ssm.inner_size, ...
    ssm_state_size = arch("ssm.state_size")
    ssm_inner_size = arch("ssm.inner_size")
    ssm_group_count = arch("ssm.group_count")
    ssm_conv_kernel = arch("ssm.conv_kernel")
    ssm_time_step_rank = arch("ssm.time_step_rank")

    return ModelProfile(
        path=path,
        architecture=architecture,
        model_name=str(model_name),
        quantisation=quant_label,
        layers=layers,
        kv_layers=kv_layers,
        full_attention_interval=interval,
        n_kv_heads=n_kv_heads,
        key_length=int(key_length),
        value_length=int(value_length),
        embedding_length=embedding_length,
        max_context=int(arch("context_length", 0) or 0),
        file_size_bytes=path.stat().st_size,
        is_embedding_model=is_embedding,
        has_sliding_window=has_swa,
        has_shared_kv_layers=has_shared_kv,
        ssm_state_size=None if ssm_state_size is None else int(ssm_state_size),
        ssm_inner_size=None if ssm_inner_size is None else int(ssm_inner_size),
        ssm_group_count=None if ssm_group_count is None else int(ssm_group_count),
        ssm_conv_kernel=None if ssm_conv_kernel is None else int(ssm_conv_kernel),
        ssm_time_step_rank=(None if ssm_time_step_rank is None
                            else int(ssm_time_step_rank)),
    )

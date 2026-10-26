"""The daemon configuration.

Every field has a working default. A fresh install runs with an empty
config, and the daemon fills in what it needs from the planner.

The config is YAML. It is read once at startup and written by
``cpt-mem0 init``. Nothing in the daemon changes it at runtime - a
running service that rewrites its own config is a service that loses
its settings on the next restart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def default_config_path() -> Path:
    """Where the config lives.

    **Through the one place that decides it.** The path is asked
    from the installer every time, and not computed here from the
    environment. Otherwise a test that runs on this machine writes
    its own settings into the configuration of the installed app -
    and the app then looks for its store in a temporary folder that
    the next test run deletes.
    """
    from .install import config_dir

    return config_dir() / "config.yaml"


def default_store_path() -> Path:
    """Where the vector store lives.

    **Through the one place that decides it** - see
    ``default_config_path``.
    """
    from .install import data_dir

    return data_dir() / "store"
    base = os.environ.get("XDG_DATA_HOME")
    if not base:
        base = str(Path.home() / ".local" / "share")
    return Path(base) / "cpt-mem0" / "store"


@dataclass
class ServerConfig:
    """The REST and MCP endpoint."""

    host: str = "127.0.0.1"
    port: int = 8765
    api_keys: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LlmConfig:
    """The language model.

    ``model`` and ``quantisation`` are None until the planner fills
    them in. The context is None for the same reason.
    """

    model: str | None = None
    quantisation: str | None = None
    model_file: Path | None = None
    context: int | None = None
    parallel: int = 2
    kv_cache_type: str = "q8_0"
    temperature: float = 0.0
    max_tokens: int = 2000
    port: int = 8766
    #: The flags for llama-server. The planner fills them in; they are
    #: not hand-written, because a wrong flag means a wrong context.
    flags: list[str] = field(default_factory=list)


@dataclass
class EmbedderConfig:
    """The embedder.

    The backend "auto" means: llama.cpp when a GPU is free, fastembed
    otherwise. The dimension count has to match the model, and the
    vectors in the store were made with one dimension count - a
    mismatch breaks the search silently.
    """

    backend: str = "auto"
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    model_file: Path | None = None
    dimensions: int = 768
    port: int = 8767
    context: int = 2048
    #: The flags for llama-server.
    flags: list[str] = field(default_factory=list)


@dataclass
class StoreConfig:
    """The vector store.

    Qdrant embedded, one directory. The daemon is the only client:
    Qdrant's embedded mode takes an exclusive lock on the directory,
    and a second client fails.
    """

    provider: str = "qdrant"
    path: Path = field(default_factory=default_store_path)
    collection: str = "mem0"


@dataclass
class LlamaCppConfig:
    """The runtime."""

    build: str = "b11838"
    variant: str = "auto"
    gpu_layers: int = 99


@dataclass
class PlannerConfigSettings:
    """The planner knobs.

    The context floors are the minimum for the service to be able to
    extract facts at all. Below them the daemon starts but cannot do
    its job, which is worse than not starting.
    """

    reserve_mib: int | None = None
    headroom_mib: int | None = None
    context_floor_total: int = 21000
    context_floor_slot: int = 12288
    kv_cache_type_order: list[str] = field(
        default_factory=lambda: ["q8_0", "f16", "q4_0"])
    allow_degraded: bool = False
    calibration: str = "auto"


@dataclass
class Config:
    """The whole configuration."""

    version: int = 1
    user_id: str = field(default_factory=lambda: os.environ.get(
        "USERNAME") or os.environ.get("USER") or "local")
    agent_id: str = "cpt-mem0"
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    planner: PlannerConfigSettings = field(
        default_factory=PlannerConfigSettings)
    check_on_start: bool = False

    def with_llm(self, **changes: Any) -> Config:
        """A copy with changes in the llm section."""
        return replace(self, llm=replace(self.llm, **changes))

    def with_embedder(self, **changes: Any) -> Config:
        """A copy with changes in the embedder section."""
        return replace(self, embedder=replace(self.embedder, **changes))


def _parse_yaml(text: str) -> dict[str, Any]:
    """Parse YAML.

    PyYAML is optional. A config written by the app is small enough to
    read without it, and a missing PyYAML must not stop the daemon
    from reading its own settings.
    """
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    except Exception as exc:
        raise ValueError(f"the config is not valid YAML: {exc}") from exc

    return _parse_minimal_yaml(text)


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """A YAML reader for the subset the app writes.

    It handles nested mappings, lists of scalars, lists of mappings,
    inline lists, quoted strings, integers, floats, booleans and null.
    It is not a general YAML implementation and does not try to be.

    The reader is recursive over the lines. The indentation says how
    deep a block goes, and the first line of a block says whether it
    is a mapping or a list.
    """
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = _strip_comment(raw)
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip())
        lines.append((indent, stripped.strip()))

    if not lines:
        return {}

    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError(f"unexpected content at line {index + 1}")
    if not isinstance(value, dict):
        raise ValueError("the config is not a mapping")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int,
                 indent: int) -> tuple[Any, int]:
    """A block at one indentation level.

    It is a list when the first line starts with a dash, a mapping
    otherwise.
    """
    if index >= len(lines):
        return {}, index

    if lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_list(lines: list[tuple[int, str]], index: int,
                indent: int) -> tuple[list[Any], int]:
    """A list of scalars or mappings."""
    items: list[Any] = []

    while index < len(lines):
        line_indent, line = lines[index]
        if line_indent < indent:
            break
        if line_indent == indent and not line.startswith("- "):
            # The list ends here; this line belongs to the parent.
            break
        if line_indent > indent and not line.startswith("- "):
            # A continuation of the item above - a mapping key that
            # belongs to the last dash. It is consumed by the branch
            # below, not here.
            break
        if not line.startswith("- "):
            break

        body = line[2:].strip()
        if ":" in body and not body.startswith(("\"", "'")):
            # A mapping that starts on the dash line. Its remaining
            # keys are indented deeper than the dash.
            key, value = _split_key_value(body)
            mapping: dict[str, Any] = {key: _scalar(value)}
            index += 1
            while index < len(lines):
                next_indent, next_line = lines[index]
                if next_indent <= indent:
                    break
                if next_line.startswith("- "):
                    break
                sub_key, sub_value = _split_key_value(next_line)
                mapping[sub_key] = _scalar(sub_value)
                index += 1
            items.append(mapping)
        else:
            items.append(_scalar(body))
            index += 1

    return items, index

def _split_key_value(line: str) -> tuple[str, str]:
    """The key and the value of a mapping line.

    The first colon separates them, but only when it is followed by a
    space or ends the line. A value like "Qwen3.5-9B" has no colon,
    and a value like "http://127.0.0.1:8765" has colons that do not
    separate anything.

    Without this, the value "Qwen3.5-9B" would be read as the key
    "Qwen3.5" with the value "9B" - silently wrong.
    """
    # A colon at the end of the line, or one followed by a space.
    for index, char in enumerate(line):
        if char != ":":
            continue
        if index + 1 == len(line):
            return line[:index].strip(), ""
        if line[index + 1] in " \t":
            return line[:index].strip(), line[index + 1:].strip()
    return line.strip(), ""


def _parse_mapping(lines: list[tuple[int, str]], index: int,
                   indent: int) -> tuple[dict[str, Any], int]:
    """A mapping of keys to values or nested blocks."""
    result: dict[str, Any] = {}

    while index < len(lines):
        line_indent, line = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"unexpected indentation at line {index + 1}")
        if line.startswith("- "):
            break

        key, value = _split_key_value(line)
        index += 1

        if value == "":
            # A nested block, if there is one. A block sequence may
            # sit on the same indentation as its key - that is the
            # usual YAML style:
            #
            #     flags:
            #     - --fit
            #     - off
            #
            # and it may also be indented deeper. Both are the same
            # thing.
            if index < len(lines):
                next_indent, next_line = lines[index]
                if next_line.startswith("- ") and next_indent >= indent:
                    child, index = _parse_list(lines, index, next_indent)
                    result[key] = child
                elif next_indent > indent:
                    child, index = _parse_block(lines, index, next_indent)
                    result[key] = child
                else:
                    result[key] = None
        else:
            result[key] = _scalar(value)

    return result, index


def _strip_comment(line: str) -> str:
    """Remove a trailing comment.

    A hash inside a quoted string is not a comment.
    """
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in ("\"", "'"):
            quote = char
            continue
        if char == "#" and (index == 0 or line[index - 1] in " \t"):
            return line[:index]
    return line


def _unescape(text: str) -> str:
    """The escapes of a double-quoted string.

    Only the ones the writer produces: a backslash and a quote.
    """
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            following = text[index + 1]
            if following in ('\\', '"', "'", "n", "t", "r"):
                result.append({"n": "\n", "t": "\t", "r": "\r"}.get(
                    following, following))
                index += 2
                continue
        result.append(char)
        index += 1
    return "".join(result)


def _scalar(text: str) -> Any:
    """The value of a scalar."""
    if text.startswith(("\"", "'")) and len(text) >= 2:
        value = text[1:-1]
        if text[0] == '"':
            return _unescape(value)
        return value
    if text in ("null", "~", ""):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text.startswith("[") and text.endswith("]"):
        # An inline list.
        body = text[1:-1].strip()
        if not body:
            return []
        return [_scalar(part.strip()) for part in body.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def load_config(path: Path | None = None) -> Config:
    """Read the config.

    A missing file is not an error: the defaults are a working
    configuration. The file is only written when something is
    changed.
    """
    target = path or default_config_path()
    if not target.is_file():
        return Config()

    text = target.read_text(encoding="utf-8")
    try:
        raw = _parse_yaml(text)
    except ValueError:
        raise
    if not isinstance(raw, dict):
        raise ValueError(f"the config at {target} is not a mapping")

    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a config from a mapping.

    Unknown keys are ignored. A config written by a newer version
    must not stop an older one from starting.
    """
    config = Config()

    # The identity fields live under "identity" in the written config,
    # but a hand-written config may put them at the top level. Both
    # are read.
    identity = raw.get("identity") or {}
    if not isinstance(identity, dict):
        identity = {}
    if "user_id" in identity:
        config.user_id = str(identity["user_id"])
    elif "user_id" in raw:
        config.user_id = str(raw["user_id"])
    if "agent_id" in identity:
        config.agent_id = str(identity["agent_id"])
    elif "agent_id" in raw:
        config.agent_id = str(raw["agent_id"])
    if "check_on_start" in raw:
        config.check_on_start = bool(raw["check_on_start"])

    server = raw.get("server") or {}
    if isinstance(server, dict):
        if "host" in server:
            config.server.host = str(server["host"])
        if "port" in server:
            config.server.port = int(server["port"])
        if isinstance(server.get("api_keys"), list):
            config.server.api_keys = list(server["api_keys"])

    llm = raw.get("llm") or {}
    if isinstance(llm, dict):
        config.llm.model = llm.get("model")
        config.llm.quantisation = llm.get("quantization") or llm.get(
            "quantisation")
        if llm.get("model_file"):
            config.llm.model_file = Path(str(llm["model_file"]))
        config.llm.context = llm.get("context")
        if "parallel" in llm:
            config.llm.parallel = int(llm["parallel"])
        if "kv_cache_type" in llm:
            config.llm.kv_cache_type = str(llm["kv_cache_type"])
        if "temperature" in llm:
            config.llm.temperature = float(llm["temperature"])
        if "max_tokens" in llm:
            config.llm.max_tokens = int(llm["max_tokens"])
        if "port" in llm:
            config.llm.port = int(llm["port"])
        if isinstance(llm.get("flags"), list):
            config.llm.flags = [str(x) for x in llm["flags"]]

    embedder = raw.get("embedder") or {}
    if isinstance(embedder, dict):
        if "backend" in embedder:
            config.embedder.backend = str(embedder["backend"])
        if "model" in embedder:
            config.embedder.model = str(embedder["model"])
        if embedder.get("model_file"):
            config.embedder.model_file = Path(str(embedder["model_file"]))
        if "dimensions" in embedder:
            config.embedder.dimensions = int(embedder["dimensions"])
        if "port" in embedder:
            config.embedder.port = int(embedder["port"])
        if "context" in embedder:
            config.embedder.context = int(embedder["context"])
        if isinstance(embedder.get("flags"), list):
            config.embedder.flags = [str(x) for x in embedder["flags"]]

    store = raw.get("store") or {}
    if isinstance(store, dict):
        if "provider" in store:
            config.store.provider = str(store["provider"])
        if store.get("path"):
            config.store.path = Path(str(store["path"]))
        if "collection" in store:
            config.store.collection = str(store["collection"])

    llama = raw.get("llama_cpp") or {}
    if isinstance(llama, dict):
        if "build" in llama:
            config.llama_cpp.build = str(llama["build"])
        if "variant" in llama:
            config.llama_cpp.variant = str(llama["variant"])
        if "gpu_layers" in llama:
            config.llama_cpp.gpu_layers = int(llama["gpu_layers"])

    planner = raw.get("planner") or {}
    if isinstance(planner, dict):
        if "reserve_mib" in planner:
            config.planner.reserve_mib = planner["reserve_mib"]
        if "headroom_mib" in planner:
            config.planner.headroom_mib = planner["headroom_mib"]
        if "context_floor_total" in planner:
            config.planner.context_floor_total = int(
                planner["context_floor_total"])
        if "context_floor_slot" in planner:
            config.planner.context_floor_slot = int(
                planner["context_floor_slot"])
        if isinstance(planner.get("kv_cache_type_order"), list):
            config.planner.kv_cache_type_order = [
                str(x) for x in planner["kv_cache_type_order"]]
        if "allow_degraded" in planner:
            config.planner.allow_degraded = bool(planner["allow_degraded"])
        if "calibration" in planner:
            config.planner.calibration = str(planner["calibration"])

    return config


def to_dict(config: Config) -> dict[str, Any]:
    """The config as a mapping, ready to be written."""
    return {
        "version": config.version,
        "identity": {
            "user_id": config.user_id,
            "agent_id": config.agent_id,
        },
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "api_keys": config.server.api_keys,
        },
        "llm": {
            "model": config.llm.model,
            "quantization": config.llm.quantisation,
            "model_file": str(config.llm.model_file)
            if config.llm.model_file else None,
            "context": config.llm.context,
            "parallel": config.llm.parallel,
            "kv_cache_type": config.llm.kv_cache_type,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "port": config.llm.port,
            "flags": list(config.llm.flags),
        },
        "embedder": {
            "backend": config.embedder.backend,
            "model": config.embedder.model,
            "model_file": str(config.embedder.model_file)
            if config.embedder.model_file else None,
            "dimensions": config.embedder.dimensions,
            "port": config.embedder.port,
            "context": config.embedder.context,
            "flags": list(config.embedder.flags),
        },
        "store": {
            "provider": config.store.provider,
            "path": str(config.store.path),
            "collection": config.store.collection,
        },
        "llama_cpp": {
            "build": config.llama_cpp.build,
            "variant": config.llama_cpp.variant,
            "gpu_layers": config.llama_cpp.gpu_layers,
        },
        "planner": {
            "reserve_mib": config.planner.reserve_mib,
            "headroom_mib": config.planner.headroom_mib,
            "context_floor_total": config.planner.context_floor_total,
            "context_floor_slot": config.planner.context_floor_slot,
            "kv_cache_type_order": list(config.planner.kv_cache_type_order),
            "allow_degraded": config.planner.allow_degraded,
            "calibration": config.planner.calibration,
        },
        "update": {
            "check_on_start": config.check_on_start,
        },
    }


def dump_yaml(data: dict[str, Any], indent: int = 0) -> str:
    """Write the mapping as YAML.

    PyYAML is used when it is there. The fallback writer covers
    everything the config needs: nested mappings, lists of scalars and
    lists of mappings.
    """
    try:
        import yaml
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except ImportError:
        pass

    lines: list[str] = []
    pad = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(dump_yaml(value, indent + 1).rstrip("\n"))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{pad}{key}: []")
            elif all(isinstance(v, dict) for v in value):
                lines.append(f"{pad}{key}:")
                for item in value:
                    # The dash goes on the key's level, the mapping
                    # keys of the item one level deeper:
                    #
                    #     api_keys:
                    #     - name: local
                    #       key: abc
                    #
                    # The first key of the item shares the line with
                    # the dash, the rest are indented by two spaces
                    # relative to it.
                    item_pad = pad + "    "
                    first = True
                    for sub_key, sub_value in item.items():
                        if isinstance(sub_value, (dict, list)):
                            body = dump_yaml({sub_key: sub_value},
                                             indent + 2).rstrip("\n")
                            if first:
                                lines.append(f"{pad}  - {body}")
                            else:
                                lines.append(body)
                        else:
                            text = f"{sub_key}: {_write_scalar(sub_value)}"
                            if first:
                                lines.append(f"{pad}  - {text}")
                            else:
                                lines.append(f"{item_pad}{text}")
                        first = False
            else:
                lines.append(f"{pad}{key}:")
                for item in value:
                    lines.append(f"{pad}  - {_write_scalar(item)}")
        else:
            lines.append(f"{pad}{key}: {_write_scalar(value)}")
    return "\n".join(lines) + "\n"


def _write_scalar(value: Any) -> str:
    """A scalar as YAML."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(c in text for c in ":#{}[],&*?|<>=!%@`") or \
            text.strip() != text or "\\" in text:
        # A backslash is an escape character in YAML. A Windows path
        # without escaping it is not valid YAML: "C:\Users" is read
        # as an escape sequence.
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def save_config(config: Config, path: Path | None = None) -> Path:
    """Write the config."""
    target = path or default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_yaml(to_dict(config)), encoding="utf-8")
    return target

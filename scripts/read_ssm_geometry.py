"""Read the SSM geometry of every catalogue model from its GGUF header.

The recurrent-state buffer only exists for hybrid architectures, and
its size is not derivable from the attention geometry. The keys are in
the GGUF header, so they can be read over an HTTP range request
without downloading the model.

Usage:
    python scripts/read_ssm_geometry.py [--catalogue catalogue.json]

Writes the geometry back into catalogue.json, adding the ssm.* keys to
the geometry block of every model and marking the measured recurrent
state size where one is known.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.catalogue import default_catalogue_path  # noqa: E402

#: How much of the file the header needs. The metadata block of a
#: multi-billion parameter model is well under this.
HEADER_BYTES = 32 * 1024 * 1024

#: The recurrent state sizes measured against a real llama-server.
#:
#: Qwen3.5-9B-Q8_0 on an RTX 3060:
#:   llama_memory_recurrent: CUDA0 RS buffer size = 100.50 MiB
#:
#: The size does not depend on the context. It depends on the number of
#: state-space layers and on the state size, so it is stored per model
#: rather than computed.
MEASURED_RS_MIB = {
    "Qwen3.5-9B": 100.50,
}


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    return data[offset:offset + length].decode("utf-8", "replace"), offset + length


def _read_value(data: bytes, offset: int, value_type: int):
    if value_type == 0:
        return data[offset], offset + 1
    if value_type == 1:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    if value_type == 2:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    if value_type == 3:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    if value_type == 4:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    if value_type == 5:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if value_type == 6:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    if value_type == 7:
        return data[offset] != 0, offset + 1
    if value_type == 8:
        return _read_string(data, offset)
    if value_type == 9:
        element_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        values = []
        for _ in range(count):
            value, offset = _read_value(data, offset, element_type)
            values.append(value)
        return values, offset
    if value_type == 10:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    if value_type == 11:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    if value_type == 12:
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    raise ValueError(f"unknown GGUF value type {value_type}")


def parse_metadata(data: bytes) -> dict:
    """Parse the metadata block of a GGUF header."""
    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    count = struct.unpack_from("<Q", data, 16)[0]
    offset = 24
    metadata: dict = {}
    for _ in range(count):
        key, offset = _read_string(data, offset)
        value_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        value, offset = _read_value(data, offset, value_type)
        metadata[key] = value
    return metadata


def fetch_header(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(
        url, headers={"Range": f"bytes=0-{HEADER_BYTES}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", default=str(default_catalogue_path()))
    parser.add_argument("--write", action="store_true",
                        help="write the geometry back into the catalogue")
    args = parser.parse_args()

    path = Path(args.catalogue)
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = raw["models"] if isinstance(raw, dict) else raw

    for model in models:
        geometry = model.setdefault("geometry", {})
        architecture = geometry.get("architecture", "")
        prefix = f"{architecture}."
        print(f"\n{model['id']}  ({architecture})")

        # Find a GGUF of this model on Hugging Face.
        repo = model.get("repo", "")
        quant = model["quantisations"][0]
        file_name = quant["file"]
        url = (f"https://huggingface.co/{repo}/resolve/main/{file_name}")
        try:
            data = fetch_header(url)
            metadata = parse_metadata(data)
        except (urllib.error.URLError, ValueError, OSError) as exc:
            print(f"  could not read the header: {exc}")
            continue

        ssm_keys = {k: v for k, v in metadata.items()
                    if k.startswith(prefix + "ssm.")}
        if not ssm_keys:
            print("  no SSM keys - this model has no recurrent state")
            model.pop("recurrent_state_mib", None)
            continue

        for key in sorted(ssm_keys):
            short = key[len(prefix):]
            geometry[short] = ssm_keys[key]
            print(f"  {short:20} {ssm_keys[key]}")

        measured = MEASURED_RS_MIB.get(model["id"])
        if measured is not None:
            model["recurrent_state_mib"] = measured
            model["recurrent_state_source"] = "measured"
            print(f"  recurrent_state_mib {measured} (measured)")
        else:
            print("  recurrent_state_mib not measured yet")

    if args.write:
        path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"\nwritten: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

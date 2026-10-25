"""Verify the daemon against real llama.cpp servers.

The engine test uses stub endpoints. That proves the mem0
integration but not the other half: that llama.cpp serves the
OpenAI-compatible endpoints the way mem0 expects them.

This script starts real llama-server processes and points the real
mem0 at them.

**Two modes:**

``--embedder-only``
    Starts only the embedder. It needs about 420 MiB of VRAM, so it
    runs next to an existing stack. This answers the protocol
    question: does llama.cpp serve embeddings that mem0 accepts, with
    the right dimension count?

``--full``
    Starts the embedder and the LLM. It needs the whole card. The
    LLM and the embedder stay resident at the same time, which is
    what the app does.

**Run this only with a free card** for the full mode. The embedder
mode needs little.

Usage:
    python scripts/verify_daemon.py --embedder-only
    python scripts/verify_daemon.py --full \\
        --model-file <path> --embedder-file <path>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.engine import Engine, llama_endpoint  # noqa: E402
from cpt_mem0.supervisor import (  # noqa: E402
    ServerSpec,
    Supervisor,
    gpu_buffers,
    gpu_used_mib,
)

DEFAULT_EMBEDDER = (Path.home() / ".cache/huggingface/embed-models"
                    / "nomic-embed-text-v1.5.Q4_K_M.gguf")

#: The flags the app uses for the embedder. Measured on the running
#: stack, not from the documentation.
EMBEDDER_FLAGS = [
    "--parallel", "1",
    "-c", "2048",
    "-ngl", "99",
    "--embeddings",
    "--pooling", "mean",
    "--kv-unified",
]


def find_llama_server() -> Path | None:
    """The llama-server of the installed runtime."""
    from cpt_mem0.install import current_installation

    installation = current_installation()
    if installation.binary is not None:
        return installation.binary

    home = Path.home()
    for pattern in (
        ".unsloth/llama.cpp/build/bin/*/llama-server.exe",
        "llama.cpp/build/bin/*/llama-server.exe",
    ):
        for path in sorted(home.glob(pattern)):
            if path.is_file():
                return path
    return None


def embed_once(port: int, text: str, timeout: float = 60.0):
    """Ask the embedder for a vector.

    This is the exact call mem0 makes. If it works here, it works
    there.
    """
    payload = json.dumps({"input": [text],
                           "model": "nomic-embed-text-v1.5"}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["data"][0]["embedding"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder-only", action="store_true",
                        help="start only the embedder")
    parser.add_argument("--full", action="store_true",
                        help="start the embedder and the LLM")
    parser.add_argument("--model-file", help="path to the GGUF file")
    parser.add_argument("--embedder-file", default=str(DEFAULT_EMBEDDER),
                        help="path to the embedding model")
    parser.add_argument("--embedder-port", type=int, default=8767)
    parser.add_argument("--llm-port", type=int, default=8766)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dimensions", type=int, default=768)
    args = parser.parse_args()

    if not args.full and not args.embedder_only:
        parser.error("one of --embedder-only or --full is needed")

    binary = find_llama_server()
    if binary is None:
        print("llama-server not found. Run 'cpt-mem0 init' first.",
              file=sys.stderr)
        return 2

    embedder_file = Path(args.embedder_file)
    if not embedder_file.is_file():
        print(f"the embedder model is missing: {embedder_file}",
              file=sys.stderr)
        return 2

    specs = [ServerSpec(
        name="embedder",
        binary=binary,
        model=embedder_file,
        port=args.embedder_port,
        flags=EMBEDDER_FLAGS,
        embeddings=True,
    )]

    if args.full:
        if not args.model_file:
            print("--full needs --model-file", file=sys.stderr)
            return 2
        model_file = Path(args.model_file)
        if not model_file.is_file():
            print(f"the model is missing: {model_file}", file=sys.stderr)
            return 2

        from cpt_mem0.catalogue import load_catalogue

        catalogue = load_catalogue(ROOT / "catalogue.json")
        entry = None
        for candidate in catalogue.sorted_entries():
            if candidate.model_id == "Qwen3.5-9B":
                entry = candidate
                break
        flags = (entry.as_flags() if entry else
                 ["-c", "40960", "--parallel", "2", "-ngl", "99",
                  "--fit", "off", "--flash-attn", "on",
                  "--reasoning-format", "none", "--no-context-shift",
                  "--kv-unified"])
        specs.append(ServerSpec(
            name="llm",
            binary=binary,
            model=model_file,
            port=args.llm_port,
            flags=flags,
        ))

    workdir = Path(tempfile.mkdtemp(prefix="cpt-mem0-daemon-"))
    logs = workdir / "logs"
    for spec in specs:
        spec.log_path = logs / f"{spec.name}.log"

    supervisor = Supervisor(specs)
    problems: list[str] = []

    print("=== the daemon against real llama.cpp ===", flush=True)
    print(f"  binary   : {binary}", flush=True)
    for spec in specs:
        print(f"  {spec.name:9}: {spec.model.name} on port {spec.port}",
              flush=True)
    print(f"  VRAM before: {gpu_used_mib()} MiB", flush=True)

    try:
        for spec in specs:
            print(f"\n--- starting the {spec.name} ---", flush=True)
            state = supervisor.start(spec.name, timeout=args.timeout)
            print(f"  ready on port {spec.port}", flush=True)

            # The log is the source of truth. The per-process reading
            # from nvidia-smi is N/A on Windows with the WDDM driver.
            buffers = gpu_buffers(spec.log_path)
            print(f"  device: {buffers['device']}", flush=True)
            print(f"  layers on the GPU: "
                  f"{buffers['offloaded_layers']}/{buffers['total_layers']}",
                  flush=True)
            print(f"  model buffer: {buffers['model_mib']} MiB", flush=True)
            if buffers["offloaded_layers"] is not None and \
                    buffers["total_layers"] is not None and \
                    buffers["offloaded_layers"] < buffers["total_layers"]:
                problems.append(
                    f"the {spec.name} put only {buffers['offloaded_layers']} "
                    f"of {buffers['total_layers']} layers on the GPU")
            if buffers["model_mib"] is not None and \
                    buffers["model_mib"] < 50:
                problems.append(
                    f"the {spec.name} holds only {buffers['model_mib']} MiB "
                    f"of GPU memory - it is probably not on the GPU")

        # The embedder has to answer with a vector of the right size.
        print("\n--- the embedder ---", flush=True)
        vector = embed_once(args.embedder_port,
                            "The user rides a bicycle to the office.")
        print(f"  {len(vector)} dimensions", flush=True)
        if len(vector) != args.dimensions:
            problems.append(
                f"the embedder returned {len(vector)} dimensions, the "
                f"store was created with {args.dimensions}")

        # The real mem0 against the real embedder.
        print("\n--- mem0 against the real embedder ---", flush=True)
        engine = Engine(
            llm=llama_endpoint(args.llm_port, "llm"),
            embedder=llama_endpoint(args.embedder_port, "nomic"),
            store_path=workdir / "store",
            dimensions=args.dimensions,
            user_id="willow")
        engine.open()
        print("  the store is open", flush=True)

        client = engine._memory.vector_store.client
        for collection in client.get_collections().collections:
            info = client.get_collection(collection.name)
            size = getattr(info.config.params.vectors, "size", None)
            print(f"  the collection has {size} dimensions", flush=True)
            if size != args.dimensions:
                problems.append(
                    f"the collection has {size} dimensions, expected "
                    f"{args.dimensions}")

        engine.close()
        print("  the store is closed", flush=True)

    except Exception as exc:
        import traceback
        print(f"\nERROR: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        problems.append(f"{type(exc).__name__}: {exc}")
    finally:
        supervisor.stop_all()
        print(f"\n  VRAM after: {gpu_used_mib()} MiB", flush=True)
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("The daemon works against the real llama.cpp servers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

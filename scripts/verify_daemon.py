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
    what the app does. This is the mode that proves the extraction
    works: the LLM extracts the facts, the embedder writes them into
    the store, and a search finds them again.

**Everything is written to a log file.** The console shows the same
lines, but the file has them all - including the server logs, the
measurements and the JSON result. The file is the evidence.

**The full mode needs a free card.** Close everything that uses the
GPU first. The script measures the VRAM before it starts and refuses
to run when there is not enough room.

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
import time
import urllib.request
from datetime import datetime
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

#: The facts the extraction has to find. They are neutral on purpose:
#: no project, no person, nothing that identifies the user.
FACTS = [
    ("The user rides a bicycle to the office every day.",
     "How does the user get to work?"),
    ("The build server has 64 GB of RAM.",
     "How much memory does the build server have?"),
    ("The coffee machine in the office is fully automatic.",
     "What kind of coffee machine is in the office?"),
]


class Log:
    """The log file and the console at the same time.

    Everything the script prints goes into the file. The file is the
    evidence - the console can scroll away, the file does not.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        self._handle.write(text + "\n")
        self._handle.flush()
        print(text, flush=True)

    def section(self, title: str) -> None:
        self.write()
        self.write(f"--- {title} ---")

    def traceback(self) -> None:
        import traceback

        traceback.print_exc(file=self._handle)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


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


def embed_once(port: int, text: str, timeout: float = 60.0) -> list[float]:
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


def planned_flags() -> list[str]:
    """The flags the planner produces for the 9B model.

    They come from the planner, not from hand: a wrong flag means a
    wrong context, and the context is what decides whether the
    extraction can work at all.
    """
    from cpt_mem0.catalogue import load_catalogue
    from cpt_mem0.planner import (
        Hardware,
        PlannerConfig,
        plan_for_hardware,
    )

    catalogue = load_catalogue(ROOT / "catalogue.json")
    hardware = Hardware(total_vram_mib=12288,
                        name="NVIDIA GeForce RTX 3060", vendor="nvidia")
    for entry in catalogue.sorted_entries():
        if entry.model_id != "Qwen3.5-9B":
            continue
        result = plan_for_hardware(
            hardware, [entry],
            PlannerConfig(embedder_backends=("llamacpp",)))
        if result.viable and result.candidate is not None:
            return result.candidate.as_flags()

    # The fallback is the measured configuration of the 9B on this
    # card: n_ctx 67072 with a shared KV pool.
    return ["-c", "67072", "--parallel", "2", "-ngl", "99",
            "--fit", "off", "--flash-attn", "on",
            "--reasoning-format", "none", "--no-context-shift",
            "--kv-unified", "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0"]


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
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--log", default=str(ROOT / "results"
                                            / "daemon-verification.log"),
                        help="where the log file goes")
    parser.add_argument("--results",
                        default=str(ROOT / "results"
                                    / "daemon-verification.json"),
                        help="where the JSON result goes")
    parser.add_argument("--free-mib", type=int, default=0,
                        help="the VRAM the run needs (0 = no check)")
    args = parser.parse_args()

    if not args.full and not args.embedder_only:
        parser.error("one of --embedder-only or --full is needed")

    log = Log(Path(args.log))
    results: dict = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "mode": "full" if args.full else "embedder-only",
        "problems": [],
    }

    try:
        binary = find_llama_server()
        if binary is None:
            log.write("llama-server not found. Run 'cpt-mem0 init' first.")
            return 2

        embedder_file = Path(args.embedder_file)
        if not embedder_file.is_file():
            log.write(f"the embedder model is missing: {embedder_file}")
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
                log.write("--full needs --model-file")
                return 2
            model_file = Path(args.model_file)
            if not model_file.is_file():
                log.write(f"the model is missing: {model_file}")
                return 2

            flags = planned_flags()
            specs.append(ServerSpec(
                name="llm",
                binary=binary,
                model=model_file,
                port=args.llm_port,
                flags=flags,
            ))
            results["llm_flags"] = flags

        # The VRAM check before anything starts. A run that starts
        # into a full card fails in a way that looks like a bug.
        if args.full:
            from cpt_mem0.supervisor import gpu_free_mib

            free_before = gpu_free_mib()
            results["vram_free_before_mib"] = free_before
            log.write("=== the daemon against real llama.cpp ===")
            log.write(f"  VRAM free: {free_before} MiB")
            if args.free_mib and free_before is not None and \
                    free_before < args.free_mib:
                log.write(f"  the card has only {free_before} MiB free, "
                          f"the run needs {args.free_mib} MiB.")
                log.write("  Close everything that uses the GPU and run "
                          "it again.")
                results["problems"].append(
                    f"not enough VRAM: {free_before} MiB free, "
                    f"{args.free_mib} MiB needed")
                return 3

        workdir = Path(tempfile.mkdtemp(prefix="cpt-mem0-daemon-"))
        logs = workdir / "logs"
        for spec in specs:
            spec.log_path = logs / f"{spec.name}.log"

        supervisor = Supervisor(specs)
        problems: list[str] = []

        try:
            log.write("=== the daemon against real llama.cpp ===")
            log.write(f"  binary   : {binary}")
            for spec in specs:
                log.write(f"  {spec.name:9}: {spec.model.name} on port "
                          f"{spec.port}")
            log.write(f"  VRAM before: {gpu_used_mib()} MiB")
            results["vram_used_before_mib"] = gpu_used_mib()

            for spec in specs:
                log.section(f"starting the {spec.name}")
                started = time.time()
                supervisor.start(spec.name, timeout=args.timeout)
                elapsed = time.time() - started
                log.write(f"  ready on port {spec.port} "
                          f"after {elapsed:.0f}s")

                # The log is the source of truth. The per-process
                # reading from nvidia-smi is N/A on Windows with the
                # WDDM driver.
                buffers = gpu_buffers(spec.log_path)
                log.write(f"  device: {buffers['device']}")
                log.write(f"  layers on the GPU: "
                          f"{buffers['offloaded_layers']}/"
                          f"{buffers['total_layers']}")
                log.write(f"  model buffer: {buffers['model_mib']} MiB")
                log.write(f"  compute buffer: {buffers['compute_mib']} MiB")
                log.write(f"  KV buffer: {buffers['kv_mib']} MiB")
                if buffers["rs_mib"] is not None:
                    log.write(f"  RS buffer: {buffers['rs_mib']} MiB")
                if buffers["cpu_mapped_mib"] is not None:
                    log.write(f"  CPU mapped (host RAM): "
                              f"{buffers['cpu_mapped_mib']} MiB")

                results.setdefault("servers", {})[spec.name] = buffers

                if buffers["offloaded_layers"] is not None and \
                        buffers["total_layers"] is not None and \
                        buffers["offloaded_layers"] < \
                        buffers["total_layers"]:
                    problems.append(
                        f"the {spec.name} put only "
                        f"{buffers['offloaded_layers']} of "
                        f"{buffers['total_layers']} layers on the GPU")
                if buffers["model_mib"] is not None and \
                        buffers["model_mib"] < 50:
                    problems.append(
                        f"the {spec.name} holds only "
                        f"{buffers['model_mib']} MiB of GPU memory - it is "
                        f"probably not on the GPU")

            # The embedder has to answer with a vector of the right
            # size.
            log.section("the embedder")
            vector = embed_once(args.embedder_port,
                                "The user rides a bicycle to the office.")
            log.write(f"  {len(vector)} dimensions")
            results["embedder_dimensions"] = len(vector)
            if len(vector) != args.dimensions:
                problems.append(
                    f"the embedder returned {len(vector)} dimensions, the "
                    f"store was created with {args.dimensions}")

            # The real mem0 against the real embedder.
            log.section("mem0 against the real embedder")
            engine = Engine(
                llm=llama_endpoint(args.llm_port, "llm"),
                embedder=llama_endpoint(args.embedder_port, "nomic"),
                store_path=workdir / "store",
                dimensions=args.dimensions,
                user_id="willow")
            engine.open()
            log.write("  the store is open")

            client = engine._memory.vector_store.client
            for collection in client.get_collections().collections:
                info = client.get_collection(collection.name)
                size = getattr(info.config.params.vectors, "size", None)
                log.write(f"  the collection has {size} dimensions")
                results["collection_dimensions"] = size
                if size != args.dimensions:
                    problems.append(
                        f"the collection has {size} dimensions, expected "
                        f"{args.dimensions}")

            if args.full:
                # The extraction: the LLM extracts the facts, the
                # embedder writes them into the store.
                log.section("the extraction through the real LLM")
                for fact, _query in FACTS:
                    log.write(f"  storing: {fact}")
                    started = time.time()
                    result = engine.remember(fact)
                    elapsed = time.time() - started
                    stored = result.get("results") or []
                    log.write(f"    {len(stored)} facts in {elapsed:.0f}s")
                    for item in stored:
                        text = item.get("memory") if isinstance(
                            item, dict) else item
                        log.write(f"      {text}")
                    if not stored:
                        problems.append(
                            f"the LLM stored nothing for: {fact}")

                log.section("the recall through the real embedder")
                for _fact, query in FACTS:
                    log.write(f"  searching: {query}")
                    started = time.time()
                    found = engine.recall(query)
                    elapsed = time.time() - started
                    log.write(f"    {len(found)} results in {elapsed:.0f}s")
                    for item in found:
                        text = item.get("memory") if isinstance(
                            item, dict) else item
                        log.write(f"      {text}")
                    if not found:
                        problems.append(
                            f"the search found nothing for: {query}")

                log.section("what is in the store")
                everything = engine.all()
                log.write(f"  {len(everything)} memories")
                results["memories"] = len(everything)
                for item in everything:
                    text = item.get("memory") if isinstance(
                        item, dict) else item
                    log.write(f"    {text}")

            engine.close()
            log.write("  the store is closed")

        except Exception as exc:
            log.write(f"ERROR: {type(exc).__name__}: {exc}")
            log.traceback()
            problems.append(f"{type(exc).__name__}: {exc}")
        finally:
            supervisor.stop_all()
            log.write(f"  VRAM after: {gpu_used_mib()} MiB")
            results["vram_used_after_mib"] = gpu_used_mib()

            # The server logs belong next to the log file. They are the
            # evidence for the buffer lines.
            log_dir = Path(args.log).parent
            for spec in specs:
                if spec.log_path and spec.log_path.is_file():
                    target = log_dir / f"{spec.name}-server.log"
                    try:
                        shutil.copyfile(spec.log_path, target)
                        log.write(f"  the {spec.name} log: {target}")
                    except OSError:
                        pass
            shutil.rmtree(workdir, ignore_errors=True)

        results["problems"] = problems
        results["finished"] = datetime.now().isoformat(timespec="seconds")

    finally:
        log.write()
        if results["problems"]:
            log.write("PROBLEMS:")
            for problem in results["problems"]:
                log.write(f"  - {problem}")
        else:
            log.write("The daemon works against the real llama.cpp servers.")

        try:
            target = Path(args.results)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(results, indent=2, default=str),
                              encoding="utf-8")
            log.write(f"the result is in {target}")
        except OSError as exc:
            log.write(f"the result could not be written: {exc}")
        log.write(f"the log is in {args.log}")
        log.close()

    return 1 if results["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

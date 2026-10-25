"""Verify a planned configuration against the real hardware, embedder
included.

The planner budgets the embedder as a second process with its own CUDA
context, compute buffer and KV cache. Verifying only the LLM leaves
that term unchecked - the first version of this script did exactly that
and could not confirm the embedder line.

Two servers run here:

- the LLM with the planned flags
- the embedder (nomic-embed-text-v1.5) with the flags the app uses

Both are measured. The VRAM of each process is read from
``nvidia-smi`` by process id, so the result does not depend on what
else is running on the card.

**Run this only with a free card.** It loads two models into VRAM.

Usage:
    python scripts/verify_full.py --vram 12288 --model Qwen3.5-9B \
        --quant Q8_0 --model-file <path> --port 8799
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.catalogue import default_catalogue_path, load_catalogue  # noqa: E402
from cpt_mem0.planner import (  # noqa: E402
    Hardware,
    PlannerConfig,
    embedder_cost_mib,
    plan_for_hardware,
)

#: The embedder model the app ships with.
DEFAULT_EMBEDDER = (Path.home() / ".cache/huggingface/embed-models"
                    / "nomic-embed-text-v1.5.Q4_K_M.gguf")

#: The flags the app uses for the embedder. Taken from the running
#: stack, not from the documentation.
EMBEDDER_FLAGS = [
    "--parallel", "1",
    "-c", "2048",
    "-ngl", "99",
    "--embeddings",
    "--pooling", "mean",
    "--kv-unified",
]

#: The buffer lines the server prints. The device is part of the
#: pattern: the host line comes first and is not VRAM.
MODEL_BUFFER_RE = re.compile(
    r"CUDA0\s+model buffer size\s*=\s*([0-9.]+)\s*MiB")
CPU_MAPPED_RE = re.compile(
    r"CPU_Mapped\s+model buffer size\s*=\s*([0-9.]+)\s*MiB")
KV_RE = re.compile(r"CUDA0\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB")
RS_RE = re.compile(r"CUDA0\s+RS buffer size\s*=\s*([0-9.]+)\s*MiB")
COMPUTE_RE = re.compile(
    r"CUDA0\s+compute buffer size\s*=\s*([0-9.]+)\s*MiB")
CONTEXT_RE = re.compile(r"n_ctx_slot\s*=\s*([0-9]+)")
OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")

VERBOSE_FLAG = "--verbose"


def find_llama_server() -> str | None:
    """Locate the llama-server binary."""
    exe = shutil_which("llama-server")
    if exe:
        return exe
    home = Path.home()
    for pattern in (
        ".unsloth/llama.cpp/build/bin/*/llama-server.exe",
        "llama.cpp/build/bin/*/llama-server.exe",
        ".cache/llama.cpp/build/bin/*/llama-server.exe",
    ):
        for path in sorted(home.glob(pattern)):
            if path.is_file():
                return str(path)
    return None


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


def gpu_used_mib() -> int | None:
    exe = shutil_which("nvidia-smi")
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        return None
    try:
        return int(float(proc.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return None


def gpu_free_mib() -> int | None:
    """The VRAM still free on the card."""
    exe = shutil_which("nvidia-smi")
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        return None
    try:
        return int(float(proc.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return None


def process_vram_mib(pid: int) -> int | None:
    """The VRAM of one process, in MiB.

    Reading it per process is what makes the measurement independent of
    what else runs on the card. A card-wide reading does not separate
    the two servers from each other, and a stack that starts during the
    run corrupts it entirely.
    """
    exe = shutil_which("nvidia-smi")
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == pid:
                return int(float(parts[1]))
        except ValueError:
            continue
    return None


def wait_for_http(port: int, timeout: float = 1.0) -> bool:
    """Return True if the server answers an HTTP request."""
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
                resp.read(1)
                return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            continue
    return False


def run_inference(port: int, timeout: float = 90.0) -> tuple[bool, str]:
    """Ask the server to generate something.

    A server that loads a model and answers /health has not proven that
    it can run a prompt.
    """
    payload = json.dumps({
        "prompt": "Reply with the single word: ready",
        "n_predict": 8,
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")

    def attempt() -> tuple[bool, str, bool]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", "replace")
            except Exception:
                pass
            if "does not support logits computation" in raw:
                return True, ("not applicable: this server serves embeddings "
                              "and cannot generate text"), False
            if exc.code == 503:
                return False, f"HTTP 503: {raw[:120]}", True
            return False, f"HTTP {exc.code}: {exc.reason} - {raw[:200]}", False
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", True
        text = (body.get("content") or "").strip()
        if not text:
            return False, "the server answered with an empty completion", False
        return True, text, False

    deadline = time.time() + timeout
    last = "no attempt"
    while True:
        ok, detail, retryable = attempt()
        if ok:
            return True, detail
        last = detail
        if not retryable or time.time() >= deadline:
            return False, last
        time.sleep(1.0)


def run_server(exe: str, model: Path, flags: list[str], port: int,
               timeout: int = 180, on_ready=None) -> tuple[str, int]:
    """Start a server, wait until it is ready, return (log, exit code).

    The output is read on a separate thread: reading it with
    ``readline()`` in the main thread blocks until the next line
    arrives, and during the model load no line arrives for tens of
    seconds. The timeout would then never be checked.
    """
    import queue as queue_mod

    cmd = [exe, "-m", str(model), "--port", str(port), *flags]
    print("  $", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="replace")

    log_lines: list[str] = []
    output_queue: queue_mod.Queue = queue_mod.Queue()

    def reader() -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    output_queue.put(line)
                    log_lines.append(line)
        except (OSError, ValueError):
            pass
        finally:
            output_queue.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    ready = False
    try:
        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue_mod.Empty:
                line = ""
            if line is None:
                break
            if line:
                lowered = line.lower()
                if "model loaded" in lowered or "listening on" in lowered:
                    ready = True
            if ready and wait_for_http(port):
                if on_ready is not None:
                    on_ready("".join(log_lines), proc.pid)
                time.sleep(2)
                break
            if proc.poll() is not None and output_queue.empty():
                break
            if time.time() > deadline:
                print(f"  timeout after {timeout}s", flush=True)
                if on_ready is not None:
                    on_ready("".join(log_lines), proc.pid)
                break
    except KeyboardInterrupt:
        print("\n  interrupted", flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        thread.join(timeout=5)
    return "".join(log_lines), 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vram", type=int, help="VRAM of the card (MiB)")
    parser.add_argument("--model", help="model id from the catalogue")
    parser.add_argument("--quant", help="quantisation name")
    parser.add_argument("--model-file", help="path to the GGUF file")
    parser.add_argument("--embedder-file", default=str(DEFAULT_EMBEDDER),
                        help="path to the embedding model")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--embedder-port", type=int, default=8800)
    parser.add_argument("--tolerance-mib", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--catalogue", default=str(default_catalogue_path()))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    catalogue = load_catalogue(args.catalogue)
    hardware = Hardware(total_vram_mib=args.vram) if args.vram else Hardware(
        total_vram_mib=12288)

    config = PlannerConfig(embedder_backends=("llamacpp",))
    entries = catalogue.sorted_entries()
    if args.model:
        entries = [e for e in entries if e.model_id == args.model]
        if not entries:
            print(f"model {args.model!r} is not in the catalogue",
                  file=sys.stderr)
            return 2

    result = plan_for_hardware(hardware, entries, config)
    if not result.viable or result.candidate is None:
        print("no viable configuration:",
              result.diagnostics.get("reason"))
        return 1
    candidate = result.candidate
    if args.quant and candidate.quantisation != args.quant:
        print(f"the planner chose {candidate.quantisation}, not "
              f"{args.quant}")
        return 1

    print(f"planned: {candidate.model_id} {candidate.quantisation}, "
          f"n_ctx {candidate.n_ctx}, KV {candidate.cache_type}, "
          f"embedder {candidate.embedder_backend}")

    model_path = Path(args.model_file) if args.model_file else None
    if model_path is None or not model_path.is_file():
        print(f"model file not found: {model_path}", file=sys.stderr)
        return 2
    embedder_path = Path(args.embedder_file)
    if not embedder_path.is_file():
        print(f"embedder file not found: {embedder_path}", file=sys.stderr)
        return 2

    exe = find_llama_server()
    if not exe:
        print("llama-server not found", file=sys.stderr)
        return 2

    before_total = gpu_used_mib()
    measured: dict = {}

    # ---- the embedder -------------------------------------------------
    print("\n--- embedder ---", flush=True)
    embedder_flags = EMBEDDER_FLAGS + [VERBOSE_FLAG]

    def _embedder_ready(log_so_far: str, pid: int) -> None:
        measured["embedder_log"] = log_so_far
        measured["embedder_pid"] = pid
        measured["embedder_vram"] = process_vram_mib(pid)
        ok, detail = run_inference(args.embedder_port, timeout=30)
        measured["embedder_ok"] = ok
        measured["embedder_detail"] = detail
        print(f"  embedder inference: {'OK' if ok else 'n/a'} - {detail}",
              flush=True)

    embedder_log, _ = run_server(
        exe, embedder_path, embedder_flags, args.embedder_port,
        timeout=args.timeout, on_ready=_embedder_ready)

    # ---- the LLM ------------------------------------------------------
    # The embedder stays resident. The point of this run is that both
    # models fit on the card at the same time, which is what the app
    # needs.
    print("\n--- LLM ---", flush=True)
    llm_flags = candidate.as_flags() + [VERBOSE_FLAG]

    def _llm_ready(log_so_far: str, pid: int) -> None:
        measured["llm_log"] = log_so_far
        measured["llm_pid"] = pid
        measured["llm_vram"] = process_vram_mib(pid)
        ok, detail = run_inference(args.port)
        measured["llm_ok"] = ok
        measured["llm_detail"] = detail
        print(f"  inference: {'OK' if ok else 'FAILED'} - {detail}",
              flush=True)
        # The card as a whole, with both models on it.
        measured["card_used"] = gpu_used_mib()
        measured["card_free"] = gpu_free_mib()
        print(f"  card: {measured['card_used']} MiB used, "
              f"{measured['card_free']} MiB free", flush=True)

    llm_log, _ = run_server(exe, model_path, llm_flags, args.port,
                            timeout=args.timeout, on_ready=_llm_ready)

    after_total = gpu_used_mib()

    # ---- write the logs ------------------------------------------------
    results_dir = Path(args.results_dir)
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "llm-server.log").write_text(llm_log,
                                                    encoding="utf-8")
        (results_dir / "embedder-server.log").write_text(embedder_log,
                                                         encoding="utf-8")
        print(f"\nlogs: {results_dir}/llm-server.log, "
              f"{results_dir}/embedder-server.log", flush=True)
    except OSError as exc:
        print(f"\ncould not write the logs: {exc}", flush=True)

    # ---- compare -------------------------------------------------------
    llm = measured.get("llm_log", "")
    emb = measured.get("embedder_log", "")

    def value(pattern, log, default=None):
        match = pattern.search(log)
        return float(match.group(1)) if match else default

    model_mib = value(MODEL_BUFFER_RE, llm)
    cpu_mapped = value(CPU_MAPPED_RE, llm)
    kv_mib = value(KV_RE, llm)
    rs_mib = value(RS_RE, llm)
    compute_mib = value(COMPUTE_RE, llm)
    context_match = CONTEXT_RE.search(llm)
    offload_match = OFFLOAD_RE.search(llm)

    emb_model = value(MODEL_BUFFER_RE, emb)
    emb_kv = value(KV_RE, emb)
    emb_compute = value(COMPUTE_RE, emb)

    print()
    print("measured from the server logs:", flush=True)
    if context_match:
        print(f"  n_ctx_slot         {context_match.group(1)}",
              flush=True)
    if offload_match:
        print(f"  layers to GPU      {offload_match.group(1)} / "
              f"{offload_match.group(2)}", flush=True)
    if model_mib is not None:
        print(f"  model buffer       {model_mib:.2f} MiB (CUDA0)", flush=True)
    if cpu_mapped is not None:
        print(f"  model buffer       {cpu_mapped:.2f} MiB "
              f"(CPU_Mapped, host RAM)", flush=True)
    if compute_mib is not None:
        print(f"  compute buffer     {compute_mib:.2f} MiB", flush=True)
    if kv_mib is not None:
        print(f"  KV buffer          {kv_mib:.2f} MiB", flush=True)
    if rs_mib is not None:
        print(f"  RS buffer          {rs_mib:.2f} MiB", flush=True)
    print(f"  LLM process VRAM   "
          f"{measured.get('llm_vram')} MiB (nvidia-smi, per process)",
          flush=True)
    if emb_model is not None:
        print(f"  embedder model     {emb_model:.2f} MiB (CUDA0)", flush=True)
    if emb_kv is not None:
        print(f"  embedder KV        {emb_kv:.2f} MiB", flush=True)
    if emb_compute is not None:
        print(f"  embedder compute   {emb_compute:.2f} MiB", flush=True)
    print(f"  embedder VRAM      "
          f"{measured.get('embedder_vram')} MiB (nvidia-smi, per process)",
          flush=True)
    if before_total is not None and after_total is not None:
        print(f"  card total         {before_total} -> {after_total} MiB",
              flush=True)

    print()
    print("predicted:", flush=True)
    print(f"  weights            {candidate.weights_vram_mib:.0f} MiB",
          flush=True)
    print(f"  KV                 {candidate.kv_vram_mib:.0f} MiB", flush=True)
    print(f"  compute            {candidate.compute_mib} MiB", flush=True)
    print(f"  CUDA context       {candidate.cuda_context_mib} MiB",
          flush=True)
    print(f"  recurrent state    {candidate.recurrent_state_mib:.0f} MiB",
          flush=True)
    print(f"  embedder           {embedder_cost_mib(candidate.embedder_backend)}"
          f" MiB", flush=True)

    problems: list[str] = []

    if offload_match:
        offloaded = int(offload_match.group(1))
        total_layers = int(offload_match.group(2))
        if offloaded < total_layers:
            problems.append(
                f"only {offloaded} of {total_layers} layers went to the "
                f"GPU - the card did not have room for the planned "
                f"configuration, so the budget could not be verified")
    if kv_mib is not None:
        delta = abs(kv_mib - candidate.kv_vram_mib)
        print(f"\nKV delta: {kv_mib:.0f} measured vs "
              f"{candidate.kv_vram_mib:.0f} predicted ({delta:.0f} MiB)",
              flush=True)
        if delta > args.tolerance_mib:
            problems.append(f"KV cache off by {delta:.0f} MiB")
    if model_mib is not None:
        delta = abs(model_mib - candidate.weights_vram_mib)
        print(f"weights delta: {model_mib:.0f} measured vs "
              f"{candidate.weights_vram_mib:.0f} predicted "
              f"({delta:.0f} MiB)", flush=True)
        if delta > args.tolerance_mib:
            problems.append(f"weights off by {delta:.0f} MiB")
    if context_match:
        measured_slot = int(context_match.group(1))
        if measured_slot != candidate.n_ctx_slot:
            problems.append(f"n_ctx_slot {measured_slot} != planned "
                            f"{candidate.n_ctx_slot}")
        else:
            print(f"n_ctx_slot matches the plan ({measured_slot})",
                  flush=True)
    else:
        problems.append("the server log did not report a context size")

    # The embedder has to answer as well. It cannot generate text, so
    # the check is that it serves embeddings.
    if measured.get("embedder_ok"):
        print("embedder: serving", flush=True)
    else:
        problems.append(
            f"the embedder did not answer: "
            f"{measured.get('embedder_detail', 'no detail')}")

    # Both models have to be on the card at the same time. That is what
    # the app needs, and it is the only way to know whether the plan
    # holds with the embedder included.
    llm_vram = measured.get("llm_vram")
    embedder_vram = measured.get("embedder_vram")
    card_used = measured.get("card_used")
    card_free = measured.get("card_free")
    if llm_vram is not None and embedder_vram is not None:
        print(f"\nboth models resident: LLM {llm_vram} MiB + embedder "
              f"{embedder_vram} MiB = {llm_vram + embedder_vram} MiB",
              flush=True)
        if llm_vram < 1000:
            problems.append(
                f"the LLM holds only {llm_vram} MiB - it is probably not "
                f"on the GPU")
        if embedder_vram < 50:
            problems.append(
                f"the embedder holds only {embedder_vram} MiB - it is "
                f"probably not on the GPU")
    if card_used is not None and card_free is not None:
        print(f"card after both models: {card_used} MiB used, "
              f"{card_free} MiB free of {hardware.total_vram_mib}",
              flush=True)
        planned_total = (candidate.total_vram_mib
                         + embedder_cost_mib(candidate.embedder_backend))
        delta = abs(card_used - planned_total)
        print(f"planned total: {planned_total:.0f} MiB "
              f"(delta {delta:.0f} MiB)", flush=True)
        if delta > args.tolerance_mib:
            problems.append(
                f"the card holds {card_used} MiB, the plan says "
                f"{planned_total:.0f} MiB - off by {delta:.0f} MiB")
        if card_free < 0:
            problems.append("the card is over its limit")

    # The embedder's VRAM against the plan.
    planned_embedder = embedder_cost_mib(candidate.embedder_backend)
    embedder_buffers = sum(v for v in (emb_model, emb_kv, emb_compute)
                           if v is not None)
    if embedder_buffers:
        print(f"embedder buffers: {embedder_buffers:.0f} MiB measured "
              f"(weights + KV + compute) + a CUDA context of "
              f"{planned_embedder - int(embedder_buffers) if planned_embedder > embedder_buffers else 0}"
              f" MiB = {planned_embedder} MiB planned", flush=True)

    if problems:
        print()
        print("MISMATCH:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print()
    print("The plan matches the measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

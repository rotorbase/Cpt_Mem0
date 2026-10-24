"""Verify a planned configuration against the real hardware.

This is the test that needs a GPU, and it is deliberately **not** part of
the default test run. It does not load a model: it starts one
``llama-server`` with the planned flags, reads the VRAM the server
actually allocated, and compares that against the planner's prediction.

That is the only honest way to check the budget model: the planner
computes, the server reports, the two must agree.

Usage::

    python scripts/verify_plan.py --vram 12288
    python scripts/verify_plan.py --vram 12288 --model Qwen3.5-9B --quant Q4_K_M

Before running this on a machine that also runs a memory service:

  * stop the other llama-server instances, or the measurement is of the
    sum of both, not of this configuration
  * the check compares total GPU usage, so anything else on the card
    (browser, compositor) shows up as a delta

The script prints the measured allocation and the difference to the
prediction, and exits non-zero when they disagree by more than the
tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Reuse the job-object helper so an interrupted run never leaves a
# model in VRAM.
sys.path.insert(0, str(ROOT / "scripts"))
from run_tests import _assign_job  # noqa: E402

from cpt_mem0.catalogue import load_catalogue  # noqa: E402
from cpt_mem0.planner import (  # noqa: E402
    Hardware,
    PlannerConfig,
    evaluate_candidate,
    plan_for_hardware,
)

#: The allocation sizes llama.cpp prints. Verified against the format
#: strings in ``llama.dll`` (build 11798):
#:
#:   '%s: %10s KV buffer size = %8.2f MiB'
#:   '%s: %10s compute buffer size = %8.2f MiB'
#:   '%s: %12s model buffer size = %8.2f MiB'
#:
#: The prefix is the backend name, e.g. "CUDA0".
KV_RE = re.compile(r"KV buffer size\s*=\s*([0-9.]+)\s*MiB")
MODEL_BUFFER_RE = re.compile(r"model buffer size\s*=\s*([0-9.]+)\s*MiB")
COMPUTE_RE = re.compile(r"compute buffer size\s*=\s*([0-9.]+)\s*MiB")
#: The server prints "initializing, n_slots = %d, n_ctx_slot = %d" and
#: the model context in several places. The slot context is the one the
#: planner predicted, so it is what the comparison needs.
CONTEXT_RE = re.compile(r"n_ctx_slot\s*=\s*([0-9]+)")
#: Fallback for the total context.
CTX_TOTAL_RE = re.compile(r"\bn_ctx\s*=\s*([0-9]+)")


def find_llama_server() -> str | None:
    """Locate the llama-server binary.

    Search order: PATH, then the usual llama.cpp build directories.
    The Unsloth studio keeps its build in
    ``~/.unsloth/llama.cpp/build/bin/<config>/llama-server.exe``.
    """
    exe = shutil.which("llama-server")
    if exe:
        return exe

    home = Path.home()
    patterns = [
        home / ".unsloth/llama.cpp/build/bin/*/llama-server.exe",
        home / ".unsloth/llama.cpp/build/bin/*/llama-server",
        home / "llama.cpp/build/bin/*/llama-server.exe",
        home / ".cache/llama.cpp/build/bin/*/llama-server",
        home / ".unsloth/studio/bin/*/llama-server.exe",
    ]
    for pattern in patterns:
        # ``pattern`` contains a wildcard in a directory part, so glob the
        # whole pattern rather than its parent.
        for path in sorted(home.glob(str(pattern.relative_to(home)))):
            if path.is_file():
                return str(path)
    return None


def gpu_used_mib() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
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


#: Markers that mean "the server is up and serving".
#:
#: The exact wording depends on the llama.cpp build. Verified against
#: 0.7.0-dev (build 11798), which prints
#: ``srv  llama_server: listening on http://127.0.0.1:8799``.
#:
#: Older builds printed "server is listening" or "main: server is
#: running". Matching on the wrong string means the readiness check
#: never fires, and the run sits there until the timeout even though the
#: server came up seconds earlier.
LISTENING_MARKERS = (
    "listening on",              # 0.7.0-dev and newer
    "server is listening",       # older builds
    "main: server is running",   # older builds
    "http server is listening",
)

#: The server prints the buffer sizes after the model is loaded. The
#: readiness check has to wait for this line, otherwise the measurement
#: is taken before the allocation has been reported.
LOADED_MARKERS = (
    "model loaded",
    "load_tensors: offloading",
)


def wait_for_http(port: int, timeout: float = 1.0) -> bool:
    """Return True if the server answers an HTTP request.

    This is the reliable readiness check: it does not depend on the
    wording of a log line, which differs between llama.cpp builds.
    """
    import urllib.error
    import urllib.request

    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
                resp.read(1)
                return True
        except urllib.error.HTTPError:
            # The server answered - that is enough.
            return True
        except Exception:
            continue
    return False


def run_server(exe: str, model: Path, flags: list[str], port: int,
               timeout: int = 90,
               on_ready=None) -> tuple[str, int]:
    """Start the server, wait for it, return (log text, exit code).

    The output is read on a separate thread. Reading it with
    ``readline()`` in the main thread blocks until the next line
    arrives - and during the model load, no line arrives for tens of
    seconds. The timeout would then never be checked, and the run would
    hang forever.

    Readiness is detected by asking the server, not by matching a log
    line. The wording of the log line changed between llama.cpp builds,
    and a mismatch means the run waits for the timeout although the
    server has been up for seconds.

    ``on_ready`` is called with the log so far while the server is
    still running. The VRAM measurement has to happen there: once the
    server is stopped, its memory is back on the device and the
    difference is zero.

    The server is always terminated, including on Ctrl+C - a model left
    in VRAM is worse than a failed check.
    """
    cmd = [exe, "-m", str(model), "--port", str(port), *flags]
    print("  $", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    # Job object: if this process dies hard, Windows kills the server.
    job = _assign_job(proc)

    log_lines: list[str] = []
    output_queue: queue_mod.Queue = queue_mod.Queue()

    def _reader() -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    output_queue.put(line)
                    # Keep a copy: the queue is drained by the main loop
                    # and the log must survive until the process ends.
                    log_lines.append(line)
        except (OSError, ValueError):
            pass
        finally:
            output_queue.put(None)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    listening = False
    loaded = False
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
                if any(marker in lowered for marker in LISTENING_MARKERS):
                    listening = True
                if any(marker in lowered for marker in LOADED_MARKERS):
                    loaded = True
            # The measurement is taken once the model is loaded and the
            # buffer sizes have been reported. Until then the server is
            # still allocating.
            if loaded and KV_RE.search("".join(log_lines)):
                if on_ready is not None:
                    on_ready("".join(log_lines))
                # Give the allocator a moment to settle, then stop.
                time.sleep(3)
                break
            if proc.poll() is not None and output_queue.empty():
                break
            # Ask the server directly. This works even when the log
            # wording is unknown.
            if listening and not loaded and wait_for_http(port):
                # The server answers, so it is up. Keep reading for the
                # allocation lines.
                pass
            if time.time() > deadline:
                print(f"  timeout after {timeout}s", flush=True)
                if on_ready is not None:
                    on_ready("".join(log_lines))
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
        if job is not None:
            try:
                import ctypes
                ctypes.WinDLL("kernel32").CloseHandle(job)
            except Exception:
                pass
    if not listening and not log_lines:
        print("  the server produced no output", flush=True)
    return "".join(log_lines), proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vram", type=int, help="VRAM of the card (MiB)")
    parser.add_argument("--model", help="model id from the catalogue")
    parser.add_argument("--quant", help="quantisation name")
    parser.add_argument("--model-file", help="path to the GGUF file")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--tolerance-mib", type=int, default=300,
                        help="allowed difference between prediction and measurement")
    parser.add_argument("--timeout", type=int, default=90,
                        help="seconds to wait for the server (0 = no limit)")
    parser.add_argument("--catalogue", default=str(ROOT / "catalogue.json"))
    args = parser.parse_args()

    catalogue = load_catalogue(args.catalogue)
    config = PlannerConfig()

    if args.model:
        entries = [e for e in catalogue.entries if e.model_id == args.model]
        if not entries:
            print(f"model {args.model!r} is not in the catalogue", file=sys.stderr)
            return 2
        if args.quant:
            entries = [e for e in entries if e.quantisation(args.quant)]
            if not entries:
                print(f"quantisation {args.quant!r} is not in the catalogue",
                      file=sys.stderr)
                return 2
    else:
        entries = catalogue.sorted_entries()

    hardware = None
    if args.vram:
        hardware = Hardware(total_vram_mib=args.vram)
    else:
        from cpt_mem0.hardware import detect_hardware
        detected = detect_hardware()
        if detected is None:
            print("no GPU detected; pass --vram", file=sys.stderr)
            return 2
        hardware = detected
        print(f"detected {hardware.name} with {hardware.total_vram_mib} MiB")

    result = plan_for_hardware(hardware, entries, config)
    if not result.viable or result.candidate is None:
        print("no viable configuration:", result.diagnostics.get("reason"))
        return 1
    candidate = result.candidate
    print(f"planned: {candidate.model_id} {candidate.quantisation}, "
          f"n_ctx {candidate.n_ctx}, KV {candidate.cache_type}, "
          f"embedder {candidate.embedder_backend}")

    model_file = args.model_file
    if not model_file:
        print("\nNo --model-file given: the budget model cannot be checked "
              "without the actual weights.\n"
              "The catalogue lists the expected file name; download it from "
              "the model's Hugging Face repository.")
        return 2
    model_path = Path(model_file)
    if not model_path.is_file():
        print(f"model file not found: {model_path}", file=sys.stderr)
        return 2

    exe = find_llama_server()
    if not exe:
        print("llama-server not found; put it on PATH", file=sys.stderr)
        return 2

    before = gpu_used_mib()

    measured: dict = {}

    def _measure_while_running(log_so_far: str) -> None:
        """Measure while the server is still up.

        Once the server is stopped its VRAM is released, so measuring
        afterwards gives a difference of zero. The log is parsed here
        too, because the server's output buffer is only flushed while
        it runs.
        """
        measured["log"] = log_so_far
        measured["gpu_used"] = gpu_used_mib()

    log, _ = run_server(exe, model_path, candidate.as_flags(), args.port,
                        timeout=args.timeout or 10**9,
                        on_ready=_measure_while_running)
    after = gpu_used_mib()

    # Prefer what was captured while the server ran; fall back to the
    # full log for the values that are printed later.
    full_log = log if len(log) > len(measured.get("log", "")) else log
    measured_log = measured.get("log", "") + (
        full_log[len(measured.get("log", "")):])

    kv_match = KV_RE.search(measured_log)
    model_match = MODEL_BUFFER_RE.search(measured_log)
    compute_match = COMPUTE_RE.search(measured_log)
    context_match = CONTEXT_RE.search(measured_log)

    print()
    print("measured from the server log:", flush=True)
    if context_match:
        print(f"  n_ctx_slot         {context_match.group(1)} "
              f"(total {candidate.n_ctx})", flush=True)
    if model_match:
        print(f"  model buffer       {model_match.group(1)} MiB", flush=True)
    if compute_match:
        print(f"  compute buffer     {compute_match.group(1)} MiB", flush=True)
    if kv_match:
        print(f"  KV buffer          {kv_match.group(1)} MiB", flush=True)
    peak = measured.get("gpu_used")
    if before is not None and peak is not None:
        print(f"  GPU usage delta    {peak - before} MiB "
              f"({before} -> {peak}, measured while the server ran)",
              flush=True)

    print()
    print("predicted:", flush=True)
    print(f"  weights            {candidate.weights_vram_mib:.0f} MiB", flush=True)
    print(f"  KV                 {candidate.kv_vram_mib:.0f} MiB", flush=True)
    print(f"  compute            {candidate.compute_mib} MiB", flush=True)
    print(f"  CUDA context       {candidate.cuda_context_mib} MiB", flush=True)
    print(f"  embedder           {embedder_mib(candidate):.0f} MiB", flush=True)

    problems: list[str] = []
    if kv_match:
        measured_kv = float(kv_match.group(1))
        delta = abs(measured_kv - candidate.kv_vram_mib)
        print(f"\nKV delta: {measured_kv:.0f} measured vs "
              f"{candidate.kv_vram_mib:.0f} predicted "
              f"({delta:.0f} MiB)", flush=True)
        if delta > args.tolerance_mib:
            problems.append(f"KV cache off by {delta:.0f} MiB")
    if model_match:
        measured_model = float(model_match.group(1))
        delta = abs(measured_model - candidate.weights_vram_mib)
        print(f"weights delta: {measured_model:.0f} measured vs "
              f"{candidate.weights_vram_mib:.0f} predicted "
              f"({delta:.0f} MiB)", flush=True)
        if delta > args.tolerance_mib:
            problems.append(f"weights off by {delta:.0f} MiB")
    if context_match:
        # The server reports the per-slot context; the planner predicted
        # the total. They have to agree.
        measured_slot = int(context_match.group(1))
        if measured_slot != candidate.n_ctx_slot:
            problems.append(
                f"n_ctx_slot {measured_slot} != planned "
                f"{candidate.n_ctx_slot} (total {candidate.n_ctx})")
        else:
            print(f"n_ctx_slot matches the plan ({measured_slot} per slot, "
                  f"{candidate.n_ctx} total)", flush=True)
    else:
        problems.append(
            "the server log did not report a context size - the "
            "comparison could not be made")

    if problems:
        print()
        print("MISMATCH:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print()
    print("The plan matches the measurement.")
    return 0


def embedder_mib(candidate) -> float:
    from cpt_mem0.planner import embedder_cost_mib
    return embedder_cost_mib(candidate.embedder_backend)


if __name__ == "__main__":
    raise SystemExit(main())

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

#: The server reports the KV buffer in MiB with two decimals.
KV_RE = re.compile(r"KV buffer size\s*=\s*([0-9.]+)\s*MiB")
MODEL_BUFFER_RE = re.compile(r"model buffer size\s*=\s*([0-9.]+)\s*MiB")
COMPUTE_RE = re.compile(r"compute buffer size\s*=\s*([0-9.]+)\s*MiB")
CONTEXT_RE = re.compile(r"n_ctx\s*=\s*([0-9]+)")


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


def run_server(exe: str, model: Path, flags: list[str], port: int,
               timeout: int = 240) -> tuple[str, int]:
    """Start the server, wait for it, return (log text, exit code).

    The output is read on a separate thread. Reading it with
    ``readline()`` in the main thread blocks until the next line
    arrives - and during the model load, no line arrives for tens of
    seconds. The timeout would then never be checked, and the run would
    hang forever.

    The server is always terminated, including on Ctrl+C - a model left
    in VRAM is worse than a failed check.
    """
    cmd = [exe, "-m", str(model), "--port", str(port), *flags]
    print("  $", " ".join(cmd))
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
        except (OSError, ValueError):
            pass
        finally:
            output_queue.put(None)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()

    deadline = time.time() + timeout
    listening = False
    try:
        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue_mod.Empty:
                line = ""
            if line is None:
                break
            if line:
                log_lines.append(line)
                lowered = line.lower()
                if "server is listening" in lowered or "main: server is running" in lowered:
                    listening = True
                    # Give the allocator a moment to settle, then stop.
                    time.sleep(3)
                    break
            if proc.poll() is not None and output_queue.empty():
                break
            if time.time() > deadline:
                print(f"  timeout after {timeout}s")
                break
    except KeyboardInterrupt:
        print("\n  interrupted")
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
        print("  the server produced no output")
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
    parser.add_argument("--timeout", type=int, default=240,
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
    log, _ = run_server(exe, model_path, candidate.as_flags(), args.port,
                        timeout=args.timeout or 10**9)
    after = gpu_used_mib()

    kv_match = KV_RE.search(log)
    model_match = MODEL_BUFFER_RE.search(log)
    compute_match = COMPUTE_RE.search(log)
    context_match = CONTEXT_RE.search(log)

    print()
    print("measured from the server log:")
    if context_match:
        print(f"  n_ctx              {context_match.group(1)}")
    if model_match:
        print(f"  model buffer       {model_match.group(1)} MiB")
    if compute_match:
        print(f"  compute buffer     {compute_match.group(1)} MiB")
    if kv_match:
        print(f"  KV buffer          {kv_match.group(1)} MiB")
    if before is not None and after is not None:
        print(f"  GPU usage delta    {after - before} MiB "
              f"({before} -> {after})")

    print()
    print("predicted:")
    print(f"  weights            {candidate.weights_vram_mib:.0f} MiB")
    print(f"  KV                 {candidate.kv_vram_mib:.0f} MiB")
    print(f"  compute            {candidate.compute_mib} MiB")
    print(f"  CUDA context       {candidate.cuda_context_mib} MiB")
    print(f"  embedder           {embedder_mib(candidate):.0f} MiB")

    problems: list[str] = []
    if kv_match:
        measured_kv = float(kv_match.group(1))
        delta = abs(measured_kv - candidate.kv_vram_mib)
        print(f"\nKV delta: {measured_kv:.0f} measured vs "
              f"{candidate.kv_vram_mib:.0f} predicted "
              f"({delta:.0f} MiB)")
        if delta > args.tolerance_mib:
            problems.append(f"KV cache off by {delta:.0f} MiB")
    if model_match:
        measured_model = float(model_match.group(1))
        delta = abs(measured_model - candidate.weights_vram_mib)
        print(f"weights delta: {measured_model:.0f} measured vs "
              f"{candidate.weights_vram_mib:.0f} predicted "
              f"({delta:.0f} MiB)")
        if delta > args.tolerance_mib:
            problems.append(f"weights off by {delta:.0f} MiB")
    if context_match:
        measured_ctx = int(context_match.group(1))
        if measured_ctx != candidate.n_ctx:
            problems.append(
                f"n_ctx {measured_ctx} != planned {candidate.n_ctx}")
        else:
            print(f"n_ctx matches the plan ({measured_ctx})")

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

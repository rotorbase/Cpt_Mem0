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
#: The prefix is the backend name, e.g. "CUDA0" or "CPU_Mapped". The
#: patterns that name the device live further down, together with the
#: reason they have to.
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


def peak_gpu_usage(samples: int = 8,
                   interval: float = 0.5) -> int | None:
    """The highest VRAM usage seen while the server is running.

    A single reading of ``nvidia-smi`` is not the allocation: the
    server keeps allocating after it reports the model as loaded, and a
    single sample caught a run at 3804 MiB while the model alone held
    8055 MiB on the card. Sampling several times and keeping the
    highest value measures the peak instead of a moment.

    The sampling stops early when the value has not changed for two
    rounds - there is no point in waiting for a number that will not
    move.
    """
    peak = gpu_used_mib()
    if peak is None:
        return None
    stable = 0
    previous = peak
    for _ in range(samples):
        time.sleep(interval)
        current = gpu_used_mib()
        if current is None:
            break
        if current > peak:
            peak = current
        if current == previous:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        previous = current
    return peak


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

#: The server only prints the buffer sizes at debug level. Without this
#: flag the run waits for a line that never arrives, and the check ends
#: in a timeout although the server came up in seconds.
#:
#: Verified on llama.cpp 0.7.0-dev (build 11798): with the default
#: verbosity the log contains "model loaded" but no "buffer size" line
#: at all.
VERBOSE_FLAG = "--verbose"

#: The server reports one buffer line per device, and the order in the
#: log is not the order of importance:
#:
#: ==========================  ==================  =============
#: Line                       Device               Counts as VRAM
#: ==========================  ==================  =============
#: ``CPU_Mapped model buffer``  host, memory-mapped  no
#: ``CUDA0 model buffer``       the GPU              yes
#: ``CUDA0 KV buffer``          the GPU              yes
#: ``CUDA_Host compute``        the host             no
#: ``CUDA0 compute``            the GPU              yes
#: ==========================  ==================  =============
#:
#: A first version of these patterns took the first match in the log,
#: which is the ``CPU_Mapped`` line. It reported 1030 MiB of weights
#: for a model whose GPU allocation was 8087 MiB. The patterns now
#: name the device.
MODEL_BUFFER_RE = re.compile(
    r"CUDA0\s+model buffer size\s*=\s*([0-9.]+)\s*MiB")
KV_RE = re.compile(
    r"CUDA0\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB")
COMPUTE_RE = re.compile(
    r"CUDA0\s+compute buffer size\s*=\s*([0-9.]+)\s*MiB")

#: The memory-mapped part of the model stays in host RAM. It is not
#: VRAM, but it is part of the model's footprint, so it is reported
#: separately.
CPU_MAPPED_RE = re.compile(
    r"CPU_Mapped\s+model buffer size\s*=\s*([0-9.]+)\s*MiB")

#: How many layers went to the GPU. When this is fewer than the model
#: has, the server had to keep the rest in host RAM - and the planned
#: VRAM does not apply.
OFFLOAD_RE = re.compile(
    r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")


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


def run_inference(port: int, timeout: float = 60.0) -> tuple[bool, str]:
    """Ask the server to generate something and report what came back.

    A server that loads a model and answers /health has not proven that
    it can run a prompt. The first version of this script only checked
    readiness, so a run could report "the model is in VRAM" without any
    token ever being generated.

    The request is deliberately small - one short completion. It is a
    liveness check, not a benchmark.

    Returns ``(ok, detail)``. ``detail`` carries the generated text or
    the error, so a failure says what went wrong instead of just "no".

    An embedding-only server answers a completion request with
    ``"the current context does not support logits computation"``. That
    is not a failure of the server - it is what such a server does. The
    result is reported as ``(True, "not applicable: ...")`` so the
    caller can tell it apart from a real failure.
    """
    import json
    import urllib.error
    import urllib.request

    payload = json.dumps({
        "prompt": "Reply with the single word: ready",
        "n_predict": 8,
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")

    def _attempt() -> tuple[bool, str, bool]:
        """One request. Returns (ok, detail, retryable)."""
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
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
                # The server is still loading the model. The log says
                # "model loaded" before it accepts requests, so the first
                # attempt can arrive too early.
                return False, f"HTTP 503: {raw[:120]}", True
            return False, f"HTTP {exc.code}: {exc.reason} - {raw[:200]}", False
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", True

        text = (body.get("content") or "").strip()
        if not text:
            return False, "the server answered with an empty completion", False
        return True, text, False

    deadline = time.time() + timeout
    last_detail = "no attempt"
    while True:
        ok, detail, retryable = _attempt()
        if ok:
            return True, detail
        last_detail = detail
        if not retryable or time.time() >= deadline:
            return False, last_detail
        time.sleep(1.0)


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
            if loaded and not KV_RE.search("".join(log_lines)):
                # The model is in VRAM and the server answers requests,
                # but it did not print a KV buffer line. Waiting for a
                # line that never comes means the run ends in a timeout
                # with a working server and no measurement.
                #
                # The KV cache is optional in the log: an embedding model
                # has none, and a build may report it differently. What
                # matters for the budget is the total VRAM, which is
                # measured directly.
                if wait_for_http(port):
                    if on_ready is not None:
                        on_ready("".join(log_lines))
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
    parser.add_argument("--results-dir",
                        help="where to write the server log "
                             "(default: <repo>/results)")
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

        The VRAM is sampled several times and the highest value is
        kept. A single reading is not enough: the allocation continues
        after the server reports the model as loaded, and one run
        measured 3804 MiB while the model alone had 8055 MiB on the
        card.
        """
        measured["log"] = log_so_far
        measured["gpu_used"] = peak_gpu_usage()
        # Prove the server can actually generate. A server that answers
        # /health has not proven that.
        ok, detail = run_inference(args.port)
        measured["inference_ok"] = ok
        measured["inference_detail"] = detail
        print(f"  inference: {'OK' if ok else 'FAILED'} - {detail}",
              flush=True)

    # The server only reports the buffer sizes at debug level.
    server_flags = candidate.as_flags() + [VERBOSE_FLAG]

    log, _ = run_server(exe, model_path, server_flags, args.port,
                        timeout=args.timeout or 10**9,
                        on_ready=_measure_while_running)
    after = gpu_used_mib()

    # Write the server log next to the other results. Without it the
    # numbers below cannot be checked afterwards - and the first run
    # that reported a wrong weight figure could not be diagnosed
    # because the log was gone.
    results_dir = Path(args.results_dir) if args.results_dir else (
        Path(__file__).resolve().parent.parent / "results")
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / "server.log"
        log_path.write_text(log, encoding="utf-8")
        print(f"\nserver log: {log_path}", flush=True)
    except OSError as exc:
        print(f"\ncould not write the server log: {exc}", flush=True)

    # Prefer what was captured while the server ran; fall back to the
    # full log for the values that are printed later.
    full_log = log if len(log) > len(measured.get("log", "")) else log
    measured_log = measured.get("log", "") + (
        full_log[len(measured.get("log", "")):])

    kv_match = KV_RE.search(measured_log)
    model_match = MODEL_BUFFER_RE.search(measured_log)
    compute_match = COMPUTE_RE.search(measured_log)
    context_match = CONTEXT_RE.search(measured_log)
    cpu_mapped_match = CPU_MAPPED_RE.search(measured_log)
    offload_match = OFFLOAD_RE.search(measured_log)

    print()
    print("measured from the server log:", flush=True)
    if context_match:
        print(f"  n_ctx_slot         {context_match.group(1)} "
              f"(the -c value, per slot)", flush=True)
    if offload_match:
        offloaded = int(offload_match.group(1))
        total_layers = int(offload_match.group(2))
        print(f"  layers to GPU      {offloaded} / {total_layers}", flush=True)
    if model_match:
        print(f"  model buffer       {model_match.group(1)} MiB "
              f"(CUDA0)", flush=True)
    if cpu_mapped_match:
        print(f"  model buffer       {cpu_mapped_match.group(1)} MiB "
              f"(CPU_Mapped, host RAM - not VRAM)", flush=True)
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
    if offload_match:
        offloaded = int(offload_match.group(1))
        total_layers = int(offload_match.group(2))
        if offloaded < total_layers:
            # The server had to keep layers in host RAM. The planned
            # VRAM was not available - probably because something else
            # was on the card. Every comparison below is against a
            # configuration the planner never intended, so the budget
            # cannot be checked here.
            problems.append(
                f"only {offloaded} of {total_layers} layers went to the "
                f"GPU - the card did not have room for the planned "
                f"configuration, so the budget could not be verified")
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
        # The server reports the per-slot context, which with
        # --kv-unified is the -c value itself. The planner predicts the
        # same number, so they have to agree exactly.
        measured_slot = int(context_match.group(1))
        if measured_slot != candidate.n_ctx_slot:
            problems.append(
                f"n_ctx_slot {measured_slot} != planned "
                f"{candidate.n_ctx_slot}")
        else:
            print(f"n_ctx_slot matches the plan ({measured_slot})", flush=True)
    else:
        problems.append(
            "the server log did not report a context size - the "
            "comparison could not be made")

    if "inference_ok" in measured:
        if measured["inference_ok"]:
            print(f"inference: the server generated a response",
                  flush=True)
        else:
            problems.append(
                f"the server loaded the model but did not generate "
                f"anything: {measured.get('inference_detail', 'no detail')}")
    else:
        problems.append(
            "no inference was attempted - the server never reached the "
            "point where a request could be sent")

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

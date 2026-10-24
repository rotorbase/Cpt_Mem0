"""Test entry point.

Two suites, deliberately separated:

**logic** (default)
    Pure arithmetic and file parsing. No GPU, no network, no model, no
    server. Runs in well under a second and is safe to run at any time.

**hardware** (``--hardware``)
    Starts a real ``llama-server`` with the planned flags and compares
    the allocation against the prediction. Needs a GPU and a model file.
    It loads a model into VRAM, so do not run it while another memory
    service is using the card - the measurement would be of both.

What it touches
---------------

Nothing outside this repository and the results directory:

* the logic suite writes ``.pytest_cache/`` and the result files
* no configuration file, no service, no environment variable is changed
* the hardware suite starts one ``llama-server`` and stops it again

Results are written to ``results/`` (override with ``--results-dir``):
``pytest-junit.xml`` for CI, ``results.json`` for humans, and the
console output in ``logic.log`` / ``hardware.log``.

Usage::

    python scripts/run_tests.py
    python scripts/run_tests.py --hardware --vram 12288 \\
        --model-file ~/.cache/huggingface/hub/Qwen3.5-9B-Q4_K_M.gguf

Interrupting
------------

Ctrl+C stops everything. The logic suite dies immediately; the hardware
suite terminates the server it started, so no model is left in VRAM.
``--timeout`` puts an upper bound on each phase.
"""

from __future__ import annotations

import argparse
import json
import os
import queue as queue_mod
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _child_env() -> dict:
    """Environment for the child processes.

    ``PYTHONPATH`` points at ``src/`` so the package is importable
    without installing it. The existing value is preserved, not
    replaced - the tool must not disturb the caller's environment.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(SRC)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _terminate(proc: subprocess.Popen) -> None:
    """Stop a child process, escalating if it does not go."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        pass


def _assign_job(proc: subprocess.Popen) -> int | None:
    """Put the child in a Windows job object that dies with the parent.

    Without this, a hard interrupt of the parent (Ctrl+C in some
    terminals, a console close) leaves the child running: the Python
    cleanup never executes, and a model stays in VRAM.

    A job object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` makes the
    operating system terminate the whole tree when the handle closes -
    including when the parent is killed rather than exiting cleanly.

    Returns the job handle, or ``None`` if the platform does not support
    it.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("ReadOperationCount", "WriteOperationCount",
                         "OtherOperationCount", "ReadTransferCount",
                         "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation",
                 JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            kernel32.CloseHandle(job)
            return None

        proc._handle  # noqa: B018 - ensure the process handle exists
        assigned = kernel32.AssignProcessToJobObject(job, proc._handle)
        if not assigned:
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def run_phase(name: str, cmd: list[str], cwd: Path,
              timeout: int | None, log_path: Path | None) -> dict:
    """Run one phase, capture its output, return a result record.

    The output is read on a separate thread. Reading it in the main
    thread would block in ``readline()``, and the timeout would only be
    checked between output lines - a process that hangs silently would
    then never be interrupted.
    """
    print(f"=== {name} ===")
    print("  $", " ".join(cmd))
    started = time.perf_counter()
    lines: list[str] = []
    status = "ok"
    interrupted = False

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=_child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace")
    except OSError as exc:
        print(f"  cannot start: {exc}")
        return {"phase": name, "status": "error", "returncode": -1,
                "duration_s": 0.0, "error": str(exc), "output": []}

    # The job handle stays open for the lifetime of this function. When it
    # goes out of scope - including when the parent is killed - Windows
    # terminates the whole child tree.
    job = _assign_job(proc)

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

    deadline = time.time() + timeout if timeout else None
    eof = False
    try:
        while True:
            try:
                line = output_queue.get(timeout=0.1)
            except queue_mod.Empty:
                line = ""
            if line is None:
                eof = True
            elif line:
                lines.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            if eof and proc.poll() is not None:
                break
            if proc.poll() is not None and output_queue.empty():
                break
            if deadline and time.time() > deadline:
                status = "timeout"
                print(f"\n  timeout after {timeout}s - terminating")
                _terminate(proc)
                break
    except KeyboardInterrupt:
        interrupted = True
        status = "interrupted"
        print("\n  interrupted - terminating")
        _terminate(proc)
    finally:
        if proc.poll() is None:
            _terminate(proc)
        thread.join(timeout=5)
        if job is not None:
            try:
                import ctypes
                ctypes.WinDLL("kernel32").CloseHandle(job)
            except Exception:
                pass

    duration = time.perf_counter() - started
    returncode = proc.returncode if proc.returncode is not None else -1
    if status == "ok" and returncode != 0:
        status = "failed"
    if interrupted:
        status = "interrupted"

    if log_path is not None:
        log_path.write_text("".join(lines), encoding="utf-8")

    return {"phase": name, "status": status, "returncode": returncode,
            "duration_s": round(duration, 3), "output": lines}


def parse_pytest_summary(output: list[str]) -> dict:
    """Extract the pass/fail counts from pytest's output."""
    passed = failed = skipped = 0
    for line in output:
        text = line.strip()
        m = re.search(r"(\d+) passed", text)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", text)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) skipped", text)
        if m:
            skipped = int(m.group(1))
    return {"passed": passed, "failed": failed, "skipped": skipped}


def run_logic(args: argparse.Namespace, results_dir: Path) -> dict:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    junit = results_dir / "pytest-junit.xml"
    cmd += ["--junitxml", str(junit)]
    result = run_phase("logic tests (no GPU)", cmd, ROOT,
                        args.timeout, results_dir / "logic.log")
    result["tests"] = parse_pytest_summary(result["output"])
    result["junit_xml"] = str(junit)
    return result


def run_hardware(args: argparse.Namespace, results_dir: Path) -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "verify_plan.py")]
    if args.vram:
        cmd += ["--vram", str(args.vram)]
    if args.model:
        cmd += ["--model", args.model]
    if args.quant:
        cmd += ["--quant", args.quant]
    if args.model_file:
        cmd += ["--model-file", args.model_file]
    if args.tolerance_mib is not None:
        cmd += ["--tolerance-mib", str(args.tolerance_mib)]
    if args.timeout:
        cmd += ["--timeout", str(args.timeout)]
    result = run_phase("hardware verification (loads a model)", cmd, ROOT,
                        args.timeout, results_dir / "hardware.log")
    return result


def write_results(results_dir: Path, phases: list[dict],
                  status: str, duration: float) -> Path:
    path = results_dir / "results.json"
    payload = {
        "tool": "cpt-mem0",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration, 3),
        "status": status,
        "phases": [
            {k: v for k, v in phase.items() if k != "output"}
            for phase in phases
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", action="store_true",
                        help="also run the GPU verification")
    parser.add_argument("--vram", type=int)
    parser.add_argument("--model")
    parser.add_argument("--quant")
    parser.add_argument("--model-file")
    parser.add_argument("--tolerance-mib", type=int)
    parser.add_argument("--timeout", type=int,
                        help="seconds per phase (0 = no limit)")
    parser.add_argument("--results-dir",
                        help="where to write the results "
                             "(default: <repo>/results)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    phases: list[dict] = []

    logic = run_logic(args, results_dir)
    phases.append(logic)

    if logic["status"] != "ok":
        duration = time.perf_counter() - started
        write_results(results_dir, phases, logic["status"], duration)
        print()
        print(f"logic tests failed ({logic['status']}) - "
              "hardware verification skipped")
        print(f"results: {results_dir}")
        return logic["returncode"] or 1

    if args.hardware:
        print()
        phases.append(run_hardware(args, results_dir))
    else:
        print()
        print("Hardware verification skipped. Run with --hardware on a card "
              "that is free of other models.")

    duration = time.perf_counter() - started
    failed = [p for p in phases if p["status"] != "ok"]
    status = "failed" if failed else "ok"
    results_path = write_results(results_dir, phases, status, duration)

    print()
    print("--- summary ---")
    for phase in phases:
        tests = phase.get("tests")
        detail = ""
        if tests:
            detail = (f"  ({tests['passed']} passed, {tests['failed']} failed, "
                      f"{tests['skipped']} skipped)")
        print(f"  {phase['phase']:38} {phase['status']:10} "
              f"{phase['duration_s']:6.2f}s{detail}")
    print(f"  total {duration:.2f}s")
    print()
    print(f"results: {results_path}")
    print(f"  {results_dir / 'pytest-junit.xml'} (CI format)")
    print(f"  {results_dir / 'results.json'}")
    print(f"  {results_dir / 'logic.log'}, {results_dir / 'hardware.log'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

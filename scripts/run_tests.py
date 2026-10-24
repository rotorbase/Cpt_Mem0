"""Test entry point.

Two suites, deliberately separated:

**logic** (default)
    Pure arithmetic and file parsing. No GPU, no network, no model, no
    server. Runs in under a second and is safe to run at any time.

**hardware** (``--hardware``)
    Starts a real ``llama-server`` with the planned flags and compares
    the allocation against the prediction. Needs a GPU and a model file.
    It loads a model into VRAM, so do not run it while another memory
    service is using the card - the measurement would be of both.

Usage::

    python scripts/run_tests.py
    python scripts/run_tests.py --hardware --vram 12288 \\
        --model-file ~/.cache/huggingface/hub/Qwen3.5-9B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_logic() -> int:
    print("=== logic tests (no GPU) ===")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT)
    return proc.returncode


def run_hardware(args: argparse.Namespace) -> int:
    print("=== hardware verification (loads a model) ===")
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
    proc = subprocess.run(cmd, cwd=ROOT)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hardware", action="store_true",
                        help="also run the GPU verification")
    parser.add_argument("--vram", type=int)
    parser.add_argument("--model")
    parser.add_argument("--quant")
    parser.add_argument("--model-file")
    parser.add_argument("--tolerance-mib", type=int)
    args = parser.parse_args()

    status = run_logic()
    if status != 0:
        return status
    if not args.hardware:
        print()
        print("Hardware verification skipped. Run with --hardware on a card "
              "that is free of other models.")
        return 0
    print()
    return run_hardware(args)


if __name__ == "__main__":
    raise SystemExit(main())

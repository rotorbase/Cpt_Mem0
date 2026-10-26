"""Hardware detection.

Read-only. Nothing here starts a process, loads a model or allocates
VRAM. The only commands invoked are ``nvidia-smi`` and its ROCm
equivalent, both read-only queries.

If no GPU is found the caller decides what to do - a CPU-only fallback
is a legitimate configuration for the embedder, but the LLM still needs
a GPU to be useful.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from . import proc
from .planner import Hardware


@dataclass(frozen=True)
class GpuInfo:
    """Raw information as reported by the vendor tool."""

    vendor: str
    name: str
    total_vram_mib: int
    driver_version: str = ""
    tool: str = ""


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        done = proc.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 1, ""
    return done.returncode, done.stdout


def _parse_nvidia_smi() -> GpuInfo | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    code, out = _run([
        exe,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if code != 0 or not out.strip():
        return None
    first = out.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        total = int(float(parts[1]))
    except ValueError:
        return None
    return GpuInfo(
        vendor="nvidia",
        name=parts[0],
        total_vram_mib=total,
        driver_version=parts[2] if len(parts) > 2 else "",
        tool=exe,
    )


def _parse_rocm_smi() -> GpuInfo | None:
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    code, out = _run([exe, "--showmeminfo", "vram", "--csv"])
    if code != 0 or not out.strip():
        return None
    total_mib = 0
    name = ""
    for line in out.splitlines():
        if "GPU" in line and "VRAM TOTAL" in line.upper():
            for token in re.findall(r"(\d+)", line):
                value = int(token)
                if value > total_mib:
                    total_mib = value
        if "Card series" in line or "Card model" in line:
            name = name or line.split(",")[-1].strip()
    if total_mib <= 0:
        return None
    return GpuInfo(vendor="amd", name=name or "AMD GPU",
                   total_vram_mib=total_mib, tool=exe)


def detect_gpu() -> GpuInfo | None:
    """Return information about the primary GPU, or ``None``."""
    return _parse_nvidia_smi() or _parse_rocm_smi()


def hardware_from_gpu(info: GpuInfo) -> Hardware:
    return Hardware(
        total_vram_mib=info.total_vram_mib,
        name=info.name,
        vendor=info.vendor,
        driver_version=info.driver_version,
    )


def detect_hardware() -> Hardware | None:
    """Detect the GPU and return it as planner input."""
    info = detect_gpu()
    return hardware_from_gpu(info) if info else None


def current_vram_mib() -> int | None:
    """Current VRAM usage in MiB, or ``None`` if it cannot be read.

    Used for health checks. A model that silently fell back to CPU shows
    up here as unexpectedly low usage - this is the measurement the
    project's whole premise rests on.
    """
    info = detect_gpu()
    if info is None:
        return None
    exe = info.tool
    code, out = _run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if code != 0 or not out.strip():
        return None
    try:
        return int(float(out.strip().splitlines()[0]))
    except ValueError:
        return None

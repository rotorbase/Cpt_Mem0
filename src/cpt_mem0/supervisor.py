"""Supervising the llama-server child processes.

The daemon owns up to two of them: the language model and the
embedder. They are child processes, not services - when the daemon
dies, they die with it. That is the point: the vector store has an
exclusive lock, and a store whose owner is gone is a store nobody can
open.

**A crash is not fatal.** A server that dies is restarted with an
exponential backoff, up to five times. After that the daemon gives
up on that server and reports it.

**The readiness check is not the health check.** A server that answers
``/v1/models`` is up. A server that computes on the GPU is a different
question, and only the VRAM answers it. The daemon checks both and
says which one failed.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ServerError(RuntimeError):
    """The server cannot be started or is not usable."""


@dataclass
class ServerSpec:
    """What a server needs to run."""

    name: str
    binary: Path
    model: Path
    port: int
    flags: list[str] = field(default_factory=list)
    embeddings: bool = False
    log_path: Path | None = None

    def command(self) -> list[str]:
        """The command line.

        The verbose flag is always on. Without it the server does not
        print the buffer lines, and those lines are the only way to
        tell that the model is on the GPU - the per-process reading
        from nvidia-smi is N/A on Windows with the WDDM driver.
        """
        flags = list(self.flags)
        if VERBOSE_FLAG not in flags:
            flags.append(VERBOSE_FLAG)
        command = [str(self.binary), "-m", str(self.model),
                   "--port", str(self.port), *flags]
        return command


@dataclass
class ServerState:
    """The state of a supervised server."""

    spec: ServerSpec
    process: subprocess.Popen | None = None
    restarts: int = 0
    last_error: str = ""
    ready: bool = False
    started_at: float = 0.0

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def uptime(self) -> float:
        if not self.started_at:
            return 0.0
        return time.time() - self.started_at


def http_ready(port: int, timeout: float = 1.0) -> bool:
    """Whether the server answers an HTTP request."""
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


def wait_until_ready(port: int, timeout: float = 180.0,
                     process: subprocess.Popen | None = None) -> bool:
    """Wait until the server answers.

    The process is checked as well: a server that died during the
    model load does not become ready.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if http_ready(port):
            return True
        time.sleep(0.5)
    return False


class Supervisor:
    """Owns the child processes."""

    #: How many times a crashed server is restarted.
    max_restarts = 5

    #: The backoff base in seconds: 1, 2, 4, 8, 16.
    backoff_base = 1.0

    def __init__(self, specs: list[ServerSpec],
                 on_event: Callable[[str, str], None] | None = None) -> None:
        self.states = {spec.name: ServerState(spec=spec) for spec in specs}
        self._on_event = on_event

    def _event(self, name: str, message: str) -> None:
        if self._on_event is not None:
            self._on_event(name, message)

    def start(self, name: str, timeout: float = 180.0) -> ServerState:
        """Start a server and wait until it is ready.

        The output is read on a separate thread. Reading it with
        ``readline()`` in the main thread blocks until the next line
        arrives, and during the model load no line arrives for tens of
        seconds - the timeout would never be checked.
        """
        state = self.states[name]
        spec = state.spec

        if not spec.binary.is_file():
            raise ServerError(
                f"the llama-server binary is missing: {spec.binary}. "
                f"Run 'cpt-mem0 init' to install the runtime.")
        if not spec.model.is_file():
            raise ServerError(
                f"the model file is missing: {spec.model}")

        command = spec.command()
        self._event(name, f"starting: {' '.join(command)}")

        log_handle = None
        if spec.log_path is not None:
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = spec.log_path.open("w", encoding="utf-8",
                                            errors="replace")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                cwd=str(spec.binary.parent),
            )
        except OSError as exc:
            if log_handle is not None:
                log_handle.close()
            raise ServerError(
                f"the server could not be started: {exc}") from exc

        state.process = process
        state.started_at = time.time()
        state.ready = False
        state.last_error = ""

        lines: list[str] = []
        output: queue.Queue = queue.Queue()

        def reader() -> None:
            """Read the output and write it to the log.

            The log is written here, not through a redirect: a
            redirect would replace the pipe and there would be nothing
            to read the readiness marker from.
            """
            try:
                if process.stdout is not None:
                    for line in process.stdout:
                        lines.append(line)
                        if log_handle is not None:
                            log_handle.write(line)
                            log_handle.flush()
                        output.put(line)
            except (OSError, ValueError):
                pass
            finally:
                if log_handle is not None:
                    log_handle.close()
                output.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

        # The model load takes a while. The loop watches the output for
        # the readiness marker and the timeout at the same time.
        deadline = time.time() + timeout
        while True:
            try:
                line = output.get(timeout=0.2)
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line:
                lowered = line.lower()
                if "model loaded" in lowered or "listening on" in lowered:
                    state.ready = True
            if state.ready and http_ready(spec.port):
                self._event(name, f"ready on port {spec.port}")
                return state
            if process.poll() is not None and output.empty():
                break
            if time.time() > deadline:
                state.last_error = f"timeout after {timeout:.0f}s"
                self._event(name, state.last_error)
                self.stop(name)
                raise ServerError(
                    f"the {name} server did not become ready in "
                    f"{timeout:.0f}s. The log is at {spec.log_path}"
                    if spec.log_path else
                    f"the {name} server did not become ready in "
                    f"{timeout:.0f}s")

        exit_code = process.poll()
        state.last_error = f"the server exited with code {exit_code}"
        self._event(name, state.last_error)
        tail = "".join(lines[-15:])
        raise ServerError(
            f"the {name} server exited with code {exit_code}.\n"
            f"{tail}")

    def stop(self, name: str, timeout: float = 20.0) -> None:
        """Stop a server.

        It is terminated first and killed only if it does not go. A
        server that is killed leaves its GPU memory behind until the
        driver reclaims it.
        """
        state = self.states.get(name)
        if state is None or state.process is None:
            return
        process = state.process
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
            except OSError:
                pass
        state.process = None
        state.ready = False
        self._event(name, "stopped")

    def stop_all(self) -> None:
        """Stop every server."""
        for name in list(self.states):
            self.stop(name)

    def restart(self, name: str, timeout: float = 180.0) -> ServerState:
        """Restart a server after a crash.

        The backoff grows with the number of restarts. After
        ``max_restarts`` the server is left down and the failure is
        reported.
        """
        state = self.states[name]
        if state.restarts >= self.max_restarts:
            raise ServerError(
                f"the {name} server crashed {state.restarts} times and "
                f"is not restarted again. Look at {state.spec.log_path}.")

        delay = self.backoff_base * (2 ** state.restarts)
        state.restarts += 1
        self._event(name, f"restarting in {delay:.0f}s "
                           f"(attempt {state.restarts} of {self.max_restarts})")
        time.sleep(delay)
        return self.start(name, timeout=timeout)

    def status(self) -> list[dict[str, Any]]:
        """The state of every server."""
        result = []
        for name, state in self.states.items():
            result.append({
                "name": name,
                "port": state.spec.port,
                "running": state.running,
                "ready": state.ready,
                "restarts": state.restarts,
                "uptime": round(state.uptime, 1),
                "pid": state.process.pid if state.process else None,
                "embeddings": state.spec.embeddings,
                "last_error": state.last_error,
            })
        return result


def gpu_used_mib() -> int | None:
    """The VRAM used on the card.

    This is the check that catches the CPU fallback. A server that
    answers and computes on the CPU holds almost nothing on the card.
    """
    import shutil

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(float(result.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return None


def gpu_free_mib() -> int | None:
    """The VRAM still free on the card.

    This is the number that decides whether a run fits. A run that
    starts into a full card fails in a way that looks like a bug, so
    the check comes before the start.
    """
    import shutil

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    try:
        return int(float(result.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return None


def process_vram_mib(pid: int) -> int | None:
    """The VRAM of one process.

    Reading it per process is what makes the measurement independent of
    what else runs on the card.
    """
    import shutil

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == pid:
                return int(float(parts[1]))
        except ValueError:
            continue
    return None


#: The buffer lines the server prints. The device is part of the
#: pattern: the host line comes first and is not VRAM.
MODEL_BUFFER_RE = re.compile(
    r"CUDA0\s+model buffer size\s*=\s*([0-9.]+)\s*MiB")
CPU_MAPPED_RE = re.compile(
    r"CPU_Mapped\s+model buffer size\s*=\s*([0-9.]+)\s*MiB")
KV_BUFFER_RE = re.compile(
    r"CUDA0\s+KV buffer size\s*=\s*([0-9.]+)\s*MiB")
RS_BUFFER_RE = re.compile(
    r"CUDA0\s+RS buffer size\s*=\s*([0-9.]+)\s*MiB")
COMPUTE_BUFFER_RE = re.compile(
    r"CUDA0\s+compute buffer size\s*=\s*([0-9.]+)\s*MiB")
OFFLOAD_RE = re.compile(
    r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
DEVICE_RE = re.compile(r"using device\s+(\S+)")

#: The flags that make the server print the buffer lines. Without
#: --verbose they are not in the log at all.
VERBOSE_FLAG = "--verbose"


def gpu_buffers(log_path: Path | None) -> dict[str, float | None]:
    """What the server log says about the VRAM it uses.

    **This is the measurement that works everywhere.** The per-process
    reading from `nvidia-smi` does not: on Windows with the WDDM
    driver it returns N/A for every process, because the card is
    managed by the display driver and not by the NVIDIA driver.

    The server log has the truth either way. It says which device it
    uses, how many layers went to the GPU, and how big each buffer
    is.
    """
    result: dict[str, float | None] = {
        "model_mib": None,
        "cpu_mapped_mib": None,
        "kv_mib": None,
        "rs_mib": None,
        "compute_mib": None,
        "offloaded_layers": None,
        "total_layers": None,
        "device": None,
    }
    if log_path is None or not log_path.is_file():
        return result

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    for line in text.splitlines():
        match = MODEL_BUFFER_RE.search(line)
        if match:
            result["model_mib"] = float(match.group(1))
        match = CPU_MAPPED_RE.search(line)
        if match:
            result["cpu_mapped_mib"] = float(match.group(1))
        match = KV_BUFFER_RE.search(line)
        if match:
            result["kv_mib"] = float(match.group(1))
        match = RS_BUFFER_RE.search(line)
        if match:
            result["rs_mib"] = float(match.group(1))
        match = COMPUTE_BUFFER_RE.search(line)
        if match:
            result["compute_mib"] = float(match.group(1))
        match = OFFLOAD_RE.search(line)
        if match:
            result["offloaded_layers"] = int(match.group(1))
            result["total_layers"] = int(match.group(2))
        match = DEVICE_RE.search(line)
        if match:
            result["device"] = match.group(1)

    return result


def check_gpu_placement(supervisor: Supervisor,
                         threshold_mib: int = 1000) -> list[str]:
    """Warnings about servers that are not on the GPU.

    A server that holds less than the threshold is almost certainly
    computing on the CPU. That failure is silent from the outside: the
    port is open, the model answers, and the speed is a sixth of what
    it should be.

    The measurement comes from the server log, not from nvidia-smi:
    the per-process reading is N/A on Windows with the WDDM driver.
    """
    warnings: list[str] = []
    for entry in supervisor.status():
        if not entry["running"]:
            continue
        spec = supervisor.states[entry["name"]].spec
        buffers = gpu_buffers(spec.log_path)

        if buffers["offloaded_layers"] is not None and \
                buffers["total_layers"] is not None:
            offloaded = buffers["offloaded_layers"]
            total = buffers["total_layers"]
            if offloaded < total:
                warnings.append(
                    f"the {entry['name']} server put only {offloaded} of "
                    f"{total} layers on the GPU - it is running partly on "
                    f"the CPU")

        model_mib = buffers["model_mib"]
        if model_mib is not None and model_mib < threshold_mib:
            warnings.append(
                f"the {entry['name']} server holds only {model_mib:.0f} MiB "
                f"of GPU memory - it is probably not on the GPU. The CUDA "
                f"runtime libraries are most likely missing from the "
                f"runtime directory.")
        elif model_mib is not None:
            # The per-process reading is not available everywhere, so
            # the log is the source.
            continue

    return warnings


def write_json(path: Path, data: Any) -> None:
    """Write a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str),
                    encoding="utf-8")

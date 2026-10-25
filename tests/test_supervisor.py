"""Tests for the supervisor.

**The models must not open a console window.** llama-server.exe is a
console program. Without ``CREATE_NO_WINDOW`` Windows opens a console
for it every time a model starts - and the app is a window, not a
terminal. The user sees a black box blinking on the desktop every
time the big switch is pressed.

The test does not start a server. It checks the flag that goes into
the call.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.supervisor import ServerSpec, Supervisor  # noqa: E402


class _FakeProcess:
    pid = 1234
    stdout = None
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass


def _spec(name="llm", binary="llama-server.exe") -> ServerSpec:
    return ServerSpec(
        name=name,
        binary=Path(binary),
        model=Path("m.gguf"),
        port=8766,
        log_path=None,
    )


def _existing_spec(tmp_path) -> ServerSpec:
    """A spec whose binary exists.

    The supervisor checks the binary before it starts anything - a
    missing binary is a clear error, not a crash.
    """
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    return ServerSpec(
        name="llm",
        binary=binary,
        model=model,
        port=8766,
        log_path=None,
    )


class TestNoConsoleWindow:
    def test_the_flag_is_used_on_windows(self, monkeypatch, tmp_path):
        """The creation flag goes into the Popen call."""
        calls = {}

        def fake_popen(command, **kwargs):
            calls.update(kwargs)
            calls["command"] = command
            return _FakeProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("cpt_mem0.supervisor.IS_WINDOWS", True)

        supervisor = Supervisor([_existing_spec(tmp_path)])
        try:
            supervisor.start("llm", timeout=1)
        except Exception:
            # The fake process cannot answer the readiness check.
            # That does not matter - the call is what is tested.
            pass

        assert "creationflags" in calls, "no creation flag was set"
        assert calls["creationflags"] == subprocess.CREATE_NO_WINDOW

    def test_the_flag_is_omitted_elsewhere(self, monkeypatch, tmp_path):
        """On Linux and macOS the flag does not exist."""
        calls = {}

        def fake_popen(command, **kwargs):
            calls.update(kwargs)
            return _FakeProcess()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("cpt_mem0.supervisor.IS_WINDOWS", False)

        supervisor = Supervisor([_existing_spec(tmp_path)])
        try:
            supervisor.start("llm", timeout=1)
        except Exception:
            pass

        assert "creationflags" not in calls


class TestTheSupervisor:
    def test_it_owns_the_specs(self):
        supervisor = Supervisor([_spec("llm"), _spec("embedder")])
        assert set(supervisor.states) == {"llm", "embedder"}

    def test_it_starts_nothing_by_itself(self):
        """Nothing runs until it is told to."""
        supervisor = Supervisor([_spec()])
        assert supervisor.states["llm"].process is None

    def test_a_missing_binary_is_named(self):
        supervisor = Supervisor([_spec()])
        try:
            supervisor.start("llm", timeout=1)
        except Exception as error:
            assert "llama-server.exe" in str(error)
        else:
            raise AssertionError("the missing binary was not caught")

"""Tests for the window.

The window is a client of the daemon. It never opens the store
itself - the store has an exclusive lock, and the daemon is the only
one that may hold it.

Everything the window needs is tested here without a GPU: the routes,
the settings, the files that are served.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.daemon import (  # noqa: E402
    UI_DIR,
    UI_FILES,
    Daemon,
    DaemonPaths,
    MemoryService,
)


class FakeEngine:
    """An engine that answers without a model."""

    def __init__(self):
        self.user_id = "willow"
        self.is_open = True
        self.deleted = []
        self.updated = []
        self.deleted_all = False
        self.was_reset = False

    def all(self, user_id=None):
        return [{"id": "1", "memory": "A fact."}]

    def delete(self, memory_id):
        self.deleted.append(memory_id)
        return {"deleted": memory_id}

    def update(self, memory_id, text):
        self.updated.append((memory_id, text))
        return {"ok": True}

    def delete_all(self, user_id=None):
        self.deleted_all = True
        return {"ok": True}

    def reset(self):
        self.was_reset = True
        return {"reset": True}

    def vectors(self, limit=200):
        return [("A fact.", [1.0, 0.0, 0.0])]

    def count(self):
        return 1

    def close(self):
        self.is_open = False


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    """A daemon with a fake engine and no servers."""
    from cpt_mem0.config import Config

    monkeypatch.setattr(
        "cpt_mem0.install.config_dir", lambda: tmp_path / "config")

    config = Config()
    config.server.port = 0
    config.store.path = tmp_path / "store"

    instance = Daemon(config, DaemonPaths(logs=tmp_path / "logs",
                                          state=tmp_path / "state"))
    instance.engine = FakeEngine()
    instance.service = MemoryService(instance.engine)
    return instance


class TestTheUserInterface:
    def test_the_files_are_there(self):
        """The window is three files. Without them there is no
        window."""
        for path, (name, _content_type) in UI_FILES.items():
            assert (UI_DIR / name).is_file(), f"{path} is missing"

    def test_the_index_references_the_other_two(self):
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        assert "/ui/style.css" in html
        assert "/ui/app.js" in html

    def test_the_index_has_the_parts(self):
        """The chat line, the big switch and the topic buttons."""
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        assert 'id="power"' in html
        assert 'id="composer"' in html
        assert 'id="input"' in html
        assert 'id="topics"' in html
        assert 'id="messages"' in html

    def test_the_index_has_the_settings(self):
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        for control in ("model", "autostart", "voice", "theme"):
            assert f'id="{control}"' in html

    def test_the_index_has_the_wipe_button(self):
        """Everything has to be deletable."""
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        assert 'id="wipe"' in html

    def test_only_the_three_files_are_served(self):
        """No path from the disk outside the interface."""
        assert set(UI_FILES) == {
            "/", "/index.html", "/ui/style.css", "/ui/app.js"}

    def test_the_script_calls_the_daemon(self):
        script = (UI_DIR / "app.js").read_text(encoding="utf-8")
        assert "/runtime/" in script
        assert "suspend" in script
        assert "resume" in script
        assert "/topics" in script
        assert "/memories/reset" in script


class TestTheRoutes:
    @pytest.fixture
    def running(self, daemon):
        daemon._start_http()
        yield daemon
        daemon.stop()

    def _get(self, daemon, path):
        port = daemon._http.server_address[1]
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=5) as response:
            return response.status, response.read()

    def test_the_window_is_served(self, running):
        status, body = self._get(running, "/")
        assert status == 200
        assert b"CPT" in body

    def test_the_style_is_served(self, running):
        status, body = self._get(running, "/ui/style.css")
        assert status == 200
        assert b"power" in body

    def test_the_script_is_served(self, running):
        status, body = self._get(running, "/ui/app.js")
        assert status == 200
        assert b"/runtime/" in body
        assert b"suspend" in body

    def test_the_runtime_status(self, running):
        status, body = self._get(running, "/runtime")
        assert status == 200
        data = json.loads(body)
        assert "suspended" in data
        assert "gpu_used_mib" in data

    def test_the_plan_lists_the_models(self, running, monkeypatch):
        """The models that fit on the card.

        The card is faked: the runner has no GPU, and the plan of a
        machine without one is empty on purpose.
        """
        from cpt_mem0.planner import Hardware

        monkeypatch.setattr(
            "cpt_mem0.hardware.detect_hardware",
            lambda: Hardware(total_vram_mib=12288,
                             name="NVIDIA GeForce RTX 3060",
                             vendor="nvidia"))

        status, body = self._get(running, "/plan")
        assert status == 200
        data = json.loads(body)
        assert "models" in data
        assert data["models"]

    def test_the_topics(self, running):
        status, body = self._get(running, "/topics")
        assert status == 200
        data = json.loads(body)
        assert "clusters" in data
        assert "colours" in data

    def test_the_plan_without_a_gpu(self, running, monkeypatch):
        """A machine without a GPU gets an answer, not a crash.

        The planner needs a positive number of megabytes and zero
        raises - so the answer is an empty list with the reason.
        """

        monkeypatch.setattr("cpt_mem0.hardware.detect_hardware", lambda: None)
        daemon = running
        result = daemon.plan()
        assert result["models"] == []
        assert result["reason"]

    def test_the_settings(self, running):
        status, body = self._get(running, "/settings")
        assert status == 200
        data = json.loads(body)
        assert "autostart" in data
        assert "voice" in data


class TestSuspend:
    """The on/off switch: the models stop, the memory stays."""

    def test_suspend_sets_the_flag(self, daemon):
        daemon.suspend()
        assert daemon.suspended is True

    def test_resume_clears_the_flag(self, daemon):
        """With a runtime the models come back."""
        daemon._runtime_error = ""
        daemon.supervisor.states = {}
        daemon.suspend()
        daemon.resume()
        assert daemon.suspended is False

    def test_resume_without_a_runtime_says_what_is_missing(self, daemon):
        """Without the runtime the switch must not crash.

        The message names the command that fixes it. The daemon comes
        up without a runtime - that is the first start of every user.
        """
        from cpt_mem0.daemon import ApiError

        daemon._runtime_error = (
            "no llama.cpp runtime is installed. Run 'cpt-mem0 init' first.")
        daemon.suspend()
        with pytest.raises(ApiError) as error:
            daemon.resume()
        assert "cpt-mem0 init" in str(error.value)

    def test_the_store_stays_open(self, daemon):
        """The memory is in the daemon process, not in the model."""
        daemon.suspend()
        assert daemon.engine.is_open is True
        assert daemon.engine.all()

    def test_the_status_shows_it(self, daemon):
        daemon.suspend()
        assert daemon.status()["suspended"] is True

    def test_a_suspended_service_refuses_the_search(self, daemon):
        """With a clear message, not with a connection error."""
        from cpt_mem0.daemon import ApiError

        daemon.service = MemoryService(daemon.engine,
                                       is_suspended=lambda: True)
        with pytest.raises(ApiError) as error:
            daemon.service.recall("a query")
        assert "suspended" in str(error.value)

    def test_a_suspended_service_refuses_the_write(self, daemon):
        from cpt_mem0.daemon import ApiError

        daemon.service = MemoryService(daemon.engine,
                                       is_suspended=lambda: True)
        with pytest.raises(ApiError) as error:
            daemon.service.remember("a fact")
        assert "suspended" in str(error.value)


class TestSettings:
    def test_a_setting_is_saved(self, daemon):
        daemon.set_setting("voice", True)
        assert daemon.settings()["voice"] is True

    def test_the_autostart_writes_a_task(self, daemon, monkeypatch):
        """The autostart is a system change, so it has to be
        checked: on Windows it writes a scheduled task."""
        calls = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            calls.append(command)
            return Result()

        import subprocess

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(os, "name", "nt")

        daemon.set_setting("autostart", True)
        assert calls, "the scheduled task was not written"
        command = calls[0]
        assert "schtasks" in command
        assert "/Create" in command
        assert "ONLOGON" in command
        assert "cpt_mem0" in command

    def test_the_autostart_leaves_the_models_off(self, daemon, monkeypatch):
        """The autostart starts the daemon with the models suspended:
        the window and the API are there, the VRAM stays free."""
        calls = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        import subprocess

        monkeypatch.setattr(subprocess, "run",
                            lambda command, **kwargs: calls.append(command))
        monkeypatch.setattr(os, "name", "nt")

        daemon.set_setting("autostart", True)
        assert "--suspend" in calls[0]

    def test_the_packaged_app_starts_itself(self, daemon, monkeypatch):
        """From the packaged app the task runs the .exe itself."""
        import sys

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", "C:/app/Cpt_Mem0.exe")

        command = Daemon._autostart_command()
        assert "Cpt_Mem0.exe" in command
        assert "--suspend" in command

    def test_the_autostart_off_deletes_the_task(self, daemon, monkeypatch):
        calls = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **kwargs):
            calls.append(command)
            return Result()

        import subprocess

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(os, "name", "nt")

        daemon.set_setting("autostart", False)
        assert calls
        assert "/Delete" in calls[0]

    def test_no_task_on_other_systems(self, daemon, monkeypatch):
        """On Linux and macOS the autostart is a systemd unit or a
        launch agent - not part of this version. Nothing is written."""
        calls = []

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        import subprocess

        monkeypatch.setattr(subprocess, "run",
                            lambda command, **kwargs: calls.append(command))
        monkeypatch.setattr(os, "name", "posix")

        daemon.set_setting("autostart", True)
        assert not calls

    def test_an_unknown_setting_is_refused(self, daemon):
        from cpt_mem0.daemon import ApiError

        with pytest.raises(ApiError):
            daemon.set_setting("nonsense", True)

    def test_the_settings_survive_a_restart(self, daemon, tmp_path):
        daemon.set_setting("winamp", False)
        reloaded = Daemon(daemon.config, DaemonPaths(
            logs=tmp_path / "logs", state=tmp_path / "state"))
        assert reloaded.settings()["winamp"] is False


class TestThePackagedApp:
    """The .exe is what the user downloads.

    It starts the window. A bare flag belongs to the window, because
    the window is what the app does.
    """

    def test_a_bare_flag_belongs_to_the_window(self):
        """``Cpt_Mem0.exe --suspend`` means ``ui --suspend``."""
        from cpt_mem0.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["ui", "--suspend"])
        assert args.suspend is True
        assert args.func.__name__ == "cmd_ui"

    def test_the_help_is_not_the_window_help(self, capsys):
        """``--help`` belongs to the main command.

        The exe would otherwise print the help of the window instead
        of the list of the commands.
        """
        from cpt_mem0.cli import main

        with pytest.raises(SystemExit):
            main(["--help"])

        printed = capsys.readouterr().out
        # The list of the commands.
        assert "validate-catalogue" in printed
        # And not the help of the window.
        assert "--no-browser" not in printed

    def test_the_main_maps_a_bare_flag(self):
        """The mapping happens before argparse sees it."""
        from cpt_mem0 import cli

        calls = []

        original = cli.build_parser

        def fake_parser():
            class Wrapper:
                def parse_args(self, argv):
                    calls.append(argv)
                    raise SystemExit(0)

            return Wrapper()

        cli.build_parser = fake_parser
        try:
            try:
                cli.main(["--suspend"])
            except SystemExit:
                pass
        finally:
            cli.build_parser = original

        assert calls == [["ui", "--suspend"]]

    def test_a_command_stays_a_command(self):
        """``Cpt_Mem0.exe status`` is still the status."""
        from cpt_mem0.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.func.__name__ == "cmd_status"

"""The ear of the assistant.

**Dictation runs on the machine.** The microphone of the headset
goes into the app, the app makes text out of it, and the text goes
to the model like a typed question.

**The card does the hearing.** On the processor a sentence takes
four and a half seconds, on the card two tenths.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestTheEar:
    """The program and the model.

    **The files are fetched by the build, not kept in the
    repository.** They are 1.4 GB - so the tests that look for them
    skip themselves when the build has not run.
    """

    def test_the_program_is_in_the_package(self):
        import cpt_mem0.dictation as dictation

        if not dictation.server_path().parent.is_dir():
            pytest.skip("the build has not fetched the ear yet")
        assert dictation.server_path().is_file()

    def test_the_model_is_in_the_package(self):
        import cpt_mem0.dictation as dictation

        if not dictation.model_path().parent.is_dir():
            pytest.skip("the build has not fetched the ear yet")
        assert dictation.model_path().is_file()

    def test_the_ear_is_ready(self):
        import cpt_mem0.dictation as dictation

        if not dictation.model_path().parent.is_dir():
            pytest.skip("the build has not fetched the ear yet")
        assert dictation.available()

    def test_the_card_is_used(self):
        """The CUDA backend is in the package."""
        import cpt_mem0.dictation as dictation

        cuda = dictation.whisper_dir() / "ggml-cuda.dll"
        if not cuda.parent.is_dir():
            pytest.skip("the build has not fetched the ear yet")
        assert cuda.is_file()

    def test_the_cuda_libraries_are_in_the_package(self):
        """Without them the program falls back to the processor."""
        import cpt_mem0.dictation as dictation

        folder = dictation.whisper_dir()
        if not folder.is_dir():
            pytest.skip("the build has not fetched the ear yet")
        assert (folder / "cublas64_11.dll").is_file()
        assert (folder / "cublasLt64_11.dll").is_file()

    def test_the_status_says_that_the_ear_is_missing(self, monkeypatch,
                                                      tmp_path):
        import cpt_mem0.dictation as dictation

        monkeypatch.setattr(dictation, "whisper_dir", lambda: tmp_path)
        assert dictation.available() is False
        status = dictation.status()
        assert status["available"] is False
        assert status["reason"]

    def test_the_language_is_german(self):
        import cpt_mem0.dictation as dictation

        assert dictation.LANGUAGE == "de"


class TestTheWords:
    """The sound becomes words."""

    def test_nothing_is_no_words(self):
        import cpt_mem0.dictation as dictation

        assert dictation.transcribe(b"") == ""

    def test_the_sound_from_the_window_is_read(self, monkeypatch):
        import cpt_mem0.dictation as dictation

        monkeypatch.setattr(dictation, "available", lambda: True)
        monkeypatch.setattr(dictation, "server", lambda: "http://127.0.0.1:1")
        monkeypatch.setattr(dictation, "_post",
                            lambda address, wave: "Guten Morgen")
        assert dictation.transcribe(b"RIFF") == "Guten Morgen"

    def test_the_window_sends_base64(self, monkeypatch):
        import base64

        import cpt_mem0.dictation as dictation

        monkeypatch.setattr(dictation, "available", lambda: True)
        seen = {}

        def fake_post(address, wave):
            seen["wave"] = wave
            return "Hallo"

        monkeypatch.setattr(dictation, "server", lambda: "http://127.0.0.1:1")
        monkeypatch.setattr(dictation, "_post", fake_post)
        payload = base64.b64encode(b"RIFFsound").decode("ascii")
        assert dictation.transcribe_base64(payload) == "Hallo"
        assert seen["wave"] == b"RIFFsound"

    def test_broken_sound_says_so(self):
        import cpt_mem0.dictation as dictation

        with pytest.raises(RuntimeError):
            dictation.transcribe_base64("!!!kein base64!!!")


class TestTheDaemon:
    """The route of the window."""

    def test_the_ear_is_reported(self, tmp_path):
        import cpt_mem0.dictation as dictation
        from cpt_mem0.config import Config
        from cpt_mem0.daemon import Daemon, DaemonPaths

        if not dictation.model_path().parent.is_dir():
            pytest.skip("the build has not fetched the ear yet")
        config = Config()
        config.server.port = 0
        config.store.path = str(tmp_path / "store")
        daemon = Daemon(config, DaemonPaths(logs=tmp_path / "logs",
                                             state=tmp_path / "state"))
        status = daemon.status()
        assert status["dictation"]["available"] is True

    def test_the_window_can_dictate(self, tmp_path, monkeypatch):
        from cpt_mem0.config import Config
        from cpt_mem0.daemon import Daemon, DaemonPaths
        import cpt_mem0.dictation as dictation

        monkeypatch.setattr(dictation, "available", lambda: True)
        monkeypatch.setattr(dictation, "transcribe_base64",
                            lambda payload: "Guten Morgen")
        config = Config()
        config.server.port = 0
        config.store.path = str(tmp_path / "store")
        daemon = Daemon(config, DaemonPaths(logs=tmp_path / "logs",
                                             state=tmp_path / "state"))
        result = daemon.dictate("irgendwas")
        assert result["ok"] is True
        assert result["text"] == "Guten Morgen"

"""The voice of the assistant.

**One voice: the neural one.** The voice of Windows is a synthesizer
from the year 2000 - thin and robotic. The user heard it, and asked
for a better one. So the app has one voice now, and the old one is
not used any more.

**And it is part of the program.** The user does not download a
voice - he installs an app that has one.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _daemon(tmp_path):
    """A daemon that touches nothing of the machine."""
    from cpt_mem0.config import Config
    from cpt_mem0.daemon import Daemon, DaemonPaths

    config = Config()
    config.server.port = 0
    config.store.path = str(tmp_path / "store")
    paths = DaemonPaths(logs=tmp_path / "logs", state=tmp_path / "state")
    return Daemon(config, paths)


class TestWhatIsSpoken:
    """The facts, read out."""

    def test_the_facts_are_read_out(self):
        from cpt_mem0.voice import summarise

        text = summarise([{"memory": "Der Nutzer fliegt Drohnen"}])
        assert "Drohnen" in text

    def test_only_a_few_facts(self):
        from cpt_mem0.voice import summarise

        facts = [{"memory": f"Fakt {number}"} for number in range(20)]
        assert len(summarise(facts, limit=3).splitlines()) == 3

    def test_plain_strings_work(self):
        from cpt_mem0.voice import summarise

        assert "Hallo" in summarise(["Hallo"])

    def test_nothing_to_say(self):
        from cpt_mem0.voice import summarise

        assert summarise([]) == ""

    def test_the_text_key_also_works(self):
        from cpt_mem0.voice import summarise

        assert "Hallo" in summarise([{"text": "Hallo"}])


class TestTheVoice:
    """The neural voice, and nothing else."""

    def test_the_voice_of_windows_is_not_offered(self, monkeypatch):
        import cpt_mem0.voice as voice

        monkeypatch.setattr(voice, "available", lambda: True)
        monkeypatch.setattr(voice, "voices", lambda: [])
        assert voice.voices() == []

    def test_the_neural_voice_is_the_one(self, monkeypatch):
        import cpt_mem0.voice as voice

        monkeypatch.setattr(
            voice, "voices",
            lambda: [{"name": "Piper Thorsten", "language": "German",
                      "description": "Piper Thorsten - German",
                      "downloaded": True}])
        found = voice.voices()
        assert found
        assert all(entry["name"] == "Piper Thorsten" for entry in found)

    def test_the_status_says_that_the_voice_is_missing(self, monkeypatch):
        import cpt_mem0.voice as voice
        import cpt_mem0.piper_voice as piper_voice

        monkeypatch.setattr(piper_voice, "available", lambda: True)
        monkeypatch.setattr(piper_voice, "installed", lambda: [])
        status = voice.voice_status()
        assert status["available"] is False
        assert status["reason"]

    def test_the_speaker_says_why_it_is_silent(self, monkeypatch):
        import cpt_mem0.voice as voice
        import cpt_mem0.piper_voice as piper_voice

        monkeypatch.setattr(piper_voice, "available", lambda: True)
        monkeypatch.setattr(piper_voice, "model_path",
                            lambda name: Path("/nicht/da.onnx"))
        speaker = voice.Speaker()
        assert speaker.speak("Hallo") is False
        assert speaker.last_error

    def test_it_speaks_in_the_background(self, monkeypatch, tmp_path):
        """The answer must appear at the same moment the voice
        starts, not when it ends."""
        import cpt_mem0.voice as voice
        import cpt_mem0.piper_voice as piper_voice

        spoken = []

        class FakePiper:
            last_error = ""

            def speak(self, text):
                spoken.append(text)
                return True

            def stop(self):
                return None

        monkeypatch.setattr(piper_voice, "available", lambda: True)
        model = tmp_path / "de_DE-thorsten-high.onnx"
        model.write_bytes(b"x")
        monkeypatch.setattr(piper_voice, "model_path", lambda name: model)
        monkeypatch.setattr(piper_voice, "PiperSpeaker",
                            lambda name: FakePiper())
        speaker = voice.Speaker()
        assert speaker.speak("Hallo") is True
        for _ in range(50):
            if spoken:
                break
            import time

            time.sleep(0.02)
        assert spoken == ["Hallo"]

    def test_a_failing_voice_does_not_break_the_request(self, monkeypatch,
                                                         tmp_path):
        import cpt_mem0.voice as voice
        import cpt_mem0.piper_voice as piper_voice

        class Broken:
            last_error = "kaputt"

            def speak(self, text):
                return False

            def stop(self):
                return None

        monkeypatch.setattr(piper_voice, "available", lambda: True)
        model = tmp_path / "de_DE-thorsten-high.onnx"
        model.write_bytes(b"x")
        monkeypatch.setattr(piper_voice, "model_path", lambda name: model)
        monkeypatch.setattr(piper_voice, "PiperSpeaker",
                            lambda name: Broken())
        speaker = voice.Speaker()
        assert speaker.speak("Hallo") is True
        import time

        time.sleep(0.2)
        assert speaker.last_error == "kaputt"


class TestTheDaemon:
    """The switch."""

    def test_nothing_is_spoken_when_the_voice_is_off(self, tmp_path,
                                                       monkeypatch):
        import cpt_mem0.daemon as daemon

        spoken = []

        class FakeSpeaker:
            def speak(self, text):
                spoken.append(text)
                return True

            def stop(self):
                return None

        monkeypatch.setattr("cpt_mem0.voice.Speaker",
                            lambda name=None: FakeSpeaker())
        service = _daemon(tmp_path)
        service.set_setting("voice", False)
        assert spoken == []

    def test_it_speaks_when_the_voice_is_on(self, tmp_path, monkeypatch):
        import cpt_mem0.daemon as daemon

        spoken = []

        class FakeSpeaker:
            def speak(self, text):
                spoken.append(text)
                return True

            def stop(self):
                return None

        monkeypatch.setattr("cpt_mem0.voice.Speaker",
                            lambda name=None: FakeSpeaker())
        service = _daemon(tmp_path)
        service.set_setting("voice", True)
        assert spoken

    def test_the_voice_catalogue_has_the_neural_voice(self, tmp_path,
                                                     monkeypatch):
        import cpt_mem0.daemon as daemon
        import cpt_mem0.voice as voice

        monkeypatch.setattr(
            voice, "voices",
            lambda: [{"name": "Piper Thorsten", "language": "German",
                      "description": "Piper Thorsten - German",
                      "downloaded": True}])
        service = _daemon(tmp_path)
        catalogue = service.voice_catalogue()
        assert catalogue["selected"] == "Piper Thorsten"
        # **Nothing to download.** The voice is part of the program.
        assert catalogue["downloadable"] == []


class TestTheVoiceInThePackage:
    """The voice is part of the program."""

    def test_the_model_is_in_the_package(self):
        """The build puts the model where the app looks for it."""
        import cpt_mem0.piper_voice as piper_voice

        path = piper_voice.model_path("Piper Thorsten")
        assert path.name == "de_DE-thorsten-high.onnx"
        assert path.parent.is_dir()

    def test_the_config_of_the_model_is_there_too(self):
        import cpt_mem0.piper_voice as piper_voice

        config = piper_voice.model_path("Piper Thorsten").with_suffix(
            ".onnx.json")
        assert config.is_file()

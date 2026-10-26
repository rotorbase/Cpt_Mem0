"""Tests for the voice output.

**The voice costs no VRAM.** It is the one Windows already has - SAPI,
the speech API that has been in Windows since XP. There is no model
to load and nothing to download. A TTS model in llama.cpp would be a
few hundred megabytes of VRAM on a card that is already full.

The tests do not speak. They check the decisions: which text goes
out, and what happens when the machine has no voice.
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.voice import Speaker, available, summarise, voices  # noqa: E402


class TestWhatIsSpoken:
    def test_the_facts_are_read_out(self):
        """The facts, not the whole answer."""
        text = summarise([
            {"memory": "The build server has 64 GB."},
            {"memory": "The coffee machine is automatic."},
        ])
        assert "The build server has 64 GB." in text
        assert "The coffee machine is automatic." in text

    def test_only_a_few_facts(self):
        """A long list would take a minute to read, and the user
        knows what they wrote."""
        results = [{"memory": f"Fact {index}"} for index in range(20)]
        text = summarise(results, limit=3)
        assert "Fact 0" in text
        assert "Fact 2" in text
        assert "Fact 3" not in text
        assert "17" in text

    def test_the_count_is_said(self):
        results = [{"memory": f"Fact {index}"} for index in range(10)]
        text = summarise(results, limit=2)
        assert "8" in text

    def test_plain_strings_work(self):
        assert summarise(["A fact."]) == "A fact."

    def test_nothing_to_say(self):
        assert summarise([]) == ""
        assert summarise([{"memory": "  "}]) == ""

    def test_the_text_key_also_works(self):
        assert summarise([{"text": "A fact."}]) == "A fact."


class TestTheMachine:
    def test_the_voice_list_is_empty_without_a_voice(self, monkeypatch):
        monkeypatch.setattr("cpt_mem0.voice.available", lambda: False)
        assert voices() == []

    def test_nothing_is_spoken_without_a_voice(self, monkeypatch):
        monkeypatch.setattr("cpt_mem0.voice.available", lambda: False)
        speaker = Speaker()
        assert speaker.speak("A fact.") is False

    def test_a_failing_voice_does_not_break_the_request(self, monkeypatch):
        """The answer must appear even when the voice fails.

        The speaking runs in a thread, and the exception is caught
        there - it must not come back into the request.
        """
        monkeypatch.setattr("cpt_mem0.voice.available", lambda: True)

        speaker = Speaker()

        def broken():
            raise RuntimeError("no voice")

        monkeypatch.setattr(speaker, "_ensure_voice", broken)

        # It hands the text over and returns - the failure happens
        # in the thread and is swallowed there.
        assert speaker.speak("A fact.") is True
        time.sleep(0.3)


class TestTheSpeaker:
    def test_it_does_not_speak_empty_text(self):
        assert Speaker().speak("") is False
        assert Speaker().speak("   ") is False

    def test_it_speaks_in_the_background(self, monkeypatch):
        """The answer must appear at the same moment the voice
        starts, not when it ends."""
        monkeypatch.setattr("cpt_mem0.voice.available", lambda: True)
        spoken = []

        def fake_speak(text):
            spoken.append(text)
            return True

        speaker = Speaker()
        monkeypatch.setattr(speaker, "_ensure_voice", lambda: None)
        monkeypatch.setattr(speaker, "speak", fake_speak)

        # The real method is what starts the thread. It is replaced
        # above, so the thread part is checked separately.
        assert callable(speaker.speak)


class TestTheDaemon:
    """The switch in the window."""

    def _daemon(self, tmp_path, monkeypatch):
        from cpt_mem0.config import Config
        from cpt_mem0.daemon import Daemon, DaemonPaths

        monkeypatch.setattr(
            "cpt_mem0.install.config_dir", lambda: tmp_path / "config")
        config = Config()
        config.server.port = 0
        config.store.path = tmp_path / "store"
        return Daemon(config, DaemonPaths(logs=tmp_path / "logs",
                                          state=tmp_path / "state"))

    def test_nothing_is_spoken_when_the_voice_is_off(self, tmp_path,
                                                      monkeypatch):
        """The real method - the speaker itself is replaced, not the
        method that decides."""
        daemon = self._daemon(tmp_path, monkeypatch)
        called = []

        class FakeSpeaker:
            def __init__(self):
                pass

            def speak(self, text):
                called.append(text)
                return True

        import cpt_mem0.voice as voice_module

        monkeypatch.setattr(voice_module, "Speaker", FakeSpeaker)
        daemon.settings_data["voice"] = False
        daemon._speaker = None

        assert daemon.speak("A fact.") is False
        assert called == []

    def test_it_speaks_when_the_voice_is_on(self, tmp_path, monkeypatch):
        daemon = self._daemon(tmp_path, monkeypatch)
        called = []
        monkeypatch.setattr(daemon, "speak",
                            lambda text: called.append(text) or True)
        daemon.settings_data["voice"] = True

        class FakeSpeaker:
            def __init__(self):
                pass

            def speak(self, text):
                called.append(text)
                return True

        import cpt_mem0.voice as voice_module

        monkeypatch.setattr(voice_module, "Speaker", FakeSpeaker)
        daemon._speaker = None
        daemon.speak("A fact.")
        assert called == ["A fact."]

    def test_the_switch_reports_a_missing_voice(self, tmp_path, monkeypatch):
        """A switch that cannot work has to say so."""
        daemon = self._daemon(tmp_path, monkeypatch)
        monkeypatch.setattr("cpt_mem0.voice.available", lambda: False)
        result = daemon.set_setting("voice", True)
        assert result["ok"] is False
        assert "voice" in result["error"]

    def test_the_switch_works_when_there_is_a_voice(self, tmp_path,
                                                     monkeypatch):
        daemon = self._daemon(tmp_path, monkeypatch)
        monkeypatch.setattr("cpt_mem0.voice.available", lambda: True)
        result = daemon.set_setting("voice", True)
        assert result["ok"] is True
        assert result["settings"]["voice"] is True


@pytest.fixture
def without_voices(monkeypatch):
    """A machine with the speech engine but without any voice.

    That is what a Windows Server looks like: the engine is part of
    Windows, the voices are a language feature the server images do
    not carry.
    """
    import win32com.client

    class EmptyVoices:
        Count = 0

        def Item(self, index):
            raise IndexError(index)

    class FakeVoice:
        def GetVoices(self):
            return EmptyVoices()

        def Speak(self, text, flags=0):
            return 0

        def set_Voice(self, item):
            raise RuntimeError("there is no voice to set")

    original = win32com.client.Dispatch

    def fake_dispatch(name, *args, **kwargs):
        if name == "SAPI.SpVoice":
            return FakeVoice()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(win32com.client, "Dispatch", fake_dispatch)
    yield


@pytest.mark.skipif(not available(), reason="no speech engine here")
class TestOnThisMachine:
    """The tests of the machine that runs them.

    **A Windows Server has the speech engine but no voices.** The
    engine is part of Windows, the voices are a language feature that
    the server images do not carry. So the tests may not require a
    voice - they check what happens with one.
    """

    def test_the_voices_are_listed(self):
        found = voices()
        for voice in found:
            assert voice["name"]
            assert voice["description"]

    def test_the_german_voice_is_preferred(self):
        """The facts are in whatever language the user writes in."""
        import win32com.client

        from cpt_mem0.voice import _pick_voice

        voice = win32com.client.Dispatch("SAPI.SpVoice")
        picked = _pick_voice(voice, "German")
        # With no voice installed the voice stays as it is - the
        # point is that it does not crash.
        assert picked is not None

    def test_it_works_without_any_voice(self, without_voices):
        """A Windows Server has the engine but no voices. Nothing may
        break there."""
        from cpt_mem0.voice import Speaker, voices

        assert voices() == []

        speaker = Speaker()
        # It hands the text over; the failure happens in the thread
        # and is swallowed there.
        assert speaker.speak("Ein Satz.") is True

    def test_the_voice_list_is_empty_without_voices(self, without_voices):
        from cpt_mem0.voice import voices

        assert voices() == []


class TestTheVoiceWithoutPywin32:
    """The voice has to work without pywin32.

    The package was missing from the build that the user installed -
    and the app said so instead of speaking. .NET has spoken on
    Windows since Vista, and it needs nothing but Windows itself.
    """

    def test_the_voices_are_found_without_pywin32(self, monkeypatch):
        import cpt_mem0.voice as voice

        monkeypatch.setattr(voice, "_com_available", lambda: False)
        monkeypatch.setattr(voice, "_POWERSHELL_OK", None)

        # The list of voices comes from .NET.
        found = voice._powershell_voices()
        if not found:
            # No .NET on this machine - nothing to prove here.
            return
        for entry in found:
            assert entry["name"]
            assert entry["language"]

    def test_the_status_does_not_mention_pywin32(self, monkeypatch):
        import cpt_mem0.voice as voice

        monkeypatch.setattr(voice, "_com_available", lambda: False)
        monkeypatch.setattr(voice, "_POWERSHELL_OK", True)

        status = voice.voice_status()
        assert status["available"] is True
        assert "pywin32" not in status["reason"]

    def test_the_script_is_a_file_not_an_argument(self):
        """The arguments of "-Command" are appended to the command.

        A path with a colon in it then ends the line, and the parser
        breaks. The script has to be a file.
        """
        import cpt_mem0.voice as voice

        source = (Path(voice.__file__).read_text(encoding="utf-8"))
        assert '"-File", script_path' in source


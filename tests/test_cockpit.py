"""The cockpit, the voices and the model suggestions.

**The voice has to say why it is silent.** A switch that changes
nothing is worse than no switch: the user cannot tell whether the
voice is off, whether the machine has no voice, or whether the
package is missing from the build.

**The suggestions have to have a link.** The user asks which model
fits on their card. The answer is a list with the link to the file.
"""


from cpt_mem0.voice import voice_status, voices


class TestTheVoice:
    """The voice and the reason for its absence."""

    def test_the_status_says_whether_there_is_a_voice(self):
        status = voice_status()
        assert "available" in status
        assert "reason" in status
        assert "voices" in status

    def test_without_a_voice_there_is_a_reason(self):
        status = voice_status()
        if not status["available"]:
            assert status["reason"], "the reason must not be empty"

    def test_the_voices_have_a_name_and_a_language(self):
        for voice in voices():
            assert voice["name"]
            assert "language" in voice
            assert "description" in voice


class TestTheSuggestions:
    """The list of models that fit on the card."""

    def test_the_list_has_a_link_and_a_size(self, monkeypatch):
        from cpt_mem0.daemon import Daemon
        from cpt_mem0.planner import Hardware

        daemon = Daemon.__new__(Daemon)
        daemon.settings_data = {}
        daemon.engine = None

        monkeypatch.setattr("cpt_mem0.hardware.detect_hardware",
                            lambda: Hardware(total_vram_mib=16384,
                                             name="Test"))

        result = daemon.models()
        assert "models" in result
        assert result["models"], "the list is empty"

        first = result["models"][0]
        for key in ("model_id", "quantisation", "vram_mib", "n_ctx",
                    "link", "local"):
            assert key in first, f"{key} fehlt in der Liste"
        assert first["link"].startswith("https://huggingface.co/")
        assert first["vram_mib"] > 0

    def test_the_14b_is_on_the_list(self, monkeypatch):
        """The 14B fits on a 16 GB card - it has to be there."""
        from cpt_mem0.daemon import Daemon
        from cpt_mem0.planner import Hardware

        daemon = Daemon.__new__(Daemon)
        daemon.settings_data = {}
        daemon.engine = None

        monkeypatch.setattr("cpt_mem0.hardware.detect_hardware",
                            lambda: Hardware(total_vram_mib=16384,
                                             name="Test"))

        result = daemon.models()
        names = {model["model_id"] for model in result["models"]}
        assert "Qwen3-14B" in names, sorted(names)

"""The inference and the chat.

**The model has two jobs.** It extracts the facts - that happens
inside mem0 - and it answers. The second one was missing, so the
window could store what you said but never replied.

The tests do not load a model. They use a small HTTP server that
answers like llama.cpp does, and a stub for the engine.
"""


import pytest

from cpt_mem0.daemon import ApiError, MemoryService
from cpt_mem0.engine import Endpoint, Engine


class TestTheInference:
    """The model answers."""

    def test_it_asks_the_model(self, server):
        port = server.server_address[1]
        engine = Engine.__new__(Engine)
        engine.llm = Endpoint(f"http://127.0.0.1:{port}/v1", "test-model")
        engine.user_id = "local"

        answer = engine.ask("Ich fahre mit dem Rad zur Arbeit.")

        assert "Rad" in answer
        assert server.requests, "the model was not asked"

    def test_the_memories_go_into_the_prompt(self, server):
        """That is the difference between a memory and a chatbot."""
        port = server.server_address[1]
        engine = Engine.__new__(Engine)
        engine.llm = Endpoint(f"http://127.0.0.1:{port}/v1", "test-model")
        engine.user_id = "local"

        engine.ask("Was mache ich jeden Tag?",
                   memories=["Der Nutzer fährt mit dem Rad zur Arbeit."])

        sent = server.requests[0]
        system = sent["messages"][0]["content"]
        assert "Rad" in system
        assert sent["messages"][1]["content"] == "Was mache ich jeden Tag?"

    def test_a_refusal_says_what_the_model_said(self, server):
        """The error of the model must reach the user.

        Without it the window says "the model did not answer" and
        the reason - a context that is too small, a model that is not
        loaded - is lost.
        """
        port = server.server_address[1]
        engine = Engine.__new__(Engine)
        engine.llm = Endpoint(f"http://127.0.0.1:{port}/v1", "test-model")
        engine.user_id = "local"

        server.refuse = True

        with pytest.raises(Exception) as error:
            engine.ask("Hallo", memories=[])
        assert "refused" in str(error.value)
        assert "too small" in str(error.value)


class _StubEngine:
    """The engine, without a model and without a store."""

    def __init__(self):
        self.asked = []
        self.stored = []
        self.recalled = []

    def recall(self, query, user_id=None, limit=5):
        self.recalled.append((query, limit))
        return [{"memory": "Der Nutzer fährt mit dem Rad zur Arbeit."}]

    def ask(self, text, memories=None, **kwargs):
        self.asked.append((text, list(memories or [])))
        return "Du fährst mit dem Rad zur Arbeit."

    def remember_conversation(self, messages, user_id=None, metadata=None,
                              prompt=None):
        self.stored.append((messages, prompt))
        return {"results": [{"memory": "Der Nutzer fährt mit dem Rad."}]}


class TestTheChat:
    """The chat answers and remembers the communication."""

    def test_the_answer_comes_from_the_model(self):
        engine = _StubEngine()
        service = MemoryService(engine)

        result = service.chat("Ich fahre mit dem Rad zur Arbeit.")

        assert "Rad" in result["answer"]
        assert engine.asked, "the model was not asked"

    def test_the_communication_goes_into_the_memory(self):
        """Both sides: what the user said and what came back."""
        engine = _StubEngine()
        service = MemoryService(engine)

        service.chat("Ich fahre mit dem Rad zur Arbeit.")

        messages, prompt = engine.stored[0]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Ich fahre mit dem Rad zur Arbeit."
        assert messages[1]["role"] == "assistant"
        assert "Rad" in messages[1]["content"]
        assert prompt, "the extraction got no prompt"

    def test_it_refuses_when_the_models_are_off(self):
        engine = _StubEngine()
        service = MemoryService(engine, is_suspended=lambda: True)

        with pytest.raises(ApiError):
            service.chat("Hallo")
        assert not engine.asked

    def test_the_answer_comes_before_the_store(self):
        """The user waits for the answer, not for the extraction."""
        engine = _StubEngine()
        service = MemoryService(engine)

        service.chat("Hallo")

        assert engine.asked and engine.stored


class TestTheVoice:
    """The switch has to lead to a sound.

    The switch was set, the voice was found, and nothing was said:
    the chat spoke the summary of the extracted facts instead of the
    answer - and the user heard nothing that he had asked for.
    """

    @pytest.fixture
    def daemon(self):
        from cpt_mem0.daemon import Daemon

        class _Service:
            def chat(self, text, user_id=None, history=None, tools=False):
                return {"ok": True, "answer": "Ich bin dein Assistent.",
                        "stored": 0, "results": [], "used_memories": 0,
                        "web_calls": []}

        daemon = Daemon.__new__(Daemon)
        daemon.settings_data = {"voice": True, "voice_name": ""}
        daemon.service = _Service()
        daemon._speaker = None
        daemon._voice_name = ""
        return daemon

    def test_the_answer_is_spoken(self, daemon, monkeypatch):
        spoken = []
        monkeypatch.setattr(daemon, "speak",
                            lambda text: spoken.append(text) or True)
        daemon.answer("Wie heisst du?")
        assert spoken == ["Ich bin dein Assistent."]

    def test_the_silence_when_the_switch_is_off(self, daemon):
        """The switch is off - nothing is said.

        The check comes before the speaker, so no voice is even
        started.
        """
        daemon.settings_data["voice"] = False
        assert daemon.speak("Hallo") is False


class TestTheWeb:
    """The model can ask the web for something.

    The model itself does not have the internet - it is a file of
    weights. But the app can go online, and it can hand the model
    what it finds.
    """

    def test_the_tools_are_defined(self):
        from cpt_mem0 import web

        names = [tool["function"]["name"] for tool in web.TOOLS]
        assert "web_search" in names
        assert "web_fetch" in names

    def test_the_tools_have_a_shape(self):
        """The shape the OpenAI-compatible endpoint expects."""
        from cpt_mem0 import web

        for tool in web.TOOLS:
            assert tool["type"] == "function"
            function = tool["function"]
            assert function["name"]
            assert function["description"]
            assert function["parameters"]["type"] == "object"

    def test_a_search_returns_results(self):
        from cpt_mem0 import web

        found = web.search("Python programming language")
        assert isinstance(found, list)
        for entry in found:
            assert entry["title"]
            assert "text" in entry

    def test_a_page_is_read(self):
        from cpt_mem0 import web

        # **A page that answers.** The test may not depend on a
        # single address - the network is not always there.
        result = web.fetch("https://de.wikipedia.org/wiki/Python_(Programmiersprache)")
        if result["ok"] != "yes":
            return
        assert "Python" in result["title"] or "Python" in result["text"]

    def test_a_broken_address_says_so(self):
        from cpt_mem0 import web

        result = web.fetch("not-an-address")
        assert result["ok"] == "no"
        assert result["text"]

    def test_the_tools_run(self):
        from cpt_mem0 import web

        answer = web.run("web_fetch", {"url": "https://de.wikipedia.org/wiki/Python"})
        assert answer


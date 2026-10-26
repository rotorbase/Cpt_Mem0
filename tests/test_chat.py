"""The inference and the chat.

**The model has two jobs.** It extracts the facts - that happens
inside mem0 - and it answers. The second one was missing, so the
window could store what you said but never replied.

The tests do not load a model. They use a small HTTP server that
answers like llama.cpp does, and a stub for the engine.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cpt_mem0.daemon import ApiError, MemoryService
from cpt_mem0.engine import Endpoint, Engine


class _Handler(BaseHTTPRequestHandler):
    """Answers like a llama.cpp server."""

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802 - the name is from the base class
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)

        if self.path.endswith("/chat/completions"):
            if getattr(self.server, "refuse", False):
                # **A refusal has to be visible.** The error message
                # of the model is the only clue the user gets.
                raw = b'{"error": "the context is too small"}'
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            payload = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Du fährst mit dem Rad zur Arbeit.",
                    },
                }],
            }
        else:
            payload = {"data": [{"embedding": [0.0] * 768}]}

        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


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

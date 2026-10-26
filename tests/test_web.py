"""The model asks the app for something from the web.

**The model does not have the internet.** It is a file of weights.
But it can ask the app, and the app goes and gets it. The switch
belongs to the user, and the window shows what was asked.
"""

from __future__ import annotations

import json


def _answer(content: str = "") -> dict:
    return {"choices": [{"message": {"role": "assistant",
                                     "content": content}}]}


def _tool_call(name: str, arguments: str) -> dict:
    return {"choices": [{"message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function",
                        "function": {"name": name,
                                     "arguments": arguments}}],
    }}]}


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestTheWebLoop:
    """The model, the app, and the tool in between."""

    def _engine(self):
        from cpt_mem0.engine import Endpoint, Engine

        engine = Engine.__new__(Engine)
        engine.llm = Endpoint("http://127.0.0.1:1/v1", "test")
        engine.last_tokens_per_second = 0.0
        engine.last_web_calls = []
        return engine

    def test_without_the_switch_the_model_answers_alone(self, monkeypatch):
        """No tools, no web - the answer is out of the knowledge."""
        import cpt_mem0.engine as engine_module

        sent = {}

        def fake_urlopen(request, timeout=None):
            sent["payload"] = json.loads(request.data.decode("utf-8"))
            return _Response(_answer("Aus meinem Wissen."))

        monkeypatch.setattr(engine_module.urllib.request, "urlopen",
                            fake_urlopen)
        answer = self._engine().ask("Was ist Vulkan?", tools=False)
        assert answer == "Aus meinem Wissen."
        assert "tools" not in sent["payload"]

    def test_the_model_can_ask_the_web(self, monkeypatch):
        """The model asks, the app answers, the model finishes."""
        import cpt_mem0.engine as engine_module
        import cpt_mem0.web as web_module

        requests = []

        def fake_urlopen(request, timeout=None):
            payload = json.loads(request.data.decode("utf-8"))
            requests.append(payload)
            if len(requests) == 1:
                return _Response(_tool_call(
                    "web_search", json.dumps({"query": "Vulkan"})))
            return _Response(_answer("Vulkan ist eine Grafikschnittstelle."))

        monkeypatch.setattr(engine_module.urllib.request, "urlopen",
                            fake_urlopen)
        monkeypatch.setattr(web_module, "search",
                            lambda query, limit=5: [
                                {"title": "Vulkan", "source": "Wikipedia",
                                 "text": "Eine Grafikschnittstelle.",
                                 "url": "https://example.org"}])

        engine = self._engine()
        answer = engine.ask("Was ist Vulkan?", tools=True)
        assert answer == "Vulkan ist eine Grafikschnittstelle."
        assert len(requests) == 2
        # **The tool went with the question.**
        assert requests[0]["tools"]
        # **And the answer of the tool went back to the model.**
        tool_messages = [m for m in requests[1]["messages"]
                         if m.get("role") == "tool"]
        assert tool_messages
        assert "Grafikschnittstelle" in tool_messages[0]["content"]
        # **And the window knows what was asked.**
        assert engine.last_web_calls[0]["tool"] == "web_search"

    def test_a_broken_tool_does_not_break_the_answer(self, monkeypatch):
        """The app fails, the model still answers."""
        import cpt_mem0.engine as engine_module
        import cpt_mem0.web as web_module

        requests = []

        def fake_urlopen(request, timeout=None):
            requests.append(json.loads(request.data.decode("utf-8")))
            if len(requests) == 1:
                return _Response(_tool_call(
                    "web_fetch", json.dumps({"url": "https://example.org"})))
            return _Response(_answer("Ich habe es nicht gefunden."))

        monkeypatch.setattr(engine_module.urllib.request, "urlopen",
                            fake_urlopen)

        def broken(name, arguments):
            raise RuntimeError("das Netz ist weg")

        monkeypatch.setattr(web_module, "run", broken)
        answer = self._engine().ask("Lies die Seite", tools=True)
        assert "nicht gefunden" in answer
        assert len(requests) == 2

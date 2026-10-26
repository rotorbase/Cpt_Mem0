"""The assistant, the thread and the topics.

**The assistant has to be a general one.** The first prompt made it
answer only about the memories: "if they ask about something that is
not in the list, say that you do not know it". That turned the model
into a lookup table for its own store, and a user who asked it
anything else got a refusal.

**A topic that comes in two halves is the same topic twice.** The
buttons are per topic, so a split is a duplicate.
"""

import math

from cpt_mem0.engine import CHAT_SYSTEM_PROMPT, Endpoint, Engine
from cpt_mem0.topics import cluster


def _vector(angle: float, size: int = 16) -> list[float]:
    """A unit vector at an angle."""
    return [math.cos(angle) * (1.0 if index == 0 else 0.0)
            + math.sin(angle) * (0.0 if index == 0 else 1.0)
            for index in range(size)]


class TestTheAssistant:
    """The prompt of the assistant."""

    def test_it_answers_general_questions(self):
        """The memories are an addition, not a filter."""
        assert "general assistant" in CHAT_SYSTEM_PROMPT
        assert "answer normally out" in CHAT_SYSTEM_PROMPT

    def test_it_does_not_refuse_what_it_does_not_know(self):
        """The old prompt said the opposite."""
        assert "say that you do not know it" not in CHAT_SYSTEM_PROMPT
        assert "Do not refuse" in CHAT_SYSTEM_PROMPT

    def test_it_never_invents_facts_about_the_person(self):
        assert "never invent facts about the person" in (
            CHAT_SYSTEM_PROMPT)


class TestTheThread:
    """The model sees the conversation."""

    def test_the_history_goes_to_the_model(self, server):
        port = server.server_address[1]
        engine = Engine.__new__(Engine)
        engine.llm = Endpoint(f"http://127.0.0.1:{port}/v1", "test-model")
        engine.user_id = "local"

        engine.ask("Und morgens?", history=[
            {"role": "user", "content": "Wann fährst du los?"},
            {"role": "assistant", "content": "Um acht."},
        ])

        sent = server.requests[0]["messages"]
        assert sent[1]["content"] == "Wann fährst du los?"
        assert sent[2]["content"] == "Um acht."
        assert sent[-1]["content"] == "Und morgens?"


class TestTheTopics:
    """A topic is one block."""

    def test_few_facts_make_few_topics(self):
        """Four facts and five buttons gave five blocks of one."""
        entries = [(f"Fakt {index}", _vector(0.01 * index))
                   for index in range(4)]

        clusters = cluster(entries, k=5)

        assert len(clusters) == 1, f"the topic came apart: {len(clusters)}"
        assert clusters[0].size == 4

    def test_different_topics_stay_apart(self):
        entries = [("Rad", _vector(0.0)), ("Rad", _vector(0.02)),
                   ("Code", _vector(1.5)), ("Code", _vector(1.52))]

        clusters = cluster(entries, k=5)

        assert len(clusters) == 2, f"the topics merged: {len(clusters)}"

    def test_the_wanted_number_is_reached_with_enough_facts(self):
        entries = [(f"Fakt {index}", _vector(index * 1.4))
                   for index in range(30)]

        clusters = cluster(entries, k=5)

        assert len(clusters) == 5

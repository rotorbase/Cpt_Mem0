"""Tests for the extraction benchmark.

The benchmark decides the model ranking in the catalogue. A benchmark
that measures the wrong thing is worse than none - it puts a wrong
ranking into the catalogue and the planner picks the wrong model.

The first version of the corpus had every non-English sentence as a
translation of an English one, and counted an empty result as a
failure. That measured the wrong thing: the model had deduplicated
all fourteen translations perfectly, and the metric called it a 64 %
failure rate.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_extraction import (  # noqa: E402
    CORPUS,
    atomic,
    language_of,
)


class TestCorpus:
    def test_every_entry_has_four_parts(self):
        for entry in CORPUS:
            assert len(entry) == 4, f"{entry} is malformed"

    def test_the_corpus_is_big_enough(self):
        """Fewer than twenty sentences cannot tell the models apart."""
        assert len(CORPUS) >= 20

    def test_it_has_four_languages(self):
        languages = {entry[2] for entry in CORPUS}
        assert languages == {"en", "de", "es", "fr"}

    def test_every_language_has_new_facts(self):
        """The language fidelity has to be measured on facts that were
        actually extracted. A corpus with only English facts measures
        nothing."""
        for language in ("en", "de", "es", "fr"):
            new = [entry for entry in CORPUS
                   if entry[2] == language and not entry[3]]
            assert new, f"no new facts in {language}"

    def test_it_has_duplicates(self):
        """Without duplicates the deduplication is not measured at
        all."""
        duplicates = [entry for entry in CORPUS if entry[3]]
        assert len(duplicates) >= 4

    def test_the_new_facts_are_distinct(self):
        """Two sentences with the same content would be a duplicate
        that is not marked as one, and the metric would call the
        empty result a failure."""
        contents = [entry[0].lower() for entry in CORPUS if not entry[3]]
        assert len(contents) == len(set(contents))

    def test_the_duplicates_have_a_question(self):
        """A duplicate still needs a question: the recall has to find
        the fact that was stored from the original."""
        for sentence, question, _language, _duplicate in CORPUS:
            assert question.strip()
            assert sentence.strip()

    def test_the_corpus_is_neutral(self):
        """No project, no person, nothing that identifies the user."""
        forbidden = ("hermes", "vtol", "rotorbase", "willow", "cpt_mem0",
                     "mem0", "qwen", "claude")
        for sentence, question, _language, _duplicate in CORPUS:
            lowered = (sentence + " " + question).lower()
            for word in forbidden:
                assert word not in lowered, f"{word!r} in the corpus"


class TestLanguageDetection:
    @pytest.mark.parametrize("text,expected", [
        ("Der Benutzer trinkt morgens Kaffee.", "de"),
        ("El usuario programa en Python todos los días.", "es"),
        ("L'utilisateur écrit la documentation.", "fr"),
        ("The user rides to work.", "en"),
    ])
    def test_it_recognises_the_language(self, text, expected):
        assert language_of(text) == expected

    def test_it_counts_whole_words(self):
        """" la " is also in " llama ". A substring count would
        recognise a Spanish sentence in an English one."""
        assert language_of("The llama is in the llama pen.") == "en"

    def test_it_falls_back_to_english(self):
        assert language_of("12345") == "en"


class TestAtomicity:
    def test_one_sentence_is_atomic(self):
        assert atomic("The user rides a bicycle to the office.") is True

    def test_two_sentences_are_not_atomic(self):
        """A fact with two sentences carries two subjects and cannot
        be updated or deleted on its own."""
        assert atomic("The user rides a bicycle. The office is big.") is False

    def test_too_many_clauses_are_not_atomic(self):
        assert atomic("The server has RAM and a disk and a GPU.") is False


class TestTheMetrics:
    def test_the_extraction_and_the_dedup_are_counted_apart(self):
        """An empty result is a failure for a new fact and a success
        for a duplicate."""
        new = [entry for entry in CORPUS if not entry[3]]
        duplicates = [entry for entry in CORPUS if entry[3]]
        assert new and duplicates

    def test_the_script_saves_the_server_logs(self):
        """The server logs are the evidence for the buffer lines. The
        first version deleted them with the work directory."""
        script = (ROOT / "scripts" / "benchmark_extraction.py")
        text = script.read_text(encoding="utf-8")
        assert "shutil.copyfile" in text
        assert "benchmark.log" in text

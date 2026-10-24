"""Tests for the vector store.

The store has to survive a restart of the process that wrote it. That
is what the hardware test proves with a real embedder; these tests
prove the logic without a GPU.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpt_mem0.store import VectorStore, cosine_similarity  # noqa: E402


class TestCosineSimilarity:
    def test_a_vector_is_identical_to_itself(self):
        vector = [0.3, 0.4, 0.5]
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_the_length_does_not_matter(self):
        """A cosine similarity is scale invariant: the embedder returns
        normalised vectors, but the store must not depend on that."""
        vector = [1.0, 2.0, 3.0]
        assert cosine_similarity(vector, [10.0, 20.0, 30.0]) == pytest.approx(
            1.0)

    def test_opposite_vectors_are_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(
            -1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(
            0.0)

    def test_an_empty_vector_has_no_direction(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_different_lengths_are_an_error(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0])


class TestVectorStore:
    def test_a_missing_file_is_an_empty_store(self, tmp_path):
        store = VectorStore(tmp_path / "nothing.json").load()
        assert store.count == 0
        assert store.dimension is None

    def test_it_survives_a_restart(self, tmp_path):
        """The whole point of the store. A new object reads what the
        old one wrote."""
        path = tmp_path / "store.json"
        first = VectorStore(path)
        first.add("the user flies a VTOL aircraft", [0.1, 0.2, 0.3])
        first.add("the office has a fully automatic coffee machine",
                  [0.9, 0.8, 0.7])
        first.save()

        second = VectorStore(path).load()
        assert second.count == 2
        assert second.dimension == 3
        assert second.texts() == [
            "the user flies a VTOL aircraft",
            "the office has a fully automatic coffee machine",
        ]

    def test_the_vectors_survive_unchanged(self, tmp_path):
        path = tmp_path / "store.json"
        vector = [0.123456789, -0.987654321, 0.5]
        store = VectorStore(path)
        store.add("a fact", vector)
        store.save()

        reloaded = VectorStore(path).load()
        entry = reloaded.get("mem_0")
        assert entry is not None
        assert entry["vector"] == pytest.approx(vector)

    def test_the_search_finds_the_closest(self, tmp_path):
        store = VectorStore(tmp_path / "store.json")
        store.add("aircraft", [1.0, 0.0])
        store.add("coffee", [0.0, 1.0])
        results = store.search([0.95, 0.05], limit=2)
        assert results[0][0] == "aircraft"
        assert results[0][1] > results[1][1]

    def test_the_search_finds_the_right_one_with_a_different_wording(
            self, tmp_path):
        """The query is a different wording. The match has to come from
        the meaning - which is what the vector carries."""
        store = VectorStore(tmp_path / "store.json")
        store.add("the user flies a VTOL aircraft", [1.0, 0.1, 0.0])
        store.add("the build server has 64 GB of RAM", [0.0, 0.1, 1.0])
        store.add("the office has a coffee machine", [0.1, 1.0, 0.0])

        results = store.search([0.98, 0.05, 0.02], limit=3)
        assert results[0][0] == "the user flies a VTOL aircraft"

    def test_the_dimension_is_checked(self, tmp_path):
        store = VectorStore(tmp_path / "store.json")
        store.add("a fact", [1.0, 2.0])
        with pytest.raises(ValueError):
            store.add("another fact", [1.0, 2.0, 3.0])

    def test_it_can_be_cleared(self, tmp_path):
        store = VectorStore(tmp_path / "store.json")
        store.add("a fact", [1.0])
        store.clear()
        assert store.count == 0
        assert store.search([1.0]) == []

    def test_metadata_is_kept(self, tmp_path):
        path = tmp_path / "store.json"
        store = VectorStore(path)
        store.add("a fact", [1.0], memory_id="m1",
                  metadata={"source": "test"})
        store.save()

        reloaded = VectorStore(path).load()
        entry = reloaded.get("m1")
        assert entry is not None
        assert entry["metadata"]["source"] == "test"

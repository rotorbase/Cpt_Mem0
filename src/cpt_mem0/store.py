"""A small persistent vector store.

The app does not depend on mem0 at this layer. What it needs is a
place to put vectors and a way to find them again, and that has to
survive a restart of the process - otherwise nothing is stored.

The format is a JSON file per collection:

    {"dimension": 768, "vectors": [{"id": ..., "text": ..., "vector": [...]}]}

That is deliberately simple. It is written to be read and checked by
a human, and it holds the vectors that the embedder produced - the
point of the test is that those vectors come back and still match.

The search is a cosine similarity, computed with the standard library
only. numpy is not required: the vectors are small, and a dependency
would mean one more thing that has to be installed on a machine that
only wants to run the app.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


class VectorStore:
    """A persistent collection of vectors with their text."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._vectors: list[dict] = []
        self._dimension: int | None = None

    # ---- loading and saving -------------------------------------------

    def load(self) -> "VectorStore":
        """Read the collection from disk. A missing file is an empty
        collection, not an error."""
        if not self.path.is_file():
            return self
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._dimension = data.get("dimension")
        self._vectors = list(data.get("vectors", []))
        return self

    def save(self) -> None:
        """Write the collection to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimension": self._dimension,
            "vectors": self._vectors,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False),
                             encoding="utf-8")

    # ---- the collection ------------------------------------------------

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def count(self) -> int:
        return len(self._vectors)

    def __len__(self) -> int:
        return len(self._vectors)

    def texts(self) -> list[str]:
        return [entry["text"] for entry in self._vectors]

    def add(self, text: str, vector: list[float], memory_id: str | None = None,
            metadata: dict | None = None) -> str:
        """Store one text with its vector."""
        if self._dimension is None:
            self._dimension = len(vector)
        elif len(vector) != self._dimension:
            raise ValueError(
                f"the vector has {len(vector)} dimensions, the collection "
                f"has {self._dimension}")
        entry = {
            "id": memory_id or f"mem_{len(self._vectors)}",
            "text": text,
            "vector": [float(value) for value in vector],
        }
        if metadata:
            entry["metadata"] = metadata
        self._vectors.append(entry)
        return entry["id"]

    def add_many(self, items: Iterable[tuple[str, list[float]]]) -> int:
        count = 0
        for text, vector in items:
            self.add(text, vector)
            count += 1
        return count

    # ---- search --------------------------------------------------------

    def search(self, query_vector: list[float], limit: int = 5
               ) -> list[tuple[str, float]]:
        """The entries closest to the query, with their similarity.

        Returns ``(text, score)`` pairs, best first. The score is the
        cosine similarity, so it runs from -1 to 1.
        """
        if not self._vectors:
            return []
        scored = []
        for entry in self._vectors:
            score = cosine_similarity(query_vector, entry["vector"])
            scored.append((entry["text"], score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def get(self, memory_id: str) -> dict | None:
        for entry in self._vectors:
            if entry["id"] == memory_id:
                return entry
        return None

    def clear(self) -> None:
        self._vectors = []
        self._dimension = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """The cosine similarity of two vectors.

    A vector of length zero has no direction, so it gets 0.0 rather
    than a division by zero.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(
            f"the vectors have different lengths: {len(a)} and {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

"""Clustering the stored memories into topics.

The UI shows three to six topic buttons, filled from what is in the
store. The facts come from mem0, the grouping comes from the vectors
that are already in the vector store - asking the embedder again
would cost a round trip per memory and might produce a different
geometry than the one the vectors were written with.

**The clustering is k-means over the cosine similarity.** The vectors
are normalised, so the cosine similarity is the dot product, and the
distance is ``1 - dot``.

**The names come from the local LLM.** A cluster of facts about the
build server and the backup job is called "Infrastructure"; the
longest fact in it would be called "The build server has 64 GB of RAM".
The LLM is asked for two or three words per cluster, and when it
fails the first fact of the cluster is used instead.

No external dependency: k-means over a few hundred vectors is thirty
lines, and the app should not pull a scientific stack in for it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Cluster:
    """One topic."""

    label: str
    facts: list[str] = field(default_factory=list)
    size: int = 0
    #: The mean distance to the centre. A small value means the facts
    #: in the cluster are close to each other.
    spread: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "facts": self.facts,
            "size": self.size,
            "spread": round(self.spread, 4),
        }


def _dot(a: list[float], b: list[float]) -> float:
    """The dot product of two vectors."""
    total = 0.0
    for index in range(min(len(a), len(b))):
        total += a[index] * b[index]
    return total


def _norm(vector: list[float]) -> float:
    """The length of a vector."""
    return math.sqrt(_dot(vector, vector))


def _normalise(vector: list[float]) -> list[float]:
    """A unit vector. A zero vector stays zero."""
    length = _norm(vector)
    if length < 1e-12:
        return list(vector)
    return [value / length for value in vector]


def kmeans(vectors: list[list[float]], k: int,
           iterations: int = 40,
           seed: int = 42) -> tuple[list[int], list[list[float]]]:
    """The cluster of every vector and the centres.

    The initial centres are the vectors that are farthest from each
    other - a better start than random, and it is deterministic.
    """
    if not vectors:
        return [], []
    if k >= len(vectors):
        return list(range(len(vectors))), [list(v) for v in vectors]

    normalised = [_normalise(vector) for vector in vectors]

    # The start: the first vector, then the one that is farthest from
    # it, and so on.
    centres = [normalised[0]]
    while len(centres) < k:
        best_index = 0
        best_distance = -1.0
        for index, vector in enumerate(normalised):
            distance = min(1.0 - _dot(vector, centre) for centre in centres)
            if distance > best_distance:
                best_distance = distance
                best_index = index
        centres.append(normalised[best_index])

    assignment = [0] * len(vectors)
    generator = random.Random(seed)

    for _ in range(iterations):
        changed = False
        for index, vector in enumerate(normalised):
            best = 0
            best_score = -2.0
            for centre_index, centre in enumerate(centres):
                score = _dot(vector, centre)
                if score > best_score:
                    best_score = score
                    best = centre_index
            if assignment[index] != best:
                assignment[index] = best
                changed = True
        if not changed:
            break

        # The new centres: the mean of the members. The mean of unit
        # vectors is not a unit vector, so it is normalised again.
        for centre_index in range(len(centres)):
            members = [normalised[index]
                       for index, group in enumerate(assignment)
                       if group == centre_index]
            if not members:
                # An empty cluster: move it to a random member.
                centres[centre_index] = normalised[
                    generator.randrange(len(normalised))]
                continue
            dimension = len(members[0])
            mean = [0.0] * dimension
            for vector in members:
                for position in range(dimension):
                    mean[position] += vector[position]
            centres[centre_index] = _normalise(
                [value / len(members) for value in mean])

    return assignment, centres


def _spread(vectors: list[list[float]], centre: list[float],
            members: list[int]) -> float:
    """The mean distance of the members to the centre."""
    if not members:
        return 0.0
    total = 0.0
    for index in members:
        total += 1.0 - _dot(_normalise(vectors[index]), centre)
    return total / len(members)


def cluster(entries: list[tuple[str, list[float]]],
            k: int = 5) -> list[Cluster]:
    """Group the facts into topics.

    ``k`` is the wanted number of clusters - the number of buttons the
    user wants to see. It is not the number of topics there are.

    **A topic that is split in two is worse than one topic too many.**
    The user sees a button per topic, and a topic that comes in two
    halves is the same topic twice. Two things prevent that:

    - With few facts there are not enough for the wanted number of
      clusters. Five facts and five buttons gave five blocks of one -
      every fact was a cluster of its own.
    - Two clusters whose centres are nearly the same are the same
      topic. They are merged.

    So the number of clusters follows the number of facts, and what
    comes out is merged when it is one thing.
    """
    if not entries:
        return []

    texts = [text for text, _ in entries]
    vectors = [vector for _, vector in entries]

    # **Fewer facts, fewer topics.** Every cluster wants at least two
    # members, or every fact is a topic of its own - which is what
    # happened with four facts and five buttons: four blocks of one
    # topic.
    #
    # The limit is only a limit. What makes a topic one topic is the
    # merge below: two clusters whose centres are nearly the same
    # are the same topic, however many the user wanted.
    count = len(entries)
    wanted = max(1, min(k, count // 2 if count >= 2 else 1))

    assignment, centres = kmeans(vectors, wanted)

    clusters: list[Cluster] = []
    for centre_index in range(len(centres)):
        members = [index for index, group in enumerate(assignment)
                   if group == centre_index]
        if not members:
            continue
        facts = [texts[index] for index in members]
        clusters.append(Cluster(
            label=facts[0],
            facts=facts,
            size=len(facts),
            spread=_spread(vectors, centres[centre_index], members),
        ))

    # **The clusters whose centres are nearly the same are the same
    # topic.** Two halves of one topic are the one topic twice, and
    # the user sees two buttons for it. The centres are compared, and
    # the members of the second one go to the first.
    clusters = _merge_similar(clusters, vectors, centres, assignment)

    # The biggest cluster first: the UI shows the most used topic at
    # the left.
    clusters.sort(key=lambda cluster: (-cluster.size, cluster.label))
    return clusters


#: Two centres closer than this are the same topic.
MERGE_SIMILARITY = 0.92


def _merge_similar(clusters: list[Cluster], vectors: list[list[float]],
                   centres: list[list[float]],
                   assignment: list[int]) -> list[Cluster]:
    """Merge the clusters that are one topic.

    The clusters are compared by their centres. When two centres are
    nearly the same, the topics are nearly the same - and the facts
    of the smaller one go to the bigger one.
    """
    if len(clusters) < 2:
        return clusters

    merged: list[Cluster] = []
    used: set[int] = set()

    for index, cluster in enumerate(clusters):
        if index in used:
            continue
        facts = list(cluster.facts)
        members = [position for position, group in enumerate(assignment)
                   if group == index]
        for other in range(index + 1, len(clusters)):
            if other in used:
                continue
            if _dot(centres[index], centres[other]) >= MERGE_SIMILARITY:
                facts.extend(clusters[other].facts)
                members.extend(position for position, group
                               in enumerate(assignment) if group == other)
                used.add(other)
        used.add(index)
        merged.append(Cluster(
            label=facts[0],
            facts=facts,
            size=len(facts),
            spread=_spread(vectors, centres[index], members),
        ))

    return merged


def name_clusters(clusters: list[Cluster], llm=None) -> list[Cluster]:
    """Give every cluster a short name.

    The LLM is asked for two or three words. When it is not available
    - the models are suspended, or the answer is not usable - the
    first fact of the cluster stays as the name. A wrong name is
    worse than a long one.
    """
    if not clusters:
        return clusters

    if llm is None:
        return clusters

    for cluster in clusters:
        facts = "\n".join(f"- {fact}" for fact in cluster.facts[:8])
        prompt = (
            "These facts belong to one topic. Give the topic a name of "
            "two or three words, in English, in the same language as "
            "the facts. Answer with the name only, nothing else.\n\n"
            f"{facts}"
        )
        try:
            answer = llm(prompt)
        except Exception:
            continue
        if not answer:
            continue
        label = str(answer).strip().strip('"').splitlines()[0].strip()
        # A name that is too long is not a name.
        if label and len(label) <= 40:
            cluster.label = label

    return clusters


def palette(count: int) -> list[str]:
    """The colours for the topic buttons.

    They are spread over the hue circle and keep a similar lightness,
    so no button is brighter than the others - the colour says which
    topic it is, not which one is more important.
    """
    if count <= 0:
        return []
    colours = []
    for index in range(count):
        hue = int(round(index * 360 / count))
        colours.append(f"hsl({hue} 72% 58%)")
    return colours

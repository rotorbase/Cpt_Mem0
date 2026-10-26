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
    #: When the topic was talked about the last time. The topics the
    #: user works on are the ones that were talked about recently.
    last_seen: float = 0.0

    #: The colour of the button. **It belongs to the topic** - the
    #: placeholders in the row would otherwise shift the colours of
    #: the topics that follow them.
    colour: str = ""
    #: The same colour, transparent - the face of the key.
    fill: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "colour": self.colour,
            "fill": self.fill,
            "facts": self.facts,
            "size": self.size,
            "spread": round(self.spread, 4),
            "last_seen": self.last_seen,
            # **The key of the topic.** The name comes from the model
            # and the model names it differently tomorrow; the first
            # fact stays the same. That is what the star is stored on.
            "fact": self.facts[0] if self.facts else "",
        }


#: How many words a name on a button may have.
NAME_WORDS = 3


def shorten(text: str, words: int = NAME_WORDS) -> str:
    """A name that fits on a button.

    **The buttons are small and there are many of them.** Without a
    name that fits, the text of a fact becomes the name of the topic
    - and a fact is a sentence, not a label. The first words carry
    the meaning; the rest is cut off.
    """
    text = (text or "").strip()
    if not text:
        return ""
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]) + "…"


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

    The entries are ``(text, vector)`` or ``(text, vector, when)``.
    """
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

    texts = [entry[0] for entry in entries]
    vectors = [entry[1] for entry in entries]
    stamps = [entry[2] if len(entry) > 2 else 0.0 for entry in entries]

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
            # **A name that fits on the button.** Without a model to
            # name the topic, the first fact is the name - and a fact
            # is a sentence.
            label=shorten(facts[0]),
            facts=facts,
            size=len(facts),
            spread=_spread(vectors, centres[centre_index], members),
        ))

    # **The clusters whose centres are nearly the same are the same
    # topic.** Two halves of one topic are the one topic twice, and
    # the user sees two buttons for it. The centres are compared, and
    # the members of the second one go to the first.
    clusters = _merge_similar(clusters, vectors, centres, assignment)

    # **When the topic was talked about.** The buttons show the topics
    # the user works on now, not the ones with the most facts. The
    # newest fact of a cluster decides when the topic was last alive.
    for cluster in clusters:
        cluster.last_seen = max(
            (stamps[index] for index, text in enumerate(texts)
             if text in cluster.facts), default=0.0)

    # The newest topic first.
    clusters.sort(key=lambda cluster: (-cluster.last_seen, -cluster.size,
                                      cluster.label))
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
            # **A name that fits on the button.** A fact is a
            # sentence, and a sentence does not fit on a button.
            label=shorten(facts[0]),
            facts=facts,
            size=len(facts),
            spread=_spread(vectors, centres[index], members),
        ))

    return merged


#: How many topics are named by the model. The others keep the
#: shortened fact as their name.
#:
#: **The model is asked once per topic.** Eighteen topics would be
#: eighteen questions before the window shows anything - and the
#: window waits for the answer. The first ones are the strongest
#: ones, and they are the ones the user sees lit.
NAMED_TOPICS = 8


def _is_own_name(label: str, facts: list[str]) -> bool:
    """Whether the name is a name and not the fact again.

    **A name is a new word.** "Fahrrad" is a name for the fact about
    riding a bicycle - "Bicycle" is the fact in a new hat. The model
    answers in the language of the facts when it does not understand
    the instruction, and then the button says the same thing twice.

    The test: the name has to contain a word that is not in the
    facts. That word is the name.
    """
    words = [word.lower().strip(".,!?;:\"'")
             for word in label.split()]
    words = [word for word in words if len(word) > 2]
    if not words:
        return False

    text = " ".join(facts).lower()
    for word in words:
        if word not in text:
            return True
    return False


def name_clusters(clusters: list[Cluster], llm=None,
                  cache: dict[str, str] | None = None) -> list[Cluster]:
    """Give every cluster a short name.

    The LLM is asked for two or three words. When it is not available
    - the models are suspended, or the answer is not usable - the
    first fact of the cluster stays as the name. A wrong name is
    worse than a long one.

    **The cache.** The name of a topic does not change from one
    second to the next, and the window asks for the topics every few
    seconds. Without the cache every poll would send a dozen
    questions to the model and the card would do nothing but answer
    them. The key is the first fact of the cluster - the one that
    carries the meaning.
    """
    if not clusters:
        return clusters

    if llm is None:
        return clusters

    for number, cluster in enumerate(clusters):
        cluster.label = shorten(cluster.label)
        if cache is not None and cluster.facts:
            remembered = cache.get(cluster.facts[0])
            if remembered:
                cluster.label = remembered
                continue
        if number >= NAMED_TOPICS:
            # **The rest keep the fact.** The window would wait for
            # eighteen answers otherwise.
            continue
        facts = "\n".join(f"- {fact}" for fact in cluster.facts[:8])
        # **The prompt is in German.** A small model follows an
        # instruction in its own language better - and the name is
        # what the user reads on a button, so it has to come out in
        # German even when the facts are in English.
        prompt = (
            "Diese Fakten gehören zu einem Thema. Nenne das Thema in "
            "**einem oder zwei Wörtern auf Deutsch**.\n\n"
            "**Nenne das Konkrete, nicht die Kategorie.** Geht es um "
            "Märchen, ist der Name \"Märchen\" und nicht \"Vorlieben\" "
            "oder \"Hobbys\". Der Knopf ist ein Thema, das der Nutzer "
            "wiedererkennt.\n\n"
            "Antworte mit dem Namen auf Deutsch, sonst nichts. Kein "
            "Satz, keine Erklärung, keine Anführungszeichen.\n\n"
            "Beispiele:\n"
            "- Fakten über Fahrradfahren → Fahrrad\n"
            "- Fakten über Kräuter in der Küche → Kräuter\n"
            "- Fakten über den Namen des Assistenten → Assistent\n"
            "- Fakten über Urlaub am Meer → Urlaub\n\n"
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
            # **And a name that is the fact again is not a name
            # either.** The model answered in the language of the
            # facts - and the button would say what it said before.
            if not _is_own_name(label, cluster.facts):
                continue
            cluster.label = label
            if cache is not None and cluster.facts:
                cache[cluster.facts[0]] = label

    return clusters


#: How many topics shine. The others are there, but quiet - like
#: the keys of a keyboard that are not being played.
LIT_TOPICS = 6


def palette(count: int) -> list[str]:
    """The colours for the topic buttons.

    **Like a keyboard of a synthesizer.** The colours are neon - the
    keys glow - and the six that are being played right now shine
    full, the rest glow quiet. The colour says which topic it is,
    the brightness says how strong it is.
    """
    if count <= 0:
        return []
    colours = []
    for index in range(count):
        hue = int(round(index * 360 / count))
        if index < LIT_TOPICS:
            colours.append(f"hsl({hue}, 95%, 62%)")
        else:
            colours.append(f"hsl({hue}, 60%, 38%)")
    return colours


def fills(count: int) -> list[str]:
    """The colour of the button, transparent.

    **The old syntax, with the comma.** The window runs on an
    Electron that does not know the new one - and a colour that the
    browser does not understand is no colour at all.
    """
    if count <= 0:
        return []
    result = []
    for index in range(count):
        hue = int(round(index * 360 / count))
        if index < LIT_TOPICS:
            result.append(f"hsla({hue}, 95%, 62%, 0.16)")
        else:
            result.append(f"hsla({hue}, 60%, 38%, 0.10)")
    return result

"""Tests for the topic clustering.

The UI shows three to six topic buttons, filled from what is in the
store. The clustering decides what the buttons mean, so it has to
group the facts by their content and not by their order.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.topics import (  # noqa: E402
    Cluster,
    cluster,
    kmeans,
    name_clusters,
    palette,
)


def _vector(values):
    """A unit vector from a few numbers."""
    length = sum(value * value for value in values) ** 0.5
    return [value / length for value in values]


class TestKmeans:
    def test_it_separates_two_groups(self):
        """Two groups that point in different directions."""
        vectors = [
            _vector([1, 0]), _vector([0.9, 0.1]),
            _vector([-1, 0]), _vector([-0.9, -0.1]),
        ]
        assignment, centres = kmeans(vectors, 2)
        assert len(centres) == 2
        assert assignment[0] == assignment[1]
        assert assignment[2] == assignment[3]
        assert assignment[0] != assignment[2]

    def test_it_is_deterministic(self):
        """The same input gives the same clusters - the UI must not
        shuffle the buttons on every reload."""
        vectors = [_vector([1, 0]), _vector([0.9, 0.1]),
                   _vector([0, 1]), _vector([0.1, 0.9])]
        first = kmeans(vectors, 2)
        second = kmeans(vectors, 2)
        assert first[0] == second[0]

    def test_more_clusters_than_vectors(self):
        assignment, centres = kmeans([_vector([1, 0])], 5)
        assert len(centres) == 1
        assert assignment == [0]

    def test_no_vectors(self):
        assignment, centres = kmeans([], 3)
        assert assignment == []
        assert centres == []

    def test_every_vector_gets_a_cluster(self):
        vectors = [_vector([1, 0]), _vector([0, 1]), _vector([1, 1])]
        assignment, _ = kmeans(vectors, 3)
        assert len(assignment) == 3
        assert all(0 <= group < 3 for group in assignment)


class TestCluster:
    def test_it_groups_by_content(self):
        entries = [
            ("The build server has 64 GB of RAM.", _vector([1, 0])),
            ("The build server runs the tests.", _vector([0.95, 0.05])),
            ("The coffee machine is automatic.", _vector([0, 1])),
            ("The coffee machine needs beans.", _vector([0.05, 0.95])),
        ]
        clusters = cluster(entries, k=2)
        assert len(clusters) == 2

        by_size = sorted(clusters, key=lambda c: -c.size)
        assert by_size[0].size == 2

        # The two build-server facts belong together.
        groups = [set(c.facts) for c in clusters]
        assert any({"The build server has 64 GB of RAM.",
                    "The build server runs the tests."} <= group
                   for group in groups)

    def test_the_biggest_cluster_is_first(self):
        """The UI shows the most used topic at the left."""
        entries = [(f"Fact {index}", _vector([1, 0]))
                   for index in range(5)]
        entries.append(("Other", _vector([0, 1])))
        clusters = cluster(entries, k=2)
        assert clusters[0].size >= clusters[-1].size

    def test_no_facts(self):
        assert cluster([], k=5) == []

    def test_every_fact_is_in_a_cluster(self):
        """Nothing may be lost - a fact that is in no cluster is a
        fact the user cannot find."""
        entries = [(f"Fact {index}", _vector([1, index * 0.1]))
                   for index in range(10)]
        clusters = cluster(entries, k=3)
        found = [fact for c in clusters for fact in c.facts]
        assert sorted(found) == sorted(fact for fact, _ in entries)


class TestNaming:
    def test_the_llm_gives_the_name(self):
        clusters = [Cluster(label="a long fact", facts=["a long fact"])]
        name_clusters(clusters, llm=lambda prompt: "Infrastructure")
        assert clusters[0].label == "Infrastructure"

    def test_without_an_llm_the_fact_stays(self):
        clusters = [Cluster(label="a long fact", facts=["a long fact"])]
        name_clusters(clusters, llm=None)
        assert clusters[0].label == "a long fact"

    def test_a_failing_llm_leaves_the_fact(self):
        def broken(prompt):
            raise RuntimeError("the models are suspended")

        clusters = [Cluster(label="a long fact", facts=["a long fact"])]
        name_clusters(clusters, llm=broken)
        assert clusters[0].label == "a long fact"

    def test_a_too_long_name_is_refused(self):
        """A name that is too long is not a name."""
        clusters = [Cluster(label="short", facts=["short"])]
        name_clusters(clusters, llm=lambda prompt: "x" * 80)
        assert clusters[0].label == "short"

    def test_a_quoted_name_is_cleaned(self):
        clusters = [Cluster(label="short", facts=["short"])]
        name_clusters(clusters, llm=lambda prompt: '"Infrastructure"')
        assert clusters[0].label == "Infrastructure"


class TestPalette:
    def test_there_is_a_colour_for_every_cluster(self):
        colours = palette(5)
        assert len(colours) == 5
        assert len(set(colours)) == 5

    def test_no_clusters(self):
        assert palette(0) == []

    def test_the_colours_are_spread_out(self):
        """Two buttons next to each other must not have the same
        colour."""
        colours = palette(6)
        hues = [int(colour.split("(")[1].split(",")[0])
                for colour in colours]
        assert sorted(hues) == [0, 60, 120, 180, 240, 300]


class TestTheNameFitsOnTheButton:
    """A fact is a sentence. A sentence does not fit on a button.

    The name of a topic comes from the model - and without a model
    from the first fact of the group. Both have to be short.
    """

    def test_the_name_is_cut_to_three_words(self):
        from cpt_mem0.topics import shorten
        assert shorten("User enjoys riding a bicycle very much.") == \
            "User enjoys riding…"
        assert shorten("Märchen") == "Märchen"
        assert shorten("") == ""

    def test_the_group_without_a_model_has_a_short_name(self):
        from cpt_mem0.topics import cluster
        entries = [
            ("User enjoys riding a bicycle very much.", [1.0, 0.0]),
            ("The user loves parsley as a favorite herb or ingredient.",
             [0.0, 1.0]),
        ]
        groups = cluster(entries, k=5)
        assert groups
        for group in groups:
            assert len(group.label) <= 24
            assert "…" in group.label


class TestTheNamingIsBounded:
    """The model is asked once per topic - and not for all of them.

    The window waits for the answer, and eighteen questions before
    anything appears is eighteen questions too many.
    """

    def test_not_every_topic_is_asked(self):
        from cpt_mem0.topics import NAMED_TOPICS, name_clusters

        asked = []

        def llm(prompt):
            asked.append(prompt)
            return "Fahrrad"

        clusters = []
        for index in range(18):
            from cpt_mem0.topics import Cluster
            clusters.append(Cluster(label=f"Thema {index}",
                                    facts=[f"Fakt {index}"], size=1))
        name_clusters(clusters, llm=llm)
        assert len(asked) == NAMED_TOPICS

    def test_the_names_are_remembered(self):
        from cpt_mem0.topics import Cluster, name_clusters

        asked = []

        def llm(prompt):
            asked.append(prompt)
            return "Fahrrad"

        cache = {}
        for _ in range(2):
            clusters = [Cluster(label="User enjoys riding a bicycle",
                                facts=["User enjoys riding a bicycle"],
                                size=1)]
            name_clusters(clusters, llm=llm, cache=cache)
        # The second time the name is known.
        assert len(asked) == 1
        assert cache["User enjoys riding a bicycle"] == "Fahrrad"


class TestTheNameIsANewWord:
    """A name is a new word.

    The model answers in the language of the facts when it does not
    understand the instruction - and then the button says what it
    said before, only shorter.
    """

    def test_a_name_that_is_the_fact_again_is_refused(self):
        from cpt_mem0.topics import Cluster, name_clusters

        def llm(prompt):
            return "Bicycle riding"

        clusters = [Cluster(label="User enjoys riding a bicycle very much.",
                            facts=["User enjoys riding a bicycle very much."],
                            size=1)]
        name_clusters(clusters, llm=llm)
        # The name stayed the shortened fact.
        assert "Bicycle" not in clusters[0].label

    def test_a_german_name_is_taken(self):
        from cpt_mem0.topics import Cluster, name_clusters

        def llm(prompt):
            return "Fahrrad"

        clusters = [Cluster(label="User enjoys riding a bicycle very much.",
                            facts=["User enjoys riding a bicycle very much."],
                            size=1)]
        name_clusters(clusters, llm=llm)
        assert clusters[0].label == "Fahrrad"

    def test_the_prompt_is_in_german(self):
        """A small model follows an instruction in its own language."""
        from cpt_mem0.topics import Cluster, name_clusters

        asked = []

        def llm(prompt):
            asked.append(prompt)
            return "Fahrrad"

        clusters = [Cluster(label="User enjoys riding a bicycle very much.",
                            facts=["User enjoys riding a bicycle very much."],
                            size=1)]
        name_clusters(clusters, llm=llm)
        assert "Deutsch" in asked[0]


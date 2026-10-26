"""The topics of the store - the names the buttons carry.

**The buttons do not invent their names.** A button that is called
"Thema 7" tells the user nothing, and a button whose name changes
every time the store grows asks for a new button every time. So the
names are fixed here, chosen after the way mem0 behaves: it stores
statements about the person, their situation, their work, their
preferences, their plans, their equipment and their environment.

Eighteen names in three groups:

- **Alltag** - who the user is and what their days look like
- **Interessen** - what they spend their time on
- **Umfeld** - the world around them

A button is white while nothing is stored for it, and it lights up
in its own colour as soon as something is. The store fills the
buttons by itself - the user does not have to file anything away.

**The words are a fallback, not the answer.** The model is asked
which of the eighteen topics a group of facts belongs to. The words
are there for the case that the model is not running - then a fact is
filed by the words it contains, and a fact that fits nothing stays
where it is.
"""

from __future__ import annotations

from typing import Any

#: The three groups, in the order they appear.
GROUPS: list[str] = ["Alltag", "Interessen", "Umfeld"]

#: The eighteen topics.
#:
#: Every topic has a name, the words that point to it, and a line
#: that tells the model what the topic is about.
TOPICS: list[dict[str, Any]] = [
    # -- Alltag ---------------------------------------------------------
    {
        "name": "Beruf",
        "group": "Alltag",
        "words": ["beruf", "arbeit", "job", "firma", "büro", "buro",
                  "projekt", "meeting", "kollege", "chef", "arbeitgeber",
                  "schicht", "werk", "büro", "branche", "gehalt", "stelle"],
        "line": "die Arbeit, der Beruf, die Firma, die Kollegen",
    },
    {
        "name": "Wohnen",
        "group": "Alltag",
        "words": ["wohnung", "haus", "miete", "wohnen", "zimmer", "küche",
                  "bad", "möbel", "haushalt", "umzug", "nachbar", "garten",
                  "heizung", "wohnzimmer", "schlafzimmer"],
        "line": "die Wohnung, das Haus, der Haushalt, die Nachbarn",
    },
    {
        "name": "Familie",
        "group": "Alltag",
        "words": ["familie", "frau", "mann", "ehemann", "ehefrau", "kind",
                  "kinder", "sohn", "tochter", "mutter", "vater", "eltern",
                  "freund", "freundin", "beziehung", "heirat", "geschwister"],
        "line": "die Familie, der Partner, die Kinder, die Freunde",
    },
    {
        "name": "Gesundheit",
        "group": "Alltag",
        "words": ["gesund", "gesundheit", "sport", "training", "arzt",
                  "zahnarzt", "schlaf", "ernährung", "diät", "krank",
                  "medikament", "fitness", " jogging", "laufen", "schmerz",
                  "therapie", "körper"],
        "line": "der Körper, die Gesundheit, der Sport, der Arzt",
    },
    {
        "name": "Geld",
        "group": "Alltag",
        "words": ["geld", "gehalt", "budget", "sparen", "versicherung",
                  "steuer", "kredit", "rente", "konto", "preis", "einkauf",
                  "einkommen", "ausgabe", "rechnung", "kosten", "lohn"],
        "line": "das Geld, das Einkommen, die Versicherung, die Preise",
    },
    {
        "name": "Zeit",
        "group": "Alltag",
        "words": ["routine", "gewohnheit", "morgens", "abends", "woche",
                  "wochenende", "urlaub", "termin", "kalender", "früh",
                  "wecker", "feierabend", "pause", "plan", "tagesplan",
                  "alltag", "wann"],
        "line": "der Tagesablauf, die Gewohnheiten, die Termine",
    },
    # -- Interessen -----------------------------------------------------
    {
        "name": "Musik",
        "group": "Interessen",
        "words": ["musik", "song", "lied", "album", "band", "konzert",
                  "playlist", "sänger", "aufnahme", "instrument", "gitarre",
                  "klavier", "radio", "spotify", "beat", "takt"],
        "line": "die Musik, die Bands, die Konzerte, das Instrument",
    },
    {
        "name": "Film",
        "group": "Interessen",
        "words": ["film", "serie", "staffel", "netflix", "schauspieler",
                  "folge", "kino", "fernsehen", "regisseur", "crew",
                  "serie", "film", "oscar", "trailer"],
        "line": "die Filme, die Serien, das Fernsehen, das Kino",
    },
    {
        "name": "Bücher",
        "group": "Interessen",
        "words": ["buch", "roman", "lesen", "autor", "kapitel", "literatur",
                  "bibliothek", "leser", "verlag", "text", "poesie",
                  "lyrik", "bibliothek"],
        "line": "die Bücher, das Lesen, die Autoren",
    },
    {
        "name": "Essen",
        "group": "Interessen",
        "words": ["essen", "kochen", "rezept", "restaurant", "pizza",
                  "mahlzeit", "speise", "getränk", "kaffee", "wein", "bier",
                  "frühstück", "abendessen", "küche", "menü", "koch"],
        "line": "das Essen, das Kochen, die Rezepte, das Restaurant",
    },
    {
        "name": "Reisen",
        "group": "Interessen",
        "words": ["reise", "urlaub", "flug", "flugzeug", "hotel", "stadt",
                  "land", "strand", "ausland", "zug", "unterkunft",
                  "sightseeing", "insel", "berg", "reise", "reisen"],
        "line": "die Reisen, der Urlaub, die Orte, die Flüge",
    },
    {
        "name": "Hobbys",
        "group": "Interessen",
        "words": ["hobb", "basteln", "garten", "foto", "handarbeit",
                  "brett", "karten", "modell", "sammeln", "angeln",
                  "kamin", "möbel", "werkzeug", "hobby", "bastelei"],
        "line": "die Hobbys, das Basteln, der Garten, die Fotos",
    },
    # -- Umfeld ---------------------------------------------------------
    {
        "name": "Technik",
        "group": "Umfeld",
        "words": ["computer", "laptop", "pc", "handy", "telefon",
                  "smartphone", "router", "software", "programm", "server",
                  "bildschirm", "tastatur", "windows", "linux", "grafikkarte",
                  "platine", "elektronik", "chipsatz"],
        "line": "die Technik: Rechner, Telefon, Netzwerk, Software",
    },
    {
        "name": "Auto & Rad",
        "group": "Umfeld",
        "words": ["auto", "rad", "fahrrad", "fahrzeug", "motor", "tankstelle",
                  "bremse", "reifen", "e-bike", "moped", "schienenersatz",
                  "fahrrad", "werkstatt", "tÜV", "kennzeichen", "garage"],
        "line": "das Auto, das Fahrrad, die Werkstatt, die Fahrt",
    },
    {
        "name": "Natur",
        "group": "Umfeld",
        "words": ["natur", "wald", "see", "meer", "fluss", "pflanze",
                  "blume", "wetter", "sonne", "regen", "wind", "himmel",
                  "wald", "gewitter", "schnee", "baum", "insel"],
        "line": "die Natur, das Wetter, der Wald, das Meer",
    },
    {
        "name": "Tiere",
        "group": "Umfeld",
        "words": ["hund", "katze", "tier", "haustier", "vogel", "pferd",
                  "aquarium", "tierarzt", "fisch", "hamster", "kaninchen",
                  "katze", "hund", "tierheim", "zoo"],
        "line": "die Tiere, die Haustiere, der Tierarzt",
    },
    {
        "name": "Politik",
        "group": "Umfeld",
        "words": ["politik", "regierung", "partei", "wahl", "krise", "krieg",
                  "europa", "bundesland", "nachrichten", "gesetz", "parlament",
                  "minister", "politik", "demo", "streik", "konzern"],
        "line": "die Politik, die Nachrichten, die Wirtschaft",
    },
    {
        "name": "Lernen",
        "group": "Umfeld",
        "words": ["lernen", "studium", "schule", "universität", "kurs",
                  "wissen", "frage", "weiterbildung", "sprache", "englisch",
                  "deutsch", "prüfung", "seminar", "vorlesung", "aufgabe",
                  "mathe", "mathematik", "programmieren", "python"],
        "line": "das Lernen, die Schule, das Studium, die Fächer",
    },
]


def by_name() -> dict[str, dict[str, Any]]:
    """The topics by their name."""
    return {str(topic["name"]): topic for topic in TOPICS}


def groups() -> list[dict[str, Any]]:
    """The three groups with their topics, in the order of the groups."""
    result = []
    for name in GROUPS:
        result.append({
            "name": name,
            "topics": [topic for topic in TOPICS if topic["group"] == name],
        })
    return result


def menu() -> str:
    """The menu for the model: the eighteen names with their numbers.

    **The model chooses from a list.** Asking a small model for a
    free name gives a free name - and a free name in English. Asking
    it for a number gives a number, and the number is one of the
    eighteen topics.
    """
    lines = []
    for number, topic in enumerate(TOPICS, 1):
        lines.append(f"{number}. {topic['name']} - {topic['line']}")
    return "\n".join(lines)


def _words(text: str) -> list[str]:
    """The words of a text, lower case and without the punctuation."""
    cleaned = []
    for raw in text.lower().replace("-", " ").replace(",", " ").split():
        word = raw.strip(".:;!?()\"'")
        if len(word) > 2:
            cleaned.append(word)
    return cleaned


def score(fact: str) -> dict[str, int]:
    """How well a fact fits every topic.

    **The words are a fallback.** A fact that is stored in German
    and mentions "Rad" belongs to "Auto & Rad" - and the word is in
    the topic. A German word inside a word also counts: "Fahrrad"
    contains "rad".
    """
    words = _words(fact)
    result: dict[str, int] = {}
    for topic in TOPICS:
        points = 0
        for word in topic["words"]:
            for candidate in words:
                if candidate == word:
                    points += 2
                elif word in candidate:
                    # "Fahrrad" contains "rad".
                    points += 1
                elif candidate in word and len(candidate) > 3:
                    points += 1
        if points:
            result[str(topic["name"])] = points
    return result


def best(fact: str) -> str:
    """The topic a fact belongs to - or the empty string.

    **The order of the list breaks the tie.** The topics are in the
    order of the groups, and the first one wins when two are equally
    good - so the answer does not change from one day to the next.
    """
    points = score(fact)
    if not points:
        return ""
    order = {str(topic["name"]): number
             for number, topic in enumerate(TOPICS)}
    return max(points.items(),
               key=lambda item: (item[1], -order.get(item[0], 0)))[0]

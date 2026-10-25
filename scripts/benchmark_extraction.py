"""Measure the extraction quality of a model.

The catalogue ranks the models by extraction quality. That ranking
has to come from a measurement, not from an impression - otherwise
the planner picks a model on a guess and the service stores nothing
useful.

**What is measured, per input sentence:**

| Metric | What it means |
|---|---|
| `json_validity` | the answer is valid JSON in the format mem0 expects |
| `atomicity` | one fact per entry, not a paragraph |
| `recall` | the fact from the input is found again by a search |
| `empty_rate` | the model extracted nothing |
| `language_fidelity` | the answer is in the language of the input |

**The corpus** is deliberately multilingual and neutral: German and
English sentences about ordinary things. No project, no person,
nothing that identifies the user.

**The recall check is the expensive part.** It needs the embedder,
because a fact that cannot be found again is a fact the service does
not have.

Usage:
    python scripts/benchmark_extraction.py --model-file <path> \\
        --embedder-file <path>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.engine import Engine, llama_endpoint  # noqa: E402
from cpt_mem0.supervisor import (  # noqa: E402
    ServerSpec,
    Supervisor,
    gpu_buffers,
    gpu_used_mib,
)

DEFAULT_EMBEDDER = (Path.home() / ".cache/huggingface/embed-models"
                    / "nomic-embed-text-v1.5.Q4_K_M.gguf")

EMBEDDER_FLAGS = [
    "--parallel", "1",
    "-c", "2048",
    "-ngl", "99",
    "--embeddings",
    "--pooling", "mean",
    "--kv-unified",
]

#: The corpus. Each entry: the sentence, the question that has to find
#: it, and the language it is in.
CORPUS = [
    ("The user rides a bicycle to the office every day.",
     "How does the user get to work?", "en"),
    ("The build server has 64 GB of RAM.",
     "How much memory does the build server have?", "en"),
    ("The coffee machine in the office is fully automatic.",
     "What kind of coffee machine is in the office?", "en"),
    ("The office is on the third floor.",
     "Which floor is the office on?", "en"),
    ("The backup runs every night at two in the morning.",
     "When does the backup run?", "en"),
    ("The team uses PostgreSQL for the production database.",
     "Which database does the team use in production?", "en"),
    ("The meeting room has a projector and a white board.",
     "What is in the meeting room?", "en"),
    ("The router is connected to the switch in the hallway.",
     "What is the router connected to?", "en"),
    ("Der Benutzer fährt jeden Tag mit dem Fahrrad zur Arbeit.",
     "Wie kommt der Benutzer zur Arbeit?", "de"),
    ("Der Build-Server hat 64 GB RAM.",
     "Wie viel RAM hat der Build-Server?", "de"),
    ("Die Kaffeemaschine im Büro ist vollautomatisch.",
     "Welche Art von Kaffeemaschine steht im Büro?", "de"),
    ("Das Büro befindet sich im dritten Stock.",
     "In welchem Stock liegt das Büro?", "de"),
    ("Das Backup läuft jede Nacht um zwei Uhr.",
     "Wann läuft das Backup?", "de"),
    ("Das Team nutzt PostgreSQL für die Produktionsdatenbank.",
     "Welche Datenbank nutzt das Team in Produktion?", "de"),
    ("Der Meetingraum hat einen Beamer und eine Tafel.",
     "Was ist im Meetingraum?", "de"),
    ("Der Router ist mit dem Switch im Flur verbunden.",
     "Womit ist der Router verbunden?", "de"),
    ("El usuario va al trabajo en bicicleta todos los días.",
     "¿Cómo llega el usuario al trabajo?", "es"),
    ("El servidor de compilación tiene 64 GB de RAM.",
     "¿Cuánta memoria tiene el servidor de compilación?", "es"),
    ("La máquina de café de la oficina es totalmente automática.",
     "¿Qué tipo de máquina de café hay en la oficina?", "es"),
    ("L'utilisateur va au travail à vélo tous les jours.",
     "Comment l'utilisateur va-t-il au travail ?", "fr"),
    ("Le serveur de build possède 64 Go de RAM.",
     "Combien de RAM le serveur de build possède-t-il ?", "fr"),
    ("La machine à café du bureau est entièrement automatique.",
     "Quel type de machine à café y a-t-il au bureau ?", "fr"),
]

#: The words that have to survive the extraction. The check is on the
#: key terms, not on the wording - the model rephrases.
KEY_TERMS = {
    "bicycle": ["bicycle", "bike", "fahrrad", "bicicleta", "vélo"],
    "64 GB RAM": ["64 gb", "64 gb ram", "64 go", "64 gb de ram"],
    "coffee machine": ["coffee machine", "kaffeemaschine", "café",
                       "máquina de café", "machine à café"],
    "third floor": ["third floor", "dritten stock", "tercer piso",
                    "troisième étage"],
    "two in the morning": ["two", "2", "zwei", "dos", "deux"],
    "PostgreSQL": ["postgresql"],
    "projector": ["projector", "beamer", "proyector", "projecteur"],
    "router": ["router"],
}


class Log:
    """The log file and the console at the same time."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        self._handle.write(text + "\n")
        self._handle.flush()
        print(text, flush=True)

    def section(self, title: str) -> None:
        self.write()
        self.write(f"--- {title} ---")

    def close(self) -> None:
        self._handle.close()


def find_llama_server() -> Path | None:
    from cpt_mem0.install import current_installation

    installation = current_installation()
    if installation.binary is not None:
        return installation.binary
    return None


def planned_flags() -> list[str]:
    from cpt_mem0.catalogue import load_catalogue
    from cpt_mem0.planner import (
        Hardware,
        PlannerConfig,
        plan_for_hardware,
    )

    catalogue = load_catalogue(ROOT / "catalogue.json")
    hardware = Hardware(total_vram_mib=12288,
                        name="NVIDIA GeForce RTX 3060", vendor="nvidia")
    for entry in catalogue.sorted_entries():
        if entry.model_id != "Qwen3.5-9B":
            continue
        result = plan_for_hardware(
            hardware, [entry],
            PlannerConfig(embedder_backends=("llamacpp",)))
        if result.viable and result.candidate is not None:
            return result.candidate.as_flags()
    return ["-c", "67072", "--parallel", "2", "-ngl", "99",
            "--fit", "off", "--flash-attn", "on",
            "--reasoning", "off", "--no-context-shift",
            "--kv-unified", "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0"]


def has_all_terms(text: str, terms: list[str]) -> bool:
    """Whether the text contains the key terms."""
    lowered = text.lower()
    return any(term in lowered for term in terms)


def atomic(text: str) -> bool:
    """Whether the fact is one fact.

    A fact that carries three different subjects is not atomic - it
    cannot be updated or deleted on its own.
    """
    sentences = [part for part in re.split(r"[.!?]", text) if part.strip()]
    if len(sentences) > 1:
        return False
    # Two clauses joined by "and" are two facts.
    return len(re.findall(r"\band\b", text, re.IGNORECASE)) <= 1


#: The words that mark a language. They are counted as whole words,
#: with a boundary on both sides - " la " also sits in " llama ".
LANGUAGE_WORDS = {
    "de": ("der", "die", "das", "und", "nicht", "mit", "wie", "was", "von",
           "ist", "sich", "auf", "einen", "einem", "benutzer", "arbeit"),
    "es": ("el", "la", "los", "las", "y", "qué", "tiene", "máquina",
           "usuario", "trabajo", "servidor", "memoria"),
    "fr": ("le", "la", "les", "et", "à", "du", "des", "utilisateur",
           "machine", "travail", "serveur", "mémoire"),
}


def language_of(text: str) -> str:
    """A rough guess of the language.

    The words are counted as whole words. A plain substring count
    does not work: " la " is also in " llama ", and a German sentence
    with one " der " would not be recognised.
    """
    words = re.findall(r"[\wÀ-ÿ]+", text.lower())
    counts = {language: 0 for language in LANGUAGE_WORDS}
    for word in words:
        for language, markers in LANGUAGE_WORDS.items():
            if word in markers:
                counts[language] += 1
    best = max(counts, key=lambda key: counts[key])
    return best if counts[best] > 0 else "en"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--embedder-file", default=str(DEFAULT_EMBEDDER))
    parser.add_argument("--llm-port", type=int, default=8766)
    parser.add_argument("--embedder-port", type=int, default=8767)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N sentences (0 = all)")
    parser.add_argument("--log", default=str(ROOT / "results"
                                            / "extraction-benchmark.log"))
    parser.add_argument("--results", default=str(ROOT / "results"
                                                / "extraction-benchmark.json"))
    parser.add_argument("--free-mib", type=int, default=10000)
    args = parser.parse_args()

    log = Log(Path(args.log))
    corpus = CORPUS[:args.limit] if args.limit else CORPUS
    results: dict = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "sentences": len(corpus),
        "problems": [],
    }

    try:
        binary = find_llama_server()
        if binary is None:
            log.write("llama-server not found. Run 'cpt-mem0 init' first.")
            return 2

        from cpt_mem0.supervisor import gpu_free_mib

        free = gpu_free_mib()
        results["vram_free_before_mib"] = free
        log.write("=== the extraction benchmark ===")
        log.write(f"  VRAM free: {free} MiB")
        if args.free_mib and free is not None and free < args.free_mib:
            log.write(f"  the card has only {free} MiB free, the run needs "
                      f"{args.free_mib} MiB.")
            results["problems"].append(
                f"not enough VRAM: {free} MiB free, {args.free_mib} needed")
            return 3

        workdir = Path(tempfile.mkdtemp(prefix="cpt-mem0-bench-"))
        specs = [
            ServerSpec(name="embedder", binary=binary,
                       model=Path(args.embedder_file),
                       port=args.embedder_port, flags=EMBEDDER_FLAGS,
                       embeddings=True,
                       log_path=workdir / "embedder.log"),
            ServerSpec(name="llm", binary=binary,
                       model=Path(args.model_file), port=args.llm_port,
                       flags=planned_flags(),
                       log_path=workdir / "llm.log"),
        ]

        supervisor = Supervisor(specs)
        try:
            for spec in specs:
                log.section(f"starting the {spec.name}")
                supervisor.start(spec.name, timeout=args.timeout)
                buffers = gpu_buffers(spec.log_path)
                log.write(f"  device: {buffers['device']}, "
                          f"layers {buffers['offloaded_layers']}/"
                          f"{buffers['total_layers']}")
                results.setdefault("servers", {})[spec.name] = buffers

            log.section("the extraction")
            engine = Engine(
                llm=llama_endpoint(args.llm_port, "llm"),
                embedder=llama_endpoint(args.embedder_port, "nomic"),
                store_path=workdir / "store",
                dimensions=768,
                user_id="benchmark")
            engine.open()

            stored: list[tuple[str, str]] = []
            json_valid = 0
            atomic_count = 0
            empty = 0
            language_ok = 0
            total_time = 0.0

            for index, (sentence, _question, language) in enumerate(corpus):
                started = time.time()
                try:
                    outcome = engine.remember(sentence)
                except Exception as exc:
                    log.write(f"  [{index + 1}/{len(corpus)}] {sentence}")
                    log.write(f"    FAILED: {type(exc).__name__}: {exc}")
                    results["problems"].append(
                        f"{sentence}: {type(exc).__name__}: {exc}")
                    continue
                elapsed = time.time() - started
                total_time += elapsed

                facts = outcome.get("results") or []
                texts = [item.get("memory") if isinstance(item, dict)
                         else str(item) for item in facts]
                texts = [text for text in texts if text]

                log.write(f"  [{index + 1}/{len(corpus)}] {sentence}")
                log.write(f"    {len(texts)} facts in {elapsed:.1f}s")
                for text in texts:
                    log.write(f"      {text}")
                    stored.append((sentence, text))
                    if atomic(text):
                        atomic_count += 1
                    if language_of(text) == language:
                        language_ok += 1

                if not texts:
                    empty += 1
                else:
                    json_valid += 1

            # The recall: every stored fact has to be findable again.
            log.section("the recall")
            found_count = 0
            for sentence, question, _language in corpus:
                found = engine.recall(question, user_id="benchmark")
                texts = [item.get("memory") if isinstance(item, dict)
                         else str(item) for item in found]
                top = texts[0] if texts else ""

                # The right fact has to be in the first three.
                match = False
                for source, fact in stored:
                    if source == sentence and fact in texts[:3]:
                        match = True
                        break
                if match:
                    found_count += 1
                log.write(f"  {question}")
                log.write(f"    top: {top}")
                log.write(f"    correct in the top 3: {match}")

            total = len(corpus)
            summary = {
                "sentences": total,
                "json_validity": json_valid / total if total else 0,
                "atomicity": atomic_count / len(stored) if stored else 0,
                "recall": found_count / total if total else 0,
                "empty_rate": empty / total if total else 0,
                "language_fidelity": language_ok / len(stored) if stored else 0,
                "average_extraction_seconds": (
                    total_time / total if total else 0),
                "facts_stored": len(stored),
            }
            results["summary"] = summary

            log.section("the summary")
            for name, value in summary.items():
                if isinstance(value, float):
                    log.write(f"  {name:32} {value:.3f}")
                else:
                    log.write(f"  {name:32} {value}")

            engine.close()
        finally:
            supervisor.stop_all()
            log.write(f"  VRAM after: {gpu_used_mib()} MiB")
            results["vram_used_after_mib"] = gpu_used_mib()
            shutil.rmtree(workdir, ignore_errors=True)

        results["finished"] = datetime.now().isoformat(timespec="seconds")

    finally:
        log.write()
        try:
            target = Path(args.results)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(results, indent=2, default=str),
                              encoding="utf-8")
            log.write(f"the result is in {target}")
        except OSError as exc:
            log.write(f"the result could not be written: {exc}")
        log.write(f"the log is in {args.log}")
        log.close()

    return 1 if results["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

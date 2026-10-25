"""The model catalogue.

The catalogue maps a GPU to a concrete GGUF file. It is data, not logic:
the planner consumes it.

Two rules shape its structure:

1. It ships with the project, in ``catalogue.json``. It is never fetched
   from the network - a tool that needs internet to start is not
   self-contained.
2. Ranking is by *measured* extraction capability, not parameter count.
   A model without a measurement ranks below every measured model, and the
   plan output labels the basis. An unmeasured model is a guess; a
   measured one is evidence.

See SPEC section 17.4 and docs/catalogue.md.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .planner import QUANTISATION_ORDER

#: Where the shipped catalogue lives, relative to the package.
CATALOGUE_FILENAME = "catalogue.json"


@dataclass(frozen=True)
class Quantisation:
    """One GGUF file of one model."""

    name: str
    file_name: str
    file_size_bytes: int
    kv_bytes_per_token: int | None = None   # None -> derive from the profile
    kv_source: str = "formula"              # measured | formula | calibrate

    @property
    def file_size_mib(self) -> float:
        return self.file_size_bytes / (1024 * 1024)


@dataclass(frozen=True)
class ExtractionMeasurement:
    """Result of an extraction benchmark for one model.

    **``language_fidelity`` is not a failure rate.** mem0 rewrites
    every fact into English: the extraction prompt is English and the
    model follows it. Measured on the 9B, 7 of 8 German, Spanish and
    French facts came back in English.

    That is what makes the cross-language recall work - a German
    question finds a fact that was stored in English, because
    everything is in the same language. ``language_normalisation``
    says how much was rewritten.
    """

    recall: float
    json_validity: float
    atomicity: float
    language_fidelity: float
    empty_rate: float
    corpus: str
    date: str
    #: How much of the input language was rewritten into English.
    language_normalisation: float = 0.0

    @property
    def measured(self) -> bool:
        return True


@dataclass
class CatalogueEntry:
    """One model with its quantisations."""

    model_id: str
    repo: str
    params: int
    architecture: str
    license: str
    quantisations: list[Quantisation] = field(default_factory=list)
    extraction: ExtractionMeasurement | None = None
    rank_order: int = 0
    notes: str = ""
    #: GGUF geometry as read from the model header. Present so the
    #: catalogue can be checked against a real file without loading it.
    geometry: dict = field(default_factory=dict)
    #: The recurrent state buffer of a hybrid model, in MiB. Zero for a
    #: model without state-space layers.
    #:
    #: A hybrid model keeps the state of its SSM layers next to the KV
    #: cache. That buffer has its own size and does not scale with the
    #: context, so it has to be part of the budget.
    #:
    #: Measured on an RTX 3060 with llama.cpp 0.7.0-dev (build 11798):
    #:   llama_memory_recurrent: CUDA0 RS buffer size = 100.50 MiB
    recurrent_state_mib: float = 0.0
    #: Where the recurrent state size came from: "measured" or
    #: "formula". A model whose size is unknown is not plannable.
    recurrent_state_source: str = "none"

    @property
    def is_hybrid(self) -> bool:
        """Whether the model has state-space layers.

        The GGUF header says so: a hybrid architecture carries
        ``<arch>.ssm.state_size``.
        """
        return any(key.endswith("ssm.state_size")
                   for key in self.geometry)

    @property
    def rank_basis(self) -> str:
        return "measured" if self.extraction is not None else "unmeasured"

    def quantisation(self, name: str) -> Quantisation | None:
        for quant in self.quantisations:
            if quant.name == name:
                return quant
        return None

    def sorted_quantisations(self) -> list[Quantisation]:
        """Quantisations in preference order (best first)."""
        def rank(quant: Quantisation) -> int:
            try:
                return QUANTISATION_ORDER.index(quant.name)
            except ValueError:
                return len(QUANTISATION_ORDER)
        return sorted(self.quantisations, key=rank)

    def kv_bytes_from_geometry(self) -> int | None:
        """The KV constant implied by the stored geometry.

        Returns ``None`` when the geometry is absent or when the
        architecture makes the closed form unreliable (sliding window,
        shared KV, pooled encoder).
        """
        geom = self.geometry
        if not geom:
            return None
        if geom.get("sliding_window") or geom.get("shared_kv_layers"):
            return None
        block_count = geom.get("block_count")
        interval = geom.get("full_attention_interval", 1) or 1
        n_kv = geom.get("n_kv_heads")
        key_length = geom.get("key_length")
        value_length = geom.get("value_length", key_length)
        if (block_count is None or n_kv is None or key_length is None
                or value_length is None):
            return None
        kv_layers = (block_count // interval if interval > 1 else block_count)
        return kv_layers * n_kv * (key_length + value_length) * 2


@dataclass
class Catalogue:
    """A set of catalogue entries."""

    version: int
    updated: str
    entries: list[CatalogueEntry] = field(default_factory=list)

    def by_id(self, model_id: str) -> CatalogueEntry | None:
        for entry in self.entries:
            if entry.model_id == model_id:
                return entry
        return None

    def sorted_entries(self) -> list[CatalogueEntry]:
        """Measured models first, then by rank_order, then by id."""
        return sorted(
            self.entries,
            key=lambda e: (e.rank_basis != "measured", e.rank_order, e.model_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated": self.updated,
            "models": [_entry_to_dict(e) for e in self.entries],
        }


def _entry_to_dict(entry: CatalogueEntry) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": entry.model_id,
        "repo": entry.repo,
        "params": entry.params,
        "architecture": entry.architecture,
        "license": entry.license,
        "rank_order": entry.rank_order,
        "quantisations": [
            {
                "name": q.name,
                "file": q.file_name,
                "size_bytes": q.file_size_bytes,
                "kv_bytes_per_token": q.kv_bytes_per_token,
                "kv_source": q.kv_source,
            }
            for q in entry.quantisations
        ],
    }
    if entry.extraction is not None:
        e = entry.extraction
        data["extraction"] = {
            "measured": True,
            "recall": e.recall,
            "json_validity": e.json_validity,
            "atomicity": e.atomicity,
            "language_fidelity": e.language_fidelity,
            "language_normalisation": e.language_normalisation,
            "empty_rate": e.empty_rate,
            "corpus": e.corpus,
            "date": e.date,
        }
    if entry.notes:
        data["notes"] = entry.notes
    return data


def entry_from_dict(data: dict[str, Any]) -> CatalogueEntry:
    extraction = None
    raw = data.get("extraction")
    if raw:
        extraction = ExtractionMeasurement(
            recall=float(raw["recall"]),
            json_validity=float(raw["json_validity"]),
            atomicity=float(raw["atomicity"]),
            language_fidelity=float(raw["language_fidelity"]),
            language_normalisation=float(
                raw.get("language_normalisation", 0.0)),
            empty_rate=float(raw["empty_rate"]),
            corpus=str(raw.get("corpus", "")),
            date=str(raw.get("date", "")),
        )
    quantisations = [
        Quantisation(
            name=str(q["name"]),
            file_name=str(q["file"]),
            file_size_bytes=int(q["size_bytes"]),
            kv_bytes_per_token=(int(q["kv_bytes_per_token"])
                                if q.get("kv_bytes_per_token") else None),
            kv_source=str(q.get("kv_source", "formula")),
        )
        for q in data.get("quantisations", [])
    ]
    return CatalogueEntry(
        model_id=str(data["id"]),
        repo=str(data.get("repo", "")),
        params=int(data.get("params", 0)),
        architecture=str(data.get("architecture", "")),
        license=str(data.get("license", "unknown")),
        quantisations=quantisations,
        extraction=extraction,
        rank_order=int(data.get("rank_order", 0)),
        notes=str(data.get("notes", "")),
        geometry=dict(data.get("geometry", {})),
        recurrent_state_mib=float(data.get("recurrent_state_mib", 0.0) or 0.0),
        recurrent_state_source=str(data.get("recurrent_state_source", "none")),
    )


def catalogue_from_dict(data: dict[str, Any]) -> Catalogue:
    return Catalogue(
        version=int(data.get("version", 1)),
        updated=str(data.get("updated", "")),
        entries=[entry_from_dict(e) for e in data.get("models", [])],
    )


def load_catalogue(path: str | Path) -> Catalogue:
    """Load a catalogue JSON file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"catalogue not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return catalogue_from_dict(data)


def default_catalogue_path() -> Path:
    """The shipped catalogue.

    **It lives inside the package.** In the source tree that is
    ``src/cpt_mem0/catalogue.json``, in the packaged app it is
    ``_internal/cpt_mem0/catalogue.json`` - the same place in both
    cases. It used to be in the repository root, which does not exist
    in the package: the app started and then said "catalogue not
    found" in the status line.

    The root is still searched as a fallback, so a checkout that has
    an old layout keeps working.
    """
    package = Path(__file__).resolve().parent / CATALOGUE_FILENAME
    if package.is_file():
        return package

    root = Path(__file__).resolve().parent.parent.parent / CATALOGUE_FILENAME
    return root


def load_default_catalogue() -> Catalogue:
    return load_catalogue(default_catalogue_path())


def merge_overlay(base: Catalogue, overlay: Catalogue) -> Catalogue:
    """Layer a local overlay on top of the shipped catalogue.

    An overlay entry with the same ``model_id`` replaces the base entry.
    The shipped catalogue file is never modified.
    """
    merged = {e.model_id: e for e in base.entries}
    for entry in overlay.entries:
        merged[entry.model_id] = entry
    return Catalogue(
        version=max(base.version, overlay.version),
        updated=overlay.updated or base.updated,
        entries=list(merged.values()),
    )


def apply_extraction_measurement(
    entry: CatalogueEntry, measurement: ExtractionMeasurement
) -> CatalogueEntry:
    """Return a copy of ``entry`` carrying a new extraction measurement."""
    return CatalogueEntry(
        model_id=entry.model_id,
        repo=entry.repo,
        params=entry.params,
        architecture=entry.architecture,
        license=entry.license,
        quantisations=list(entry.quantisations),
        extraction=measurement,
        rank_order=entry.rank_order,
        notes=entry.notes,
    )


def iter_entries(catalogue: Catalogue) -> Iterable[CatalogueEntry]:
    yield from catalogue.entries

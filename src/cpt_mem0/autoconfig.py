"""The configuration the app writes for itself.

**The app has to be able to run without a console.** That means: no
`cpt-mem0 init` with a path typed in, no YAML file edited by hand.
The app finds the models that are on the machine, asks the planner
which of them fits on the card right now, and writes the result.

**The plan is made every time the models are switched on.** The VRAM
belongs to whoever needs it. While a game has the card, the app
takes a smaller model - or none at all. When the card is free again,
the big model comes back. A configuration that was written once would
be wrong an hour later.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import FoundModel, embedder_models, find_models, llm_models


@dataclass
class AutoConfig:
    """What the app worked out for itself."""

    ok: bool = False
    reason: str = ""
    model_id: str = ""
    quantisation: str = ""
    n_ctx: int = 0
    embedder_backend: str = ""
    model_file: Path | None = None
    embedder_file: Path | None = None
    flags: list[str] = dataclasses.field(default_factory=list)
    embedder_flags: list[str] = dataclasses.field(default_factory=list)
    vram_mib: float = 0.0
    free_mib: int | None = None
    used_by_others_mib: int = 0
    candidates: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "model_id": self.model_id,
            "quantisation": self.quantisation,
            "n_ctx": self.n_ctx,
            "embedder_backend": self.embedder_backend,
            "model_file": str(self.model_file) if self.model_file else "",
            "embedder_file": str(self.embedder_file) if self.embedder_file else "",
            "flags": list(self.flags),
            "vram_mib": round(self.vram_mib, 1),
            "free_mib": self.free_mib,
            "used_by_others_mib": self.used_by_others_mib,
            "candidates": list(self.candidates),
        }


#: The flags of the embedder server.
#:
#: The embedder is small and its context is small. It gets its own
#: server, because a store that is written with one embedding and
#: read with another is a store that cannot be searched.
EMBEDDER_FLAGS = [
    "--parallel", "1",
    "-c", "4096",
    # **The embedder runs on the processor.** It is small, it only
    # works when a memory is written or searched, and the card is
    # better used for the model that answers and for the speech
    # recognition. This way the card gets about 400 MiB back.
    "-ngl", "0",
    "--threads", "8",
    # **The batch has to be as big as the text.** mem0 embeds the
    # whole conversation, and a long answer plus a long dictation
    # is more than 512 tokens. llama.cpp refuses anything that does
    # not fit into one physical batch - with the message "input
    # (586 tokens) is too large to process". The processor does not
    # care: the bigger batch costs memory, not VRAM.
    "-b", "4096",
    "-ub", "4096",
    "--embeddings",
    "--pooling", "mean",
    "--kv-unified",
]


def _entries_for(found: list[FoundModel], catalogue: Any) -> list[Any]:
    """The catalogue entries, with only the quantisations that exist.

    **The planner must not plan a file that is not on the disk.** It
    would pick the best quantisation of a model - and then the server
    would start with a path that does not exist.
    """
    entries = []
    for model in llm_models(found):
        if not model.catalogue_id:
            continue
        entry = catalogue.by_id(model.catalogue_id)
        if entry is None:
            continue
        present = {
            Path(model.path).name.lower()
            for model in found
            if model.catalogue_id == model.catalogue_id
        }
        present = {
            Path(item.path).name.lower()
            for item in found
            if item.catalogue_id == model.catalogue_id
        }
        quantisations = [
            quant for quant in entry.quantisations
            if quant.file_name.lower() in present
        ]
        if not quantisations:
            continue
        entries.append(dataclasses.replace(entry, quantisations=quantisations))
    return entries


def _file_for(found: list[FoundModel], model_id: str,
              quantisation: str) -> Path | None:
    """The file of one model in one quantisation."""
    for model in found:
        if (model.catalogue_id == model_id
                and model.quantisation == quantisation):
            return model.path
    return None


def configure(config: Any, catalogue: Any = None,
              hardware: Any = None,
              extra: Path | str | None = None) -> AutoConfig:
    """Work out a configuration that fits the card right now.

    Nothing is written here - the caller decides that. The result
    says what would be used and why.
    """
    from .catalogue import load_default_catalogue
    from .hardware import detect_hardware
    from .planner import PlannerConfig, plan_for_hardware

    if catalogue is None:
        catalogue = load_default_catalogue()
    if hardware is None:
        hardware = detect_hardware()
    if hardware is None:
        return AutoConfig(ok=False,
                          reason="the graphics card could not be read")

    found = find_models(catalogue=catalogue, extra=extra)
    result = AutoConfig()
    result.candidates = [model.to_dict() for model in found]

    llms = llm_models(found)
    if not llms:
        result.reason = (
            "no model was found. Put a GGUF file into "
            "%LOCALAPPDATA%\\cpt-mem0\\models, or download one with "
            "huggingface-cli.")
        return result

    entries = _entries_for(found, catalogue)
    if not entries:
        result.reason = (
            "the models on this machine are not in the catalogue, so "
            "their size on the card is unknown.")
        return result

    # **The plan is made for the memory that is free.** The card has
    # 12 GB, but the program that is running right now uses a part of
    # it - another game, another model, the browser. A model that
    # needs more than what is free does not start: llama.cpp gives
    # up and the app looks broken.
    #
    # The reserve is not taken off again. The free memory already
    # holds what the system needs - taking it off twice would make
    # the plan smaller than it has to be.
    from .supervisor import gpu_free_mib

    free = gpu_free_mib()
    budget = hardware
    planner_config = PlannerConfig(
        embedder_backends=("llamacpp",),
        allow_degraded=config.planner.allow_degraded,
    )
    if free is not None and 0 < free < hardware.total_vram_mib:
        budget = dataclasses.replace(hardware, total_vram_mib=free)
        planner_config = dataclasses.replace(
            planner_config, reserve_mib=0, headroom_mib=0)
        result.free_mib = free
        result.used_by_others_mib = hardware.total_vram_mib - free

    plan = plan_for_hardware(budget, entries, planner_config)
    if not plan.viable or plan.candidate is None:
        result.reason = (
            "no model of the ones on this machine fits on the card: "
            f"{plan.reason}")
        return result

    candidate = plan.candidate
    path = _file_for(found, candidate.model_id, candidate.quantisation)
    if path is None:
        result.reason = (
            f"the file for {candidate.model_id} {candidate.quantisation} "
            "was not found")
        return result

    embedders = embedder_models(found)
    embedder_file = embedders[0].path if embedders else None

    result.ok = True
    result.model_id = candidate.model_id
    result.quantisation = candidate.quantisation
    result.n_ctx = candidate.n_ctx
    result.embedder_backend = candidate.embedder_backend
    result.model_file = path
    result.embedder_file = embedder_file
    result.flags = candidate.as_flags()
    result.embedder_flags = list(EMBEDDER_FLAGS)
    result.vram_mib = candidate.total_vram_mib
    return result


def apply(config: Any, plan: AutoConfig) -> None:
    """Put the plan into the configuration."""
    if not plan.ok:
        return
    config.llm.model = plan.model_id
    config.llm.quantisation = plan.quantisation
    config.llm.context = plan.n_ctx
    config.llm.flags = list(plan.flags)
    config.llm.model_file = plan.model_file
    config.embedder.backend = plan.embedder_backend
    if plan.embedder_file is not None:
        config.embedder.model_file = plan.embedder_file
        config.embedder.flags = list(plan.embedder_flags)

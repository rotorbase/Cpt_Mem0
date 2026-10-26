"""The models that are on the machine.

**The app does not download the models.** That is what huggingface is
for, and a model is a gigabyte. But the app has to know which ones
are there, and it has to be able to say "this one fits on your
card" - otherwise the switch does nothing and the user does not know
why.

The catalogue knows how big every model is. The finder knows what is
on the disk. The planner puts the two together: it takes the models
that are really there and picks the one that fits.

**What is not a model.** The llama.cpp build ships vocabulary files
(`ggml-vocab-*.gguf`) and multimodal projectors (`mmproj-*.gguf`).
They are small and they are not language models - starting a server
with one of them fails after a minute.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Files that are not language models.
NOT_A_MODEL = ("ggml-vocab-", "mmproj-", "mtp-")

#: A model smaller than this is a fragment, not a model.
MIN_SIZE_MIB = 50.0

#: What tells an embedder apart from a language model.
EMBEDDER_MARKERS = ("embed", "bge-", "bge_", "gte-", "gte_", "e5-",
                    "minilm", "minilm")


@dataclass
class FoundModel:
    """A model file on the machine."""

    path: Path
    kind: str = "llm"
    model_id: str = ""
    quantisation: str = ""
    size_mib: float = 0.0
    #: The id in the catalogue, when the file is one of its models.
    catalogue_id: str | None = None

    @property
    def name(self) -> str:
        """What the interface shows."""
        if self.model_id and self.quantisation:
            return f"{self.model_id} {self.quantisation}"
        return self.path.stem

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "model_id": self.model_id,
            "quantisation": self.quantisation,
            "size_mib": round(self.size_mib, 1),
            "catalogue_id": self.catalogue_id,
            "name": self.name,
        }


def model_roots() -> list[Path]:
    """Where the models are looked for.

    The huggingface cache is the first one: that is where
    `huggingface-cli download` puts them, and that is what most
    people use. The own folder of the app is the second one, for the
    models that were put there by hand.
    """
    home = Path.home()
    roots = [
        home / ".cache" / "huggingface" / "hub",
        home / ".cache" / "huggingface" / "embed-models",
        home / ".cache" / "huggingface",
    ]
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    roots.append(local / "cpt-mem0" / "models")
    for extra in (home / "Documents", home / "Downloads"):
        roots.append(extra)
    return [root for root in roots if root.is_dir()]


def _is_embedder(path: Path) -> bool:
    """Whether the file is an embedder."""
    name = path.name.lower()
    if any(marker in name for marker in EMBEDDER_MARKERS):
        return True
    # The folder it lies in: `embed-models` is what the setup uses.
    return "embed" in str(path.parent).lower()


def _catalogue_index(catalogue: Any) -> dict[str, tuple[str, str]]:
    """The files of the catalogue, by file name.

    The name is the exact file name, so the match is not a guess:
    `Qwen3.5-9B-Q8_0.gguf` is in the catalogue as the Q8_0 of
    Qwen3.5-9B, and nowhere else.
    """
    index: dict[str, tuple[str, str]] = {}
    if catalogue is None:
        return index
    for entry in catalogue.sorted_entries():
        for quantisation in entry.sorted_quantisations():
            if quantisation.file_name:
                index[quantisation.file_name.lower()] = (
                    entry.model_id, quantisation.name)
    return index


def find_models(roots: list[Path] | None = None,
                catalogue: Any = None) -> list[FoundModel]:
    """Every model file on the machine.

    The catalogue is only used to name the models. A file that is not
    in the catalogue is still found - it is just not in the plan.
    """
    if roots is None:
        roots = model_roots()
    index = _catalogue_index(catalogue)

    found: list[FoundModel] = []
    seen: set[str] = set()

    for root in roots:
        if not root.is_dir():
            continue
        try:
            paths = list(root.rglob("*.gguf"))
        except OSError:
            continue
        for path in paths:
            key = str(path).lower()
            if key in seen:
                continue
            name = path.name.lower()
            if name.startswith(NOT_A_MODEL):
                continue
            try:
                size_mib = path.stat().st_size / 1048576
            except OSError:
                continue
            if size_mib < MIN_SIZE_MIB:
                continue
            seen.add(key)

            embedder = _is_embedder(path)
            model_id = ""
            quantisation = ""
            catalogue_id: str | None = None
            if not embedder and name in index:
                catalogue_id, quantisation = index[name]
                model_id = catalogue_id
            elif embedder:
                model_id = path.stem

            found.append(FoundModel(
                path=path,
                kind="embedder" if embedder else "llm",
                model_id=model_id,
                quantisation=quantisation,
                size_mib=size_mib,
                catalogue_id=catalogue_id,
            ))

    found.sort(key=lambda item: item.size_mib, reverse=True)
    return found


def llm_models(models: list[FoundModel]) -> list[FoundModel]:
    """The language models."""
    return [model for model in models if model.kind == "llm"]


def embedder_models(models: list[FoundModel]) -> list[FoundModel]:
    """The embedders."""
    return [model for model in models if model.kind == "embedder"]

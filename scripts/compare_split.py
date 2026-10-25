"""Compare one model against a split (chat + extraction) setup.

The split design costs a second llama-server: its own CUDA context,
its own compute buffer, its own KV pool and - for a hybrid model - its
own recurrent state. This script computes both variants with the
shipped planner, so the numbers in the specification come from the
code.

The extraction model gets a fixed slot of ``CONTEXT_FLOOR_SLOT``
tokens - the fact-extraction prompt needs about 10 500 of them. The
chat model gets whatever VRAM is left.

Usage:
    python scripts/compare_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.catalogue import default_catalogue_path, load_catalogue  # noqa: E402
from cpt_mem0.planner import (  # noqa: E402
    COMPUTE_MIB,
    COMPUTE_MIB_SMALL,
    CONTEXT_FLOOR_SLOT,
    CUDA_CTX_MIB,
    HOST_FRACTION,
    KV_BLOCK_TOKENS,
    Hardware,
    PlannerConfig,
    kv_bytes_for_cache,
    plan_for_hardware,
)

#: The cards the specification compares.
CARDS = [
    ("12 GB", 12288),
    ("8 GB", 8192),
    ("6 GB", 6144),
    ("5 GB", 5120),
    ("4 GB", 4096),
]

#: The chat model and the extraction model of the split design.
CHAT_MODEL = ("Qwen3.5-9B", "Q4_K_M")
EXTRACTION_MODEL = ("Qwen3-4B-Instruct-2507", "Q4_K_M")

CACHE_TYPE = "q8_0"


def _entry(catalogue, model_id: str):
    for entry in catalogue.sorted_entries():
        if entry.model_id == model_id:
            return entry
    return None


def _fixed_cost(entry, quant, cache_type: str) -> tuple[float, float]:
    """The fixed VRAM of one model plus its KV bytes per token."""
    weights = quant.file_size_mib * (1.0 - HOST_FRACTION)
    compute = (COMPUTE_MIB if quant.file_size_mib > 2000
               else COMPUTE_MIB_SMALL)
    fixed = (weights + compute + CUDA_CTX_MIB
             + entry.recurrent_state_mib)
    bytes_per_token = kv_bytes_for_cache(quant.kv_bytes_per_token, cache_type)
    return fixed, bytes_per_token


def main() -> int:
    catalogue = load_catalogue(str(default_catalogue_path()))
    config = PlannerConfig(embedder_backends=("llamacpp",))

    chat_entry = _entry(catalogue, CHAT_MODEL[0])
    chat_quant = chat_entry.quantisation(CHAT_MODEL[1]) if chat_entry else None
    extraction_entry = _entry(catalogue, EXTRACTION_MODEL[0])
    extraction_quant = (
        extraction_entry.quantisation(EXTRACTION_MODEL[1])
        if extraction_entry else None)
    if chat_entry is None or chat_quant is None:
        print("the chat model is not in the catalogue")
        return 1
    if extraction_entry is None or extraction_quant is None:
        print("the extraction model is not in the catalogue")
        return 1

    chat_fixed, chat_bpt = _fixed_cost(chat_entry, chat_quant, CACHE_TYPE)
    extraction_fixed, extraction_bpt = _fixed_cost(
        extraction_entry, extraction_quant, CACHE_TYPE)

    print("### One model against a split (chat + extraction)")
    print()
    print(f"Extraction model: {EXTRACTION_MODEL[0]} {EXTRACTION_MODEL[1]}, "
          f"fixed slot of {CONTEXT_FLOOR_SLOT} tokens")
    print(f"Chat model:       {CHAT_MODEL[0]} {CHAT_MODEL[1]}, "
          f"KV {CACHE_TYPE}")
    print()
    print("| Card | 9B chat + 4B extraction | 9B chat alone |")
    print("|---|---|---|")

    for label, total in CARDS:
        hardware = Hardware(total_vram_mib=total)

        # One model: the planner decides.
        alone = plan_for_hardware(hardware, [chat_entry], config)
        alone_text = "does not fit"
        if alone is not None and alone.viable and alone.candidate:
            alone_text = (f"{alone.candidate.n_ctx:,} "
                          f"({alone.candidate.n_ctx_slot:,}/slot)"
                          ).replace(",", " ")

        # Split: both models resident, the extraction model with a fixed
        # slot, the chat model with what is left.
        reserve = config.reserve_for(total)
        headroom = config.headroom_for(total)
        extraction_kv = CONTEXT_FLOOR_SLOT * extraction_bpt / (1024 * 1024)
        available = (total - reserve - headroom
                     - chat_fixed - extraction_fixed - extraction_kv)
        chat_tokens = int(available / (chat_bpt / (1024 * 1024)))
        chat_tokens = (chat_tokens // KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS
        if chat_tokens < CONTEXT_FLOOR_SLOT:
            split_text = "does not fit"
        else:
            split_text = (f"{chat_tokens:,} ({chat_tokens:,}/slot)"
                          ).replace(",", " ")

        print(f"| {label} | {split_text} | {alone_text} |")

    print()
    print("The split costs a second llama-server: its own CUDA context,")
    print("its own compute buffer, its own KV pool and - for a hybrid")
    print("model - its own recurrent state.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

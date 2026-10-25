"""Reproduce the card tiers from docs/verification.md section 9.2.

This is the acceptance test for the planner: the numbers in the
documentation must come out of the code, not out of a scratch script
that nobody maintains.

Two different questions are answered, because the original table mixed
them:

* **cascade** - what the planner actually returns for this card, per
  SPEC section 4.4: best model first, then best quantisation, then best
  KV cache type.
* **max context** - the largest context any configuration on this card
  can reach, which is what the original table listed.

Run with::

    python scripts/regenerate_tiers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.catalogue import default_catalogue_path, load_catalogue  # noqa: E402
from cpt_mem0.planner import (  # noqa: E402
    Hardware,
    PlannerConfig,
    embedder_cost_mib,
    evaluate_candidate,
    plan_for_hardware,
)

TIERS = [
    ("12 GB", 12288),
    ("8 GB", 8192),
    ("6 GB", 6144),
    ("5 GB", 5120),
    ("4 GB", 4096),
]

# The GPU embedder is the default; the CPU embedder is the fallback.
GPU_ONLY = PlannerConfig(embedder_backends=("llamacpp",))


def _max_context(catalogue, hardware, config):
    """The configuration with the largest reachable context."""
    best = None
    for entry in catalogue.entries:
        for quant in entry.quantisations:
            if quant.kv_bytes_per_token is None:
                continue
            for cache_type in config.cache_type_order:
                for embedder in config.embedder_backends:
                    candidate = evaluate_candidate(
                        model_id=entry.model_id,
                        quantisation=quant.name,
                        file_name=quant.file_name,
                        file_size_mib=quant.file_size_mib,
                        kv_bytes_per_token=quant.kv_bytes_per_token,
                        cache_type=cache_type,
                        hardware=hardware,
                        config=config,
                        embedder_backend=embedder,
                    )
                    if candidate is None:
                        continue
                    if best is None or candidate.n_ctx > best.n_ctx:
                        best = candidate
    return best


def main() -> int:
    catalogue = load_catalogue(default_catalogue_path())

    print("### Cascade result (SPEC section 4.4 order)")
    print()
    print("| VRAM | Configuration | `n_ctx` | Per slot | Embedder |")
    print("|---|---|---|---|---|")
    for label, total_mib in TIERS:
        hardware = Hardware(total_vram_mib=total_mib)
        result = plan_for_hardware(hardware, catalogue.sorted_entries(), GPU_ONLY)
        if result.viable and result.candidate:
            c = result.candidate
            print(f"| {label} | {c.model_id}-{c.quantisation} "
                  f"| {c.n_ctx:,} | {c.n_ctx_slot:,} | {c.embedder_backend} |"
                  .replace(",", " "))
        else:
            print(f"| {label} | **no viable configuration** | - | - | - |")

    print()
    print("### Largest reachable context per card")
    print()
    print("| VRAM | Configuration | `n_ctx` | Per slot |")
    print("|---|---|---|---|")
    for label, total_mib in TIERS:
        hardware = Hardware(total_vram_mib=total_mib)
        best = _max_context(catalogue, hardware, GPU_ONLY)
        if best is None:
            print(f"| {label} | **nothing fits** | - | - |")
        else:
            print(f"| {label} | {best.model_id}-{best.quantisation} "
                  f"| {best.n_ctx:,} | {best.n_ctx_slot:,} |"
                  .replace(",", " "))

    print()
    print("### 4 GB minimum configuration breakdown")
    print()
    hardware = Hardware(total_vram_mib=4096)
    best = _max_context(catalogue, hardware, GPU_ONLY)
    if best is not None:
        reserve = GPU_ONLY.reserve_for(4096)
        headroom = GPU_ONLY.headroom_for(4096)
        floor_kv = (GPU_ONLY.context_floor_slot * best.kv_bytes_per_token
                    / (1024 * 1024))
        overhead = (best.compute_mib + best.cuda_context_mib
                    + embedder_cost_mib(best.embedder_backend))
        print("| Item | MiB |")
        print("|---|---|")
        print(f"| {best.model_id} {best.quantisation} weights | "
              f"{best.weights_vram_mib:.0f} |")
        print(f"| KV cache, {GPU_ONLY.context_floor_slot} tokens | {floor_kv:.0f} |")
        print(f"| Process overhead | {overhead:.0f} |")
        print(f"| Reserve (10 %) + headroom (6 %) | {reserve + headroom} |")
        total = (best.weights_vram_mib + floor_kv + overhead
                 + reserve + headroom)
        print(f"| **Total required** | **{total:.0f}** |")
        print("| **Available** | **4096** |")
        print(f"| **Shortfall** | **{max(0, total - 4096):.0f}** |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

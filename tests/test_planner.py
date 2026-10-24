"""Planner tests.

Every expected number here is traceable to docs/verification.md. The
citations in the comments are the contract: if a constant changes, the
measurement is re-run and this file changes with it.
"""

from __future__ import annotations

import pytest

from cpt_mem0.catalogue import CatalogueEntry, ExtractionMeasurement, Quantisation
from cpt_mem0.planner import (
    CONTEXT_FLOOR_SLOT,
    CONTEXT_FLOOR_TOTAL,
    CUDA_CTX_MIB,
    HOST_FRACTION,
    Hardware,
    PlannerConfig,
    embedder_cost_mib,
    evaluate_candidate,
    kv_bytes_for_cache,
    plan_for_hardware,
)

# Measured constants (docs/verification.md section 1.2):
QWEN35_9B_KV_BYTES_F16 = 32768
# Nominal sizes (docs/verification.md section 9.4):
QWEN35_9B_Q4KM_MIB = 5417
QWEN35_9B_Q8_MIB = 9086
# The 12 GB reference card (docs/verification.md section 9.2):
RTX_3060_12GB = 12288


def _entry(model_id: str, size_mib: float, kv: int, rank: int = 10,
           quant: str = "Q4_K_M") -> CatalogueEntry:
    return CatalogueEntry(
        model_id=model_id,
        repo="local",
        params=9_000_000_000,
        architecture="qwen35",
        license="apache-2.0",
        quantisations=[
            Quantisation(name=quant, file_name=f"{model_id}-{quant}.gguf",
                         file_size_bytes=int(size_mib * 1024 * 1024),
                         kv_bytes_per_token=kv, kv_source="measured"),
        ],
        rank_order=rank,
    )


class TestKvCacheScaling:
    """The cache-type factors from SPEC section 4.5.

    The quantised factors are llama.cpp's block layouts, read from
    ``ggml/src/ggml-common.h``:

    ==============  ==========================  ==========
    Type            Block                       Bytes/value
    ==============  ==========================  ==========
    ``f16``         2 B half                    2.0
    ``q8_0``        2 B half scale + 32 B int8  1.0625
    ``q4_0``        2 B half scale + 16 B nib   0.5625
    ==============  ==========================  ==========

    An earlier estimate used 1.125 bytes per value for ``q8_0``, which is
    5.9 % too pessimistic and made every context figure too small.
    """

    def test_f16_is_the_reference(self):
        assert kv_bytes_for_cache(QWEN35_9B_KV_BYTES_F16, "f16") == 32768

    def test_q8_0_matches_the_ggml_block_layout(self):
        # 34 bytes per 32 values
        scaled = kv_bytes_for_cache(QWEN35_9B_KV_BYTES_F16, "q8_0")
        assert scaled == int(QWEN35_9B_KV_BYTES_F16 * 1.0625 / 2.0)
        assert scaled == 17408

    def test_q4_0_matches_the_ggml_block_layout(self):
        # 18 bytes per 32 values
        scaled = kv_bytes_for_cache(QWEN35_9B_KV_BYTES_F16, "q4_0")
        assert scaled == int(QWEN35_9B_KV_BYTES_F16 * 0.5625 / 2.0)
        assert scaled == 9216

    def test_the_documented_sizes_hold(self):
        """SPEC section 4.5: 16384 tokens of the 9B."""
        per_token_f16 = QWEN35_9B_KV_BYTES_F16
        mib = lambda b: b * 16384 / (1024 * 1024)
        assert round(mib(per_token_f16)) == 512
        assert round(mib(kv_bytes_for_cache(per_token_f16, "q8_0"))) == 272
        assert round(mib(kv_bytes_for_cache(per_token_f16, "q4_0"))) == 144

    def test_q8_0_is_roughly_half_of_f16(self):
        scaled = kv_bytes_for_cache(QWEN35_9B_KV_BYTES_F16, "q8_0")
        assert 0.50 < scaled / QWEN35_9B_KV_BYTES_F16 < 0.55

    def test_unknown_cache_type_is_rejected(self):
        with pytest.raises(ValueError):
            kv_bytes_for_cache(32768, "q3_k")


class TestEmbedderCost:
    def test_llamacpp_embedder_needs_a_second_cuda_context(self):
        # weights + KV + compute + its own CUDA context
        assert embedder_cost_mib("llamacpp") == 78 + 12 + 130 + CUDA_CTX_MIB

    def test_fastembed_needs_no_vram(self):
        assert embedder_cost_mib("fastembed") == 0

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError):
            embedder_cost_mib("nope")


class TestCandidateArithmetic:
    def test_weights_use_the_host_fraction(self):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        candidate = evaluate_candidate(
            model_id="Qwen3.5-9B",
            quantisation="Q4_K_M",
            file_name="m.gguf",
            file_size_mib=QWEN35_9B_Q4KM_MIB,
            kv_bytes_per_token=QWEN35_9B_KV_BYTES_F16,
            cache_type="q8_0",
            hardware=hardware,
            config=PlannerConfig(),
        )
        assert candidate is not None
        expected = QWEN35_9B_Q4KM_MIB * (1.0 - HOST_FRACTION)
        assert candidate.weights_vram_mib == pytest.approx(expected, rel=1e-6)

    def test_budget_stays_inside_the_card(self):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        candidate = evaluate_candidate(
            model_id="Qwen3.5-9B",
            quantisation="Q4_K_M",
            file_name="m.gguf",
            file_size_mib=QWEN35_9B_Q4KM_MIB,
            kv_bytes_per_token=QWEN35_9B_KV_BYTES_F16,
            cache_type="q8_0",
            hardware=hardware,
            config=PlannerConfig(),
        )
        assert candidate is not None
        total = (candidate.total_vram_mib
                 + PlannerConfig().reserve_for(RTX_3060_12GB)
                 + PlannerConfig().headroom_for(RTX_3060_12GB))
        assert total <= RTX_3060_12GB

    def test_n_ctx_is_divisible_by_parallel(self):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        candidate = evaluate_candidate(
            model_id="Qwen3.5-9B",
            quantisation="Q4_K_M",
            file_name="m.gguf",
            file_size_mib=QWEN35_9B_Q4KM_MIB,
            kv_bytes_per_token=QWEN35_9B_KV_BYTES_F16,
            cache_type="q8_0",
            hardware=hardware,
            config=PlannerConfig(parallel=2),
        )
        assert candidate is not None
        assert candidate.n_ctx % 2 == 0
        assert candidate.n_ctx_slot == candidate.n_ctx // 2

    def test_a_model_that_does_not_fit_returns_none(self):
        hardware = Hardware(total_vram_mib=4096)
        candidate = evaluate_candidate(
            model_id="Qwen3.5-9B",
            quantisation="Q8_0",
            file_name="m.gguf",
            file_size_mib=QWEN35_9B_Q8_MIB,
            kv_bytes_per_token=QWEN35_9B_KV_BYTES_F16,
            cache_type="f16",
            hardware=hardware,
            config=PlannerConfig(),
        )
        assert candidate is None

    def test_flags_always_disable_fit(self):
        """The planner computed the configuration; llama.cpp must not
        renegotiate it at runtime. This is the bug the project exists to
        prevent."""
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        candidate = evaluate_candidate(
            model_id="Qwen3.5-9B",
            quantisation="Q4_K_M",
            file_name="m.gguf",
            file_size_mib=QWEN35_9B_Q4KM_MIB,
            kv_bytes_per_token=QWEN35_9B_KV_BYTES_F16,
            cache_type="q8_0",
            hardware=hardware,
            config=PlannerConfig(),
        )
        assert candidate is not None
        flags = candidate.as_flags()
        assert "--fit" in flags
        assert flags[flags.index("--fit") + 1] == "off"
        assert "-c" in flags


class TestContextFloor:
    """SPEC section 9.2: a service that cannot extract facts is not a
    service."""

    def test_floor_constants(self):
        assert CONTEXT_FLOOR_SLOT == 12288
        assert CONTEXT_FLOOR_TOTAL == 21000

    def test_a_4gb_card_is_rejected(self):
        """docs/verification.md section 9.2: the 4 GB tier has no viable
        configuration with the GPU embedder.

        The KV constant used here is Qwen3-4B's (147456 B/token, dense
        architecture), not the hybrid 9B's.
        """
        hardware = Hardware(total_vram_mib=4096)
        entries = [_entry("Qwen3-4B-Instruct", 2120, 147456, rank=10,
                          quant="Q4_K_M")]
        result = plan_for_hardware(
            hardware, entries,
            PlannerConfig(embedder_backends=("llamacpp",)))
        assert result.viable is False
        assert result.diagnostics["reason"] == "context_floor_not_met"

    def test_a_12gb_card_is_accepted(self):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        entries = [_entry("Qwen3.5-9B", QWEN35_9B_Q4KM_MIB,
                          QWEN35_9B_KV_BYTES_F16, rank=10)]
        result = plan_for_hardware(hardware, entries, PlannerConfig())
        assert result.viable is True
        assert result.candidate is not None
        assert result.candidate.n_ctx >= CONTEXT_FLOOR_TOTAL
        assert result.candidate.n_ctx_slot >= CONTEXT_FLOOR_SLOT

    def test_degraded_mode_is_opt_in(self):
        hardware = Hardware(total_vram_mib=4096)
        entries = [_entry("Qwen3-4B-Instruct", 2120, 147456, rank=10)]
        config = PlannerConfig(embedder_backends=("llamacpp",))
        strict = plan_for_hardware(hardware, entries, config)
        assert strict.viable is False

        degraded = plan_for_hardware(
            hardware, entries,
            PlannerConfig(embedder_backends=("llamacpp",), allow_degraded=True))
        assert degraded.viable is True
        assert degraded.diagnostics["degraded"] is True
        assert degraded.diagnostics["warning"]

    def test_rejection_diagnostics_name_the_shortfall(self):
        hardware = Hardware(total_vram_mib=4096)
        entries = [_entry("Qwen3-4B-Instruct", 2120, 147456, rank=10)]
        result = plan_for_hardware(
            hardware, entries, PlannerConfig(embedder_backends=("llamacpp",)))
        assert result.diagnostics["available_mib"] == 4096
        assert result.diagnostics["shortfall_mib"] > 0
        assert result.diagnostics["options"]


class TestCascadeOrder:
    def test_quantisation_prefers_the_better_one_when_both_fit(self):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        entry = CatalogueEntry(
            model_id="Qwen3.5-9B", repo="local", params=9_000_000_000,
            architecture="qwen35", license="apache-2.0",
            quantisations=[
                Quantisation("Q4_K_M", "a.gguf", int(5417 * 1024 * 1024),
                             QWEN35_9B_KV_BYTES_F16),
                Quantisation("Q8_0", "b.gguf", int(9086 * 1024 * 1024),
                             QWEN35_9B_KV_BYTES_F16),
            ],
            rank_order=10,
        )
        result = plan_for_hardware(hardware, [entry], PlannerConfig())
        assert result.viable is True
        # Q8_0 is listed first in the catalogue's preference order.
        assert result.candidate.quantisation == "Q8_0"

    def test_measured_models_rank_above_unmeasured(self):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        unmeasured = _entry("Unmeasured-9B", 5417, QWEN35_9B_KV_BYTES_F16,
                            rank=1)
        measured = _entry("Measured-9B", 5417, QWEN35_9B_KV_BYTES_F16,
                          rank=99)
        measured.extraction = ExtractionMeasurement(
            recall=0.88, json_validity=1.0, atomicity=1.0,
            language_fidelity=0.9, empty_rate=0.0,
            corpus="test", date="2026-10-24")
        result = plan_for_hardware(hardware, [unmeasured, measured],
                                   PlannerConfig())
        assert result.viable is True
        assert result.candidate is not None
        assert result.candidate.model_id == "Measured-9B"
        assert result.candidate.model_rank_basis == "measured"

    def test_unmeasured_candidate_is_labelled(self):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        entries = [_entry("Unmeasured-9B", 5417, QWEN35_9B_KV_BYTES_F16)]
        result = plan_for_hardware(hardware, entries, PlannerConfig())
        assert result.viable is True
        assert result.candidate.model_rank_basis == "unmeasured"

    def test_calibration_entries_are_reported_not_guessed(self):
        """A model without a KV constant must not be silently planned."""
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        entry = CatalogueEntry(
            model_id="gemma-4-E2B-it", repo="local", params=2_000_000_000,
            architecture="gemma4", license="gemma",
            quantisations=[
                Quantisation("Q4_K_M", "g.gguf", int(3107 * 1024 * 1024),
                             None, "calibrate"),
            ],
            rank_order=5,
        )
        result = plan_for_hardware(hardware, [entry], PlannerConfig())
        assert result.viable is False
        assert "gemma-4-E2B-it" in result.diagnostics["calibration_required"]


class TestCatalogueIntegrity:
    def test_every_model_has_a_quantisation(self, catalogue):
        for entry in catalogue.entries:
            assert entry.quantisations, f"{entry.model_id} has no quantisation"

    def test_calibrate_entries_have_no_constant(self, catalogue):
        for entry in catalogue.entries:
            for quant in entry.quantisations:
                if quant.kv_source == "calibrate":
                    assert quant.kv_bytes_per_token is None
                else:
                    assert quant.kv_bytes_per_token

    def test_no_duplicate_model_ids(self, catalogue):
        ids = [e.model_id for e in catalogue.entries]
        assert len(ids) == len(set(ids))

    def test_qwen35_kv_constant_matches_the_measurement(self, catalogue):
        """docs/verification.md section 1.2: 32768 bytes per token."""
        entry = catalogue.by_id("Qwen3.5-9B")
        assert entry is not None
        for quant in entry.quantisations:
            assert quant.kv_bytes_per_token == 32768
            assert quant.kv_source == "measured"


class TestCatalogueGeometry:
    """The KV constants must follow from the measured GGUF geometry.

    Every value here was read from a real GGUF header, either from a
    local file or through an HTTP range request against the HuggingFace
    repository. The formula is:

        kv_layers   = block_count // full_attention_interval
        bytes/token = kv_layers * n_kv_heads * (key_length + value_length) * 2
    """

    # model id -> (block_count, interval, n_kv_heads, key_len, value_len)
    GEOMETRY = {
        "Qwen3.5-9B":             (32, 4, 4, 256, 256),
        "Qwen3.5-4B":             (32, 4, 4, 256, 256),
        "Qwen3-8B":               (36, 1, 8, 128, 128),
        "Qwen3-4B-Instruct-2507": (36, 1, 8, 128, 128),
        "Llama-3.1-8B-Instruct":  (32, 1, 8, 128, 128),
    }

    @staticmethod
    def _expected(block_count, interval, n_kv, klen, vlen) -> int:
        kv_layers = block_count // interval if interval > 1 else block_count
        return kv_layers * n_kv * (klen + vlen) * 2

    def test_every_constant_follows_from_the_geometry(self, catalogue):
        for model_id, geom in self.GEOMETRY.items():
            entry = catalogue.by_id(model_id)
            assert entry is not None, model_id
            expected = self._expected(*geom)
            for quant in entry.quantisations:
                assert quant.kv_bytes_per_token == expected, (
                    f"{model_id}/{quant.name}: {quant.kv_bytes_per_token} "
                    f"!= {expected}")

    def test_hybrid_models_use_the_interval(self, catalogue):
        """The bug this project exists to prevent: using all layers
        overestimates the cache fourfold for Qwen3.5."""
        entry = catalogue.by_id("Qwen3.5-9B")
        assert entry is not None
        naive = 32 * 4 * (256 + 256) * 2
        assert entry.quantisations[0].kv_bytes_per_token == naive // 4

    def test_gemma_is_marked_unreliable(self, catalogue):
        """gemma-4-E2B has sliding-window and shared-KV layers; the
        closed form does not apply."""
        entry = catalogue.by_id("gemma-4-E2B-it")
        assert entry is not None
        for quant in entry.quantisations:
            assert quant.kv_source == "calibrate"
            assert quant.kv_bytes_per_token is None

    def test_the_stored_geometry_agrees_with_the_constant(self, catalogue):
        """The catalogue carries the geometry it was built from. The
        constant must follow from it, or the two have drifted apart."""
        for entry in catalogue.entries:
            derived = entry.kv_bytes_from_geometry()
            if derived is None:
                # Unreliable architecture - the constant must be absent.
                assert entry.quantisations[0].kv_bytes_per_token is None
                continue
            for quant in entry.quantisations:
                assert quant.kv_bytes_per_token == derived, (
                    f"{entry.model_id}/{quant.name}: catalogue says "
                    f"{quant.kv_bytes_per_token}, geometry implies {derived}")


class TestCatalogueSizes:
    """The catalogue sizes must match docs/verification.md section 11.

    Those numbers came from the Hugging Face API (`?blobs=true`) on
    2026-10-24. A mismatch means the catalogue drifted from the
    documentation, or the documentation is stale.
    """

    # model id -> {quantisation: MiB}
    SIZES = {
        "Qwen3.5-9B":             {"Q8_0": 9086, "Q6_K": 7113, "Q5_K_M": 6273,
                                  "Q4_K_M": 5417, "IQ4_XS": 4929, "IQ3_XXS": 3830},
        "Qwen3-8B":               {"Q8_0": 8306, "Q6_K": 6414, "Q5_K_M": 5580,
                                  "Q4_K_M": 4795, "IQ4_XS": 4369, "IQ3_XXS": 3252},
        "Llama-3.1-8B-Instruct":  {"Q6_K": 6290, "Q5_K_M": 5467, "Q4_K_M": 4693,
                                  "IQ4_XS": 4257, "IQ3_XXS": 3168},
        "gemma-4-E2B-it":         {"Q8_0": 4814, "Q6_K": 4293, "Q5_K_M": 3201,
                                  "Q4_K_M": 2963, "IQ4_XS": 2846, "IQ3_XXS": 2263},
        "Qwen3.5-4B":             {"Q8_0": 4275, "Q6_K": 3363, "Q5_K_M": 2998,
                                  "Q4_K_M": 2614, "IQ4_XS": 2362, "IQ3_XXS": 1859},
        "Qwen3-4B-Instruct-2507": {"Q8_0": 4082, "Q6_K": 3153, "Q5_K_M": 2756,
                                  "Q4_K_M": 2382, "IQ4_XS": 2166, "IQ3_XXS": 1597},
    }

    def test_sizes_match_the_documentation(self, catalogue):
        for model_id, expected in self.SIZES.items():
            entry = catalogue.by_id(model_id)
            assert entry is not None, model_id
            for quant in entry.quantisations:
                if quant.name not in expected:
                    continue
                actual = round(quant.file_size_mib)
                assert actual == expected[quant.name], (
                    f"{model_id}/{quant.name}: {actual} MiB != "
                    f"{expected[quant.name]} MiB")

    def test_no_draft_or_projection_models(self, catalogue):
        """The gemma repository contains an MTP draft model and an
        mmproj; resolving those would silently pick the wrong file."""
        for entry in catalogue.entries:
            for quant in entry.quantisations:
                assert "MTP" not in quant.file_name
                assert "mmproj" not in quant.file_name


class TestCardHierarchy:
    """A larger card must get a better model, not just more context."""

    def test_the_9b_is_selected_on_a_12gb_card(self, catalogue):
        hardware = Hardware(total_vram_mib=RTX_3060_12GB)
        result = plan_for_hardware(hardware, catalogue.sorted_entries(),
                                   PlannerConfig())
        assert result.viable is True
        assert result.candidate is not None
        assert "9B" in result.candidate.model_id

    def test_a_smaller_card_falls_back_to_a_smaller_model(self, catalogue):
        hardware = Hardware(total_vram_mib=4096)
        result = plan_for_hardware(hardware, catalogue.sorted_entries(),
                                   PlannerConfig())
        assert result.viable is True
        assert result.candidate is not None
        assert "9B" not in result.candidate.model_id

    def test_the_hierarchy_is_monotonic(self, catalogue):
        """More VRAM never yields a smaller model."""
        previous = None
        for total in (4096, 5120, 6144, 8192, 12288):
            hardware = Hardware(total_vram_mib=total)
            result = plan_for_hardware(hardware, catalogue.sorted_entries(),
                                       PlannerConfig())
            assert result.viable is True
            assert result.candidate is not None
            params = next(e.params for e in catalogue.entries
                          if e.model_id == result.candidate.model_id)
            if previous is not None:
                assert params >= previous, (
                    f"{total} MiB picked a smaller model than a smaller card")
            previous = params

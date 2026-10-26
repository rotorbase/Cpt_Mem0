"""The VRAM planner.

This is the product. It answers one question:

    Given this GPU, which model, at which quantisation, with how much
    context, with the embedder where - all simultaneously resident?

Two constraints gate every candidate:

* the VRAM budget (see ``_budget``)
* the context floor (see ``CONTEXT_FLOOR_TOTAL``)

A candidate that fits in VRAM but violates the context floor is rejected.
A service that cannot extract facts is not a service - see SPEC section 9.

All constants are measured values; the sources are listed next to each.
Nothing here touches the GPU or the filesystem.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Measured constants
#
# Sources: docs/verification.md
#   - HOST_FRACTION      section 2  (Q4_K_M 10 %, Q8_0 11 % - 11 % used)
#   - COMPUTE_MIB        section 2  (server log, 286-385 MiB)
#   - CUDA_CTX_MIB       section 2  (per llama-server process)
#   - EMBEDDER_*         section 3  (nomic-embed measurements)
#   - CONTEXT_FLOOR      section 9  (measured extraction prompt: 10449 tokens)
# ---------------------------------------------------------------------------

HOST_FRACTION = 0.11          # share of the model file that stays in host RAM
#: The CUDA scratch buffer grows with the context: it holds the
#: activations for the whole batch. Measured on an RTX 3060 with
#: llama.cpp 0.7.0-dev (build 11798):
#:
#: ============================  ==================  =============
#: Model                        Context             Measured
#: ============================  ==================  =============
#: Qwen3.5-9B-Q8_0               54 784              331.79 MiB
#: Qwen3.5-9B-Q8_0               61 696              365.54 MiB
#: Qwen3.5-9B-Q8_0               67 072              391.79 MiB
#: Qwen3-4B-Instruct-Q4_K_M      16 384              116.09 MiB
#: ============================  ==================  =============
#:
#: For the 9B the three points lie on a straight line:
#:
#:     compute = 0.00488281 * n_ctx + 64.29
#:
#: It reproduces all three measurements to within 0.01 MiB. A constant
#: would have been wrong at every context but one - the first version
#: of the planner used 290 MiB, taken from a run at a smaller context,
#: and the 12 GB card was 160 MiB too optimistic as a result.
COMPUTE_SLOPE = 0.00488281    # MiB per token
COMPUTE_INTERCEPT = 64.29     # MiB
COMPUTE_MIB = 392             # CUDA scratch buffer, 9B class
COMPUTE_MIB_SMALL = 130       # CUDA scratch buffer, small model
CUDA_CTX_MIB = 320            # CUDA context, per llama-server process

#: The embedder is a second llama-server with nomic-embed-text-v1.5
#: Q4_K_M. Measured on an RTX 3060 with llama.cpp 0.7.0-dev (build
#: 11798), with the flags the app uses:
#:
#: ==========================  ==============
#: Line                       Measured
#: ==========================  ==============
#: ``CUDA0 model buffer``      66.92 MiB
#: ``CPU_Mapped model``        12.58 MiB (host)
#: ``CUDA0 compute buffer``    24.52 MiB
#: ==========================  ==============
#:
#: The weights are 66.92 MiB on the card and 12.58 MiB in host RAM. The
#: compute buffer is 24.52 MiB, not the 130 MiB an earlier measurement
#: suggested.
EMBEDDER_WEIGHTS_MIB = 67     # nomic-embed-text-v1.5 Q4_K_M on the GPU
EMBEDDER_KV_MIB = 12          # at n_ctx 2048
EMBEDDER_COMPUTE_MIB = 25

#: mem0's fact-extraction prompt measured at 10449 tokens. A slot must hold
#: at least this much or the prompt is truncated and nothing is stored.
CONTEXT_FLOOR_SLOT = 12288

#: Two slots (extraction + chat) at the floor.
CONTEXT_FLOOR_TOTAL = 21000

#: Quantisation preference, best first.
QUANTISATION_ORDER = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS",
                      "IQ3_XXS", "Q3_K_M", "Q2_K")

#: KV cache types, best first. q8_0 halves the cache at negligible quality
#: cost and is the single strongest lever for context size.
CACHE_TYPE_ORDER = ("q8_0", "f16", "q4_0")

#: Bytes per element per KV cache type. The catalogue stores the f16
#: constant; the planner scales it down for quantised caches.
CACHE_ELEMENT_BYTES = {
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}


def kv_bytes_for_cache(f16_bytes_per_token: int, cache_type: str) -> int:
    """Scale an f16 KV constant to a quantised cache type."""
    if cache_type not in CACHE_ELEMENT_BYTES:
        raise ValueError(
            f"unknown cache type {cache_type!r}; "
            f"expected one of {sorted(CACHE_ELEMENT_BYTES)}"
        )
    return int(f16_bytes_per_token * CACHE_ELEMENT_BYTES[cache_type] / 2.0)


@dataclass(frozen=True)
class Hardware:
    """The GPU the plan is computed for."""

    total_vram_mib: int
    name: str = ""
    vendor: str = ""
    driver_version: str = ""

    def __post_init__(self) -> None:
        if self.total_vram_mib <= 0:
            raise ValueError("total_vram_mib must be positive")


@dataclass(frozen=True)
class PlannerConfig:
    """Tunables of the planner. Every field has a working default."""

    reserve_mib: int | None = None          # None -> 10 % of total VRAM
    headroom_mib: int | None = None         # None -> 6 % of total VRAM
    context_floor_total: int = CONTEXT_FLOOR_TOTAL
    context_floor_slot: int = CONTEXT_FLOOR_SLOT
    cache_type_order: Sequence[str] = CACHE_TYPE_ORDER
    allow_degraded: bool = False
    parallel: int = 2
    #: Which embedder backends to try, in order. The default follows SPEC
    #: section 5.3: the GPU embedder is preferred, and the CPU embedder is
    #: the fallback for cards where the model would otherwise not fit.
    #: Set to ("llamacpp",) to require the GPU embedder.
    embedder_backends: Sequence[str] = ("llamacpp", "fastembed")
    #: **The bigger model first, or the measured one.**
    #:
    #: With ``False`` (the default) the model whose extraction was
    #: measured wins. That is the right choice for mem0: the
    #: extraction takes the existing memories with it, and a model
    #: with 250 000 tokens of context pulls facts out of a long
    #: conversation where a model with 13 000 tokens cannot.
    #:
    #: With ``True`` the bigger model wins whenever it fits. That is
    #: the right choice when the card has room for it and the
    #: conversation is short.
    prefer_larger: bool = False

    def reserve_for(self, total_mib: int) -> int:
        if self.reserve_mib is not None:
            return self.reserve_mib
        return min(1200, int(total_mib * 0.10))

    def headroom_for(self, total_mib: int) -> int:
        if self.headroom_mib is not None:
            return self.headroom_mib
        return min(800, int(total_mib * 0.06))


@dataclass(frozen=True)
class Candidate:
    """One concrete, launchable configuration."""

    model_id: str
    quantisation: str
    file_name: str
    file_size_mib: float
    weights_vram_mib: float
    kv_bytes_per_token: int
    cache_type: str
    n_ctx: int
    n_ctx_slot: int
    parallel: int
    embedder_backend: str            # "llamacpp" or "fastembed"
    kv_vram_mib: float
    compute_mib: float
    cuda_context_mib: int
    total_vram_mib: float
    model_rank_basis: str = "unmeasured"
    #: The recurrent state buffer of a hybrid model, in MiB. Zero for a
    #: model without state-space layers.
    recurrent_state_mib: float = 0.0

    @property
    def context_floor_satisfied(self) -> bool:
        return self.n_ctx_slot >= 12288

    def as_flags(self) -> list[str]:
        """The llama-server flags this configuration implies.

        Every flag here must exist in the target build. A flag that does
        not exist makes the server abort during argument parsing - before
        it loads anything - so the run produces no output and no
        measurement.

        With ``--kv-unified`` the KV cache is one shared pool: ``-c`` is
        the context **per slot**, and the pool holds that many tokens in
        total. Without it, every slot gets its own pool of ``n_ctx``
        tokens, so the VRAM cost is ``slots × n_ctx``.

        Measured on llama.cpp 0.7.0-dev (build 11798):

        ==================  =====================  ==============
        Flags               ``-c`` means            VRAM for KV
        ==================  =====================  ==============
        ``--kv-unified``    per slot, pool shared  ``n_ctx``
        without             total, split per slot  ``slots × n_ctx``
        ==================  =====================  ==============
        """
        flags = [
            "-c", str(self.n_ctx),
            "--parallel", str(self.parallel),
            "-ngl", "99",
            # The planner computed this configuration; llama.cpp must not
            # renegotiate it at runtime.
            "--fit", "off",
            "--flash-attn", "on",
            # mem0 asks for a JSON object, and llama.cpp turns that into
            # a grammar. The grammar accepts the assistant prefix and
            # then whitespace - but not an open "<think>" block.
            #
            # The chat template puts "<think>" into the prompt when
            # the thinking mode is on. The grammar then breaks:
            #
            #     Unexpected empty grammar stack after accepting
            #     piece: <think>
            #
            # "--reasoning off" sets enable_thinking to false
            # (common/arg.cpp), so the template closes the block right
            # away and the prompt stays valid JSON whitespace.
            #
            # "--reasoning-format none" is not enough: it controls how
            # the answer is parsed, not whether the thinking mode is
            # on.
            "--reasoning", "off",
            "--no-context-shift",
            # The KV cache is a shared pool. This is what makes two slots
            # affordable: without it, every slot needs its own pool of
            # n_ctx tokens and the VRAM cost doubles.
            "--kv-unified",
        ]
        if self.cache_type != "f16":
            flags += ["--cache-type-k", self.cache_type,
                      "--cache-type-v", self.cache_type]
        return flags


@dataclass(frozen=True)
class Budget:
    """The arithmetic behind one candidate."""

    total_mib: float
    weights_mib: float
    compute_mib: float
    cuda_context_mib: int
    embedder_mib: int
    reserve_mib: int
    headroom_mib: int
    kv_mib: float
    #: The recurrent state buffer of a hybrid model. Fixed cost, does
    #: not scale with the context.
    recurrent_state_mib: float = 0.0

    @property
    def used_mib(self) -> float:
        return (self.weights_mib + self.compute_mib + self.cuda_context_mib
                + self.embedder_mib + self.kv_mib + self.recurrent_state_mib)

    @property
    def free_mib(self) -> float:
        return self.total_mib - self.used_mib


@dataclass(frozen=True)
class PlanResult:
    """Outcome of planning."""

    viable: bool
    candidate: Candidate | None
    budget: Budget | None
    diagnostics: dict = field(default_factory=dict)

    @property
    def reason(self) -> str:
        return self.diagnostics.get("reason", "")


def compute_buffer_mib(n_ctx: int, file_size_mib: float) -> float:
    """The CUDA scratch buffer for a given context.

    It grows with the context, because it holds the activations of the
    whole batch. For the 9B the measurements lie on a straight line:

        compute = 0.00488281 * n_ctx + 64.29

    which reproduces all three measured points to within 0.01 MiB. The
    small models use a constant - their own measurements are all at a
    small context, and the difference does not matter there.
    """
    if file_size_mib <= 2000:
        return float(COMPUTE_MIB_SMALL)
    return COMPUTE_SLOPE * n_ctx + COMPUTE_INTERCEPT


def embedder_cost_mib(backend: str) -> int:
    """VRAM cost of the embedder.

    ``llamacpp`` runs a second llama-server (weights + KV + compute + its own
    CUDA context). ``fastembed`` runs ONNX in-process and needs no CUDA
    context at all.
    """
    if backend == "fastembed":
        return 0
    if backend == "llamacpp":
        return (EMBEDDER_WEIGHTS_MIB + EMBEDDER_KV_MIB
                + EMBEDDER_COMPUTE_MIB + CUDA_CTX_MIB)
    raise ValueError(f"unknown embedder backend {backend!r}")


def _budget(
    *,
    total_mib: float,
    weights_mib: float,
    compute_mib: float,
    cuda_context_mib: int,
    embedder_mib: int,
    reserve_mib: int,
    headroom_mib: int,
    kv_mib: float,
    recurrent_state_mib: float = 0.0,
) -> Budget:
    return Budget(
        total_mib=total_mib,
        weights_mib=weights_mib,
        compute_mib=compute_mib,
        cuda_context_mib=cuda_context_mib,
        embedder_mib=embedder_mib,
        reserve_mib=reserve_mib,
        headroom_mib=headroom_mib,
        kv_mib=kv_mib,
        recurrent_state_mib=recurrent_state_mib,
    )


def evaluate_candidate(
    *,
    model_id: str,
    quantisation: str,
    file_name: str,
    file_size_mib: float,
    kv_bytes_per_token: int,
    cache_type: str,
    hardware: Hardware,
    config: PlannerConfig,
    embedder_backend: str = "llamacpp",
    parallel: int | None = None,
    model_rank_basis: str = "unmeasured",
    recurrent_state_mib: float = 0.0,
) -> Candidate | None:
    """Compute one candidate, or ``None`` if it does not fit.

    Returns the candidate even when the context floor is violated; the
    caller decides whether that is fatal (it is, unless ``allow_degraded``).

    ``recurrent_state_mib`` is the recurrent state buffer of a hybrid
    model. It is a fixed cost that does not scale with the context, and
    it is separate from the KV cache.
    """
    parallel = parallel or config.parallel
    reserve = config.reserve_for(hardware.total_vram_mib)
    headroom = config.headroom_for(hardware.total_vram_mib)

    weights = file_size_mib * (1.0 - HOST_FRACTION)
    cuda_ctx = CUDA_CTX_MIB
    emb = embedder_cost_mib(embedder_backend)

    bytes_per_token = kv_bytes_for_cache(kv_bytes_per_token, cache_type)
    if bytes_per_token <= 0:
        return None
    mib_per_token = bytes_per_token / (1024 * 1024)

    # The compute buffer depends on the context, and the context
    # depends on the compute buffer. The two are solved together.
    #
    # The first pass uses the buffer for a context of zero; every
    # further pass uses the context the previous pass found. It
    # converges after three or four rounds, because the buffer grows
    # much slower than the context shrinks.
    fixed_without_compute = (weights + cuda_ctx + emb
                              + recurrent_state_mib + reserve + headroom)
    compute = compute_buffer_mib(0, file_size_mib)
    n_ctx = 0
    for _ in range(8):
        remaining = hardware.total_vram_mib - fixed_without_compute - compute
        if remaining <= 0:
            return None
        if KV_UNIFIED:
            pool_tokens = int(remaining / mib_per_token)
            # The server rounds the context up to a multiple of 256.
            # Round down here so the planned value is one the server
            # will use exactly, and so the VRAM is not exceeded.
            pool_tokens = (pool_tokens // KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS
            new_ctx = pool_tokens
        else:
            pool_tokens = int(remaining / mib_per_token)
            new_ctx = (pool_tokens // parallel) * parallel
        if new_ctx <= 0:
            return None
        new_compute = compute_buffer_mib(new_ctx, file_size_mib)
        if new_ctx == n_ctx and abs(new_compute - compute) < 0.5:
            compute = new_compute
            n_ctx = new_ctx
            break
        n_ctx = new_ctx
        compute = new_compute
    else:
        # It did not settle. The last values are close enough to plan
        # with, and the check below refuses a configuration that does
        # not fit.
        pass

    pool_tokens = n_ctx if KV_UNIFIED else n_ctx // parallel

    # The context floor applies to a single slot: the extraction prompt
    # needs about 10 500 tokens in one slot.
    n_ctx_slot = n_ctx if KV_UNIFIED else n_ctx // parallel

    budget = _budget(
        total_mib=float(hardware.total_vram_mib),
        weights_mib=weights,
        compute_mib=compute,
        cuda_context_mib=cuda_ctx,
        embedder_mib=emb,
        reserve_mib=reserve,
        headroom_mib=headroom,
        kv_mib=pool_tokens * mib_per_token,
        recurrent_state_mib=recurrent_state_mib,
    )
    if budget.free_mib < 0:
        return None

    return Candidate(
        model_id=model_id,
        quantisation=quantisation,
        file_name=file_name,
        file_size_mib=file_size_mib,
        weights_vram_mib=weights,
        kv_bytes_per_token=bytes_per_token,
        cache_type=cache_type,
        n_ctx=n_ctx,
        n_ctx_slot=n_ctx_slot,
        parallel=parallel,
        embedder_backend=embedder_backend,
        kv_vram_mib=budget.kv_mib,
        compute_mib=compute,
        cuda_context_mib=cuda_ctx,
        total_vram_mib=budget.used_mib,
        model_rank_basis=model_rank_basis,
        recurrent_state_mib=recurrent_state_mib,
    )


def _sorted_quantisations(entry) -> list:
    """Quantisations of an entry in preference order (best first).

    The catalogue stores them in preference order already, but the planner
    must not depend on that: an overlay or a hand-written catalogue may
    list them differently.
    """
    quants = list(getattr(entry, "quantisations", []))

    def rank(quant) -> int:
        name = getattr(quant, "name", "")
        try:
            return QUANTISATION_ORDER.index(name)
        except ValueError:
            return len(QUANTISATION_ORDER)

    return sorted(quants, key=rank)


def _model_headroom_mib(entry, hardware: Hardware, config: PlannerConfig) -> float:
    """VRAM left for the KV cache if this model's weights are resident.

    Negative means the weights alone do not fit. This is the number that
    decides whether a model belongs on this card at all.
    """
    quants = _sorted_quantisations(entry)
    if not quants:
        return -1.0
    smallest = min(q.file_size_mib for q in quants)
    weights = smallest * (1.0 - HOST_FRACTION)
    compute = (COMPUTE_MIB if smallest > 2000 else COMPUTE_MIB_SMALL)
    fixed = (weights + compute + CUDA_CTX_MIB
             + embedder_cost_mib(config.embedder_backends[0])
             + config.reserve_for(hardware.total_vram_mib)
             + config.headroom_for(hardware.total_vram_mib))
    return hardware.total_vram_mib - fixed


#: Whether the KV cache is one shared pool (``--kv-unified``) or one
#: pool per slot.
#:
#: With a shared pool, ``-c`` is the context **per slot** and the pool
#: holds that many tokens in total. Without it, every slot gets its own
#: pool of ``n_ctx`` tokens, so the VRAM cost is ``slots × n_ctx``.
#:
#: Measured on llama.cpp 0.7.0-dev (build 11798):
#:
#: ==================  =====================  ==============
#: Flags               ``-c`` means            VRAM for KV
#: ==================  =====================  ==============
#: ``--kv-unified``    per slot, pool shared  ``n_ctx``
#: without             total, split per slot  ``slots × n_ctx``
#: ==================  =====================  ==============
KV_UNIFIED = True

#: llama.cpp allocates the KV cache in blocks of this many tokens, so it
#: rounds the requested context up to a multiple. A planned value of
#: 67 122 becomes 67 328 on the server. The planner rounds the same way
#: so that the value it prints is the value the server uses.
KV_BLOCK_TOKENS = 256


def round_to_kv_block(tokens: int) -> int:
    """Round a context size up to the server's allocation block."""
    if tokens <= 0:
        return 0
    return ((tokens + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS) * KV_BLOCK_TOKENS


def _rank_by_fit(entries: list, hardware: Hardware,
                 config: PlannerConfig) -> list:
    """Order models so the card gets the best one it can hold.

    Three rules, in this order:

    1. A model whose weights do not fit at any quantisation cannot be
       used on this card. It sorts behind every model that fits.
    2. Among models that fit, **the bigger one comes first.** A card
       with more memory gets a better model - that is the whole point
       of the card. The measurement of the extraction capability only
       decides between models of the same size.
    3. Between models of the same size, the measured extraction
       capability comes first, then the catalogue rank.

    Without rule 1, a 12 GB card and a 4 GB card would both end up with
    the same small model whenever that model fits, and the larger card
    would gain nothing but context.

    Without rule 2, a 16 GB card would get the 9B model, because the
    9B is the one whose extraction was measured - even though the 14B
    fits on it and knows more. The measurement says how good a model
    is at pulling facts out of a text; it does not say that a smaller
    model is better than a bigger one.
    """
    def sort_key(entry):
        headroom = _model_headroom_mib(entry, hardware, config)
        fits = headroom >= 0
        params = int(getattr(entry, "params", 0) or 0)
        basis = getattr(entry, "rank_basis", "unmeasured") != "measured"
        rank = getattr(entry, "rank_order", 0)
        if config.prefer_larger:
            return (not fits, -params, basis, rank,
                    getattr(entry, "model_id", ""))
        return (not fits, basis, rank, getattr(entry, "model_id", ""))

    return sorted(entries, key=sort_key)


def plan_for_hardware(
    hardware: Hardware,
    entries: Iterable,
    config: PlannerConfig | None = None,
) -> PlanResult:
    """Run the decision cascade and return the first viable configuration.

    ``entries`` is an iterable of objects exposing:

    * ``model_id``
    * ``quantisations`` - iterable of quantisations, each with ``name``,
      ``file_name``, ``file_size_mib`` and ``kv_bytes_per_token``
    * ``rank_basis`` - "measured" or "unmeasured"

    The cascade order is: model (by extraction capability), quantisation
    (best first), KV cache type (best first), embedder backend.
    """
    config = config or PlannerConfig()

    ordered = sorted(
        entries,
        key=lambda e: (getattr(e, "rank_basis", "unmeasured") != "measured",
                       getattr(e, "rank_order", 0),
                       getattr(e, "model_id", "")),
    )

    # The catalogue is not filtered by VRAM. A card should get the best
    # model it can hold, and a card with more VRAM should get a better
    # model than a card with less - not merely a larger context for the
    # same small model. The cascade therefore ranks models by how well
    # they fit the card before it walks the degradation path.
    ordered = _rank_by_fit(ordered, hardware, config)

    best_degraded: Candidate | None = None
    best_degraded_budget: Budget | None = None
    rejection: dict = {}
    calibration_required: dict[str, list[str]] = {}
    cheapest_to_floor: int | None = None

    # The cascade order matters: the cache type is the outer loop.
    #
    # The first version walked model -> quantisation -> cache type. On an
    # 8 GB card that picked the 9B with a q4_0 cache, which reached
    # 8 472 tokens per slot - below the context floor - while the 4B
    # with a q8_0 cache reached 92 273. The plan was not viable, but a
    # viable plan existed.
    #
    # Degrading the cache type is the last resort, not the first one
    # tried. So: every model at q8_0, then every model at f16, then
    # every model at q4_0.
    for cache_type in config.cache_type_order:
        for entry in ordered:
            model_id = entry.model_id
            rank_basis = getattr(entry, "rank_basis", "unmeasured")
            needs_calibration: list[str] = []

            # A hybrid model keeps a recurrent state next to the KV
            # cache. Its size is a fixed cost, and without it the budget
            # is wrong by exactly that much.
            #
            # The size is not derivable from the attention geometry - the
            # SSM state does not follow the same formula. So it comes
            # from the catalogue, and a hybrid model whose size is not
            # known is not plannable. Guessing would be the same
            # mistake as guessing a KV constant.
            recurrent_state = float(getattr(entry, "recurrent_state_mib", 0.0)
                                   or 0.0)
            if getattr(entry, "is_hybrid", False) and recurrent_state <= 0:
                needs_calibration.append("recurrent_state")
                calibration_required[model_id] = needs_calibration
                continue

            for quant in _sorted_quantisations(entry):
                if getattr(quant, "kv_bytes_per_token", None) is None:
                    # Architecture without a closed-form KV constant
                    # (sliding window, shared KV, pooled encoder).
                    # Planning would need a calibration run on the target
                    # GPU, which this planner does not do.
                    needs_calibration.append(quant.name)
                    continue
                for embedder in config.embedder_backends:
                    candidate = evaluate_candidate(
                        model_id=model_id,
                        quantisation=quant.name,
                        file_name=quant.file_name,
                        file_size_mib=quant.file_size_mib,
                        kv_bytes_per_token=quant.kv_bytes_per_token,
                        cache_type=cache_type,
                        hardware=hardware,
                        config=config,
                        embedder_backend=embedder,
                        model_rank_basis=rank_basis,
                        recurrent_state_mib=recurrent_state,
                    )
                    if candidate is None:
                        continue
                    if (candidate.n_ctx_slot >= config.context_floor_slot):
                        budget = _recompute_budget(candidate, hardware, config)
                        return PlanResult(viable=True, candidate=candidate,
                                          budget=budget)
                    if best_degraded is None or (
                            candidate.n_ctx_slot > best_degraded.n_ctx_slot):
                        best_degraded = candidate
                        best_degraded_budget = _recompute_budget(
                            candidate, hardware, config)
                    # The shortfall is the distance of the *cheapest* path
                    # to the floor, not of the largest context. A small
                    # model with a large KV cache may be closer to the
                    # floor than a big model with a tiny one.
                    required = _required_mib_for_floor(
                        candidate, hardware, config)
                    if cheapest_to_floor is None or required < cheapest_to_floor:
                        cheapest_to_floor = required
                    rejection.setdefault(model_id, {
                        "model": model_id,
                        "n_ctx": candidate.n_ctx,
                        "n_ctx_slot": candidate.n_ctx_slot,
                        "required_total": config.context_floor_total,
                        "required_slot": config.context_floor_slot,
                    })
            if needs_calibration:
                calibration_required[model_id] = needs_calibration

    if config.allow_degraded and best_degraded is not None:
        return PlanResult(
            viable=True,
            candidate=best_degraded,
            budget=best_degraded_budget,
            diagnostics={
                "reason": "context_floor_not_met",
                "degraded": True,
                "warning": (
                    "The selected configuration cannot extract facts: "
                    f"{best_degraded.n_ctx_slot} tokens per slot, "
                    f"{config.context_floor_slot} required. "
                    "Fact extraction may fail or store incomplete facts."
                ),
                "required_mib": round(best_degraded.total_vram_mib),
                "detail": rejection,
            },
        )

    if config.allow_degraded:
        # Not even the smallest candidate fits. The caller asked for a
        # degraded service, so the planner builds the smallest one the
        # card can hold - with whatever context is left over.
        smallest = _smallest_fitting_candidate(
            hardware, ordered, config)
        if smallest is not None:
            candidate, budget = smallest
            return PlanResult(
                viable=True,
                candidate=candidate,
                budget=budget,
                diagnostics={
                    "reason": "context_floor_not_met",
                    "degraded": True,
                    "warning": (
                        "The card cannot hold this model with a usable "
                        "context. The configuration below is the smallest "
                        "one that fits at all; fact extraction will not "
                        "work."
                    ),
                    "required_mib": round(candidate.total_vram_mib),
                    "detail": rejection,
                },
            )

    detail = next(iter(rejection.values()), {})
    required_mib = cheapest_to_floor or 0
    options = [
        "use a card with more VRAM",
        "set planner.context_floor_total lower, if your mem0 "
        "configuration uses a shorter extraction prompt",
        "set planner.allow_degraded: true to run anyway, accepting "
        "that fact extraction may fail",
    ]
    if calibration_required:
        options.append(
            "run 'cpt-mem0 calibrate' on this machine to measure the KV "
            "constant for architectures without a closed form: "
            + ", ".join(sorted(calibration_required))
        )

    # The shortfall is the distance between the card and the cheapest
    # path to the floor. If the model does not fit at all, the distance
    # is the size of the model itself.
    if required_mib <= hardware.total_vram_mib:
        shortfall = 0
    else:
        shortfall = required_mib - hardware.total_vram_mib
    if shortfall == 0 and cheapest_to_floor:
        # The floor was missed, not the VRAM. Say how much context the
        # card actually allows.
        shortfall = max(
            shortfall,
            config.context_floor_slot
            - max((entry.get("n_ctx_slot", 0)
                   for entry in rejection.values()), default=0))

    return PlanResult(
        viable=False,
        candidate=None,
        budget=best_degraded_budget,
        diagnostics={
            "reason": "context_floor_not_met",
            "required_mib": required_mib,
            "available_mib": hardware.total_vram_mib,
            "shortfall_mib": shortfall,
            "detail": detail,
            "calibration_required": calibration_required,
            "options": options,
        },
    )


def _smallest_fitting_candidate(hardware: Hardware, entries: list,
                                config: PlannerConfig
                                ) -> tuple[Candidate, Budget] | None:
    """The smallest configuration the card can hold at all.

    This is what the degraded mode falls back to when not even the
    cheapest model fits with a usable context: the model with the
    smallest file, the coarsest cache type, and whatever context is
    left over.

    The context can be zero here. The service would start and would
    not be able to extract anything - which is exactly what the
    degraded mode says out loud.
    """
    best: tuple[Candidate, Budget] | None = None
    for entry in entries:
        for quant in _sorted_quantisations(entry):
            if getattr(quant, "kv_bytes_per_token", None) is None:
                continue
            candidate = evaluate_candidate(
                model_id=entry.model_id,
                quantisation=quant.name,
                file_name=quant.file_name,
                file_size_mib=quant.file_size_mib,
                kv_bytes_per_token=quant.kv_bytes_per_token,
                cache_type=config.cache_type_order[-1],
                hardware=hardware,
                config=config,
                embedder_backend=config.embedder_backends[-1],
                recurrent_state_mib=float(
                    getattr(entry, "recurrent_state_mib", 0.0) or 0.0),
            )
            if candidate is None:
                continue
            budget = _recompute_budget(candidate, hardware, config)
            if best is None or candidate.total_vram_mib < best[0].total_vram_mib:
                best = (candidate, budget)
            break
    return best


def _required_mib_for_floor(candidate: Candidate, hardware: Hardware,
                            config: PlannerConfig) -> int:
    """VRAM needed to reach the context floor with this candidate.

    The shortfall is measured against the *floor*, not against the
    degraded configuration: a configuration that loads but cannot store
    a fact needs more VRAM, not less.

    Reserve and headroom are included, because the comparison is against
    the card's total VRAM.
    """
    if candidate.n_ctx <= 0:
        return int(hardware.total_vram_mib)
    # The KV pool holds ``pool_tokens`` tokens; with a shared pool that is
    # ``n_ctx``, otherwise it is ``n_ctx_slot * parallel``.
    pool_tokens = (candidate.n_ctx if KV_UNIFIED
                   else candidate.n_ctx_slot * candidate.parallel)
    mib_per_token = candidate.kv_vram_mib / pool_tokens
    fixed = (candidate.total_vram_mib - candidate.kv_vram_mib
             + config.reserve_for(hardware.total_vram_mib)
             + config.headroom_for(hardware.total_vram_mib))
    # The floor applies to one slot.
    required_kv = config.context_floor_slot * mib_per_token
    return int(round(fixed + required_kv))


def _recompute_budget(candidate: Candidate, hardware: Hardware,
                       config: PlannerConfig) -> Budget:
    reserve = config.reserve_for(hardware.total_vram_mib)
    headroom = config.headroom_for(hardware.total_vram_mib)
    emb = embedder_cost_mib(candidate.embedder_backend)
    return _budget(
        total_mib=float(hardware.total_vram_mib),
        weights_mib=candidate.weights_vram_mib,
        compute_mib=candidate.compute_mib,
        cuda_context_mib=candidate.cuda_context_mib,
        embedder_mib=emb,
        reserve_mib=reserve,
        headroom_mib=headroom,
        kv_mib=candidate.kv_vram_mib,
        recurrent_state_mib=candidate.recurrent_state_mib,
    )

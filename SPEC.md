# Cpt_Mem0 — Technical Specification

> Self-contained local mem0 service for consumer GPUs.
> Runs a language model, an embedder and a vector store on one graphics card,
> without cloud services, without Docker, and without depending on any
> other application being installed.

**Status:** Specification (v0.1) — no implementation code yet.
**Repository:** `rotorbase/Cpt_Mem0`

---

## 1. Purpose

Give an agent a persistent semantic memory that lives entirely on the user's
own machine.

The problem this solves: **a local LLM that runs but has almost no context.**
`llama.cpp`'s `--fit` mechanism (enabled by default) silently reduces
`n_ctx` toward its `--fit-ctx` floor (4096) and pushes layers into host RAM
when the model does not fit the card. The model loads, answers, and has
almost no usable context. Users see "the model works" and no explanation.

Cpt_Mem0 computes the configuration **before** launch instead of letting
`llama.cpp` negotiate it at runtime.

### 1.1 Non-goals

- Not a chat application. The LLM endpoint is exposed for clients to use.
- Not a Hermes plugin. It shares no code, paths, configuration or lifecycle
  with any agent framework.
- Not a cloud service. No request leaves the machine.
- Not a model browser. Model selection is solved automatically, not browsed.

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| P1 | **Self-contained** | The app ships and manages its own `llama.cpp` binary, its own Python environment, its own model directory. No dependency on a system-wide `llama-server`, Ollama, or a pre-existing venv. |
| P2 | **Measure, don't assume** | Every claim about GPU state is backed by a reading. Open ports prove nothing; VRAM does. |
| P3 | **Compute, don't negotiate** | The VRAM budget is solved up front. `--fit` is turned **off**. |
| P4 | **Degrade, don't fail** | If the embedder doesn't fit on the GPU, it moves to CPU. If the largest model doesn't fit, a smaller one is selected. The service always starts. |
| P5 | **Explicit upgrades** | Nothing updates itself. The user runs `cpt-mem0 update` and sees what changes. |
| P6 | **Portable** | No hardcoded paths, no OS-specific assumptions beyond a platform abstraction layer. |
| P7 | **Data stays local** | No telemetry, no outbound requests, no analytics. |

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Cpt_Mem0 daemon  (single process, owns the store)              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ mem0 engine (in-process)                                  │  │
│  │   llm      → http://127.0.0.1:<llm_port>/v1               │  │
│  │   embedder → llama.cpp OR fastembed (in-process)          │  │
│  │   store    → Qdrant embedded (path=)                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│  REST API  +  MCP server  (localhost only)                      │
└─────────────────────────────────────────────────────────────────┘
        │                              │
        │ spawns / supervises          │ reads VRAM
        ▼                              ▼
┌──────────────────┐          ┌────────────────────┐
│ llama-server     │          │  VRAM planner      │
│ (LLM)            │          │  (reads nvidia-smi │
├──────────────────┤          │   / rocm-smi)      │
│ llama-server     │          └────────────────────┘
│ (embedder, GPU)  │
└──────────────────┘
```

### 3.1 Process model

**One** Cpt_Mem0 daemon process owns the vector store. This is a hard
requirement, not a preference: Qdrant's embedded mode takes an exclusive
filesystem lock on its storage directory (`portalocker`). A second client on
the same path fails with:

> `RuntimeError: Storage folder ... is already accessed by another instance of
> Qdrant client. If you require concurrent access, use Qdrant server instead.`

*Verified on qdrant-client 1.19/1.20, Windows and Linux.*

Consequence: the daemon serves **all** clients. Clients never open the store
directly; they talk to the REST API or MCP endpoint.

The daemon supervises up to two `llama-server` child processes (LLM and, if
GPU-embedded, the embedder). Child processes are restarted on crash with
bounded backoff.

### 3.2 Component boundaries

| Component | Responsibility | Failure behaviour |
|---|---|---|
| `planner` | Reads GPU state and model metadata, computes the launch configuration | Fatal (cannot start without a plan) |
| `runtime` | Owns the `llama-server` child processes, checks health | Restart with backoff |
| `engine` | Wraps mem0 (LLM + embedder + store) | Daemon exits with a clear error |
| `api` | REST + MCP surface, key authentication | Daemon exits |
| `models` | Model download/verification, catalogue | Per-model, non-fatal |
| `binaries` | `llama.cpp` binary provisioning and update | Fatal on first run, recoverable via `update` |

---

## 4. The VRAM planner

The planner is the core of the product. It answers one question:

> Given this GPU, which model, at which quantization, with how much context,
> with the embedder where — all simultaneously resident?

### 4.1 The budget equation

```
VRAM_total
  - VRAM_reserve           (desktop / OS / other apps)
  - VRAM_headroom          (fragmentation safety margin)
  - weights_LLM            (model file × (1 − host_fraction))
  - compute_buffer_LLM     (CUDA scratch, scales with model size)
  - cuda_context_LLM       (per llama-server process)
  - embedder_cost          (0 if CPU, else weights + KV + compute + context)
  - KV_cache(n_ctx)        = n_ctx × bytes_per_token
  = free VRAM
```

### 4.2 KV bytes per token

This is where naive implementations fail. The KV cache size is **not**
`layers × heads × dim × 2` for every model.

```
bytes_per_token =
    kv_layers × (n_kv_heads × key_length + n_kv_heads × value_length) × element_bytes
```

where `kv_layers` is:

- **standard architectures** (llama, qwen3, mistral): `kv_layers = block_count`
- **hybrid architectures** (Qwen3.5, Qwen3-Next): `kv_layers = block_count ÷ full_attention_interval`
  — only every N-th layer carries a KV cache, the others are SSM/Mamba layers

The GGUF header exposes `full_attention_interval` for hybrid models. Using
the naive formula on a hybrid model **overestimates the KV cache by the
interval factor**.

**Verification (RTX 3060, 14 real `llama-server` logs):**

| Model | `full_attention_interval` | Naive formula | Measured | Corrected formula |
|---|---|---|---|---|
| Qwen3.5-9B-Q4_K_M | 4 | 131 072 B/tok | **32 768 B/tok** | 32 768 B/tok ✓ |
| Qwen3.5-9B-Q8_0 | 4 | 131 072 B/tok | **32 768 B/tok** | 32 768 B/tok ✓ |

Measured values were derived by linear regression over 8 log files with
different `n_ctx` values (102 912 … 109 568) — exact agreement, no residual.

**Counter-example:** `nomic-embed-text-v1.5` (BERT architecture) does not
follow the per-layer formula at all. `llama.cpp` allocates a shared/minimum
pool: 12 MiB at `n_ctx` 2048 — the *same* value as for `gemma-4-E2B`, a
completely different model. This is a floor, not a slope.

### 4.3 Calibration

Because no closed form is universally correct, the planner supports two
sources and marks its confidence:

| Source | Method | Confidence |
|---|---|---|
| **Calibrated** | Read `CUDA0 KV buffer size = N MiB` from the server log at two different `n_ctx` values, derive the slope | Exact, any architecture |
| **Formula** | Compute from GGUF metadata as in 4.2 | Exact for standard + hybrid; unreliable for BERT/SWA/shared-KV |

Calibration runs once per model and is cached in the model catalogue
(`~/.local/share/cpt-mem0/models.json`). A calibration run costs one model
load (~45 s for a 9B) and is only triggered when the formula is untrustworthy
or when the user runs `cpt-mem0 calibrate`.

### 4.4 Decision cascade

The planner tries configurations in order and returns the first that fits.
Each step is a *degradation*, never a failure:

```
for model in catalogue.sorted_by_quality_desc:
    for quant in [Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, IQ3_XXS]:
        for cache_type in [q8_0, f16, q4_0]:
            for embedder_mode in [gpu, cpu]:
                cfg = compute(model, quant, cache_type, embedder_mode)
                if not cfg.fits(reserve, headroom):        continue
                if cfg.n_ctx < CONTEXT_FLOOR_TOTAL:        continue   # §4.7
                if cfg.n_ctx_slot < CONTEXT_FLOOR_SLOT:     continue   # §4.7
                return cfg
return "no viable configuration"
```

Two constraints gate every candidate: the VRAM budget (§4.1) and the
context floor (§4.7). A candidate that satisfies the budget but not the
floor is rejected — a service that cannot extract facts is not a service.

The model order is **measured extraction capability**, not parameter
count (§17.4). Measured models rank first; unmeasured models rank below
them and the output labels the basis.

The final failure message names the binding constraint:

> No viable configuration for 4096 MiB of VRAM.
> The smallest model that can extract facts needs 1654 MiB of weights plus
> 108 MiB of KV cache (12288 tokens), plus 990 MiB process overhead and
> 654 MiB reserve — 3406 MiB in total.
> Your card is 0 MiB short. Options: a card with 5 GB or more, or set
> `planner.context_floor` to a lower value if your mem0 version uses a
> shorter extraction prompt.

### 4.4.1 Configuration

```yaml
planner:
  reserve_mib: null         # null = 10 % of total VRAM
  headroom_mib: null        # null = 6 % of total VRAM
  context_floor_total: 21000    # minimum n_ctx for the service to function
  context_floor_slot: 12288     # minimum per-slot context for extraction
  kv_cache_type_order: [q8_0, f16, q4_0]
```

The floors are configurable because the extraction prompt length depends on
the mem0 version and on how the user configures fact extraction. The
defaults are the measured values.

### 4.5 KV cache quantization

`--cache-type-k` / `--cache-type-v` are the single strongest lever and are
**never set by `llama.cpp` by default**:

| Cache type | KV size (16 384 tokens) | Note |
|---|---|---|
| `f16` | 512 MiB | default |
| `q8_0` | 272 MiB | minimal quality loss — **recommended default** |
| `q4_0` | 144 MiB | larger loss on the value cache |

Halving the KV cost doubles the context for the same VRAM.

**Derivation of the quantised sizes.** The block layouts are llama.cpp's,
read from `ggml/src/ggml-common.h` (`block_q8_0`, `block_q4_0`):

| Type | Block | Bytes per 32 values | Bytes per value | Share of f16 |
|---|---|---|---|---|
| `f16` | 2 B half | 64 | 2.0 | 100 % |
| `q8_0` | 2 B half scale + 32 B int8 | 34 | 1.0625 | 53.125 % |
| `q4_0` | 2 B half scale + 16 B nibbles | 18 | 0.5625 | 28.125 % |

`llama_kv_cache` allocates K and V as `ggml_new_tensor_3d(type, n_embd_gqa,
kv_size, n_stream)`, so the row size follows `ggml_row_size` exactly —
no additional padding applies for `n_embd_gqa` divisible by 32.

These constants are implemented in `src/cpt_mem0/planner.py`
(`CACHE_ELEMENT_BYTES`) and asserted by
`tests/test_planner.py::TestKvCacheScaling`.

### 4.6 `--fit` must be disabled

The launch configuration always contains `--fit off`. With `--fit` enabled,
`llama.cpp` may reduce `n_ctx` and offload layers at any time, including
after the planner computed a valid configuration. The planner's result must
be the final word.

### 4.7 Context floor — the service must be able to extract facts

A configuration is only valid if the model can actually do mem0's job.
Measured on this machine:

```
prompt eval time = 6110.85 ms / 10449 tokens (1709.91 tokens per second)
```

mem0's fact-extraction prompt is **~10 500 tokens**. A configuration with
less context loads, starts, answers chat requests — and silently fails to
store anything. This is the same class of failure as the `--fit` problem,
just one layer up: the service appears healthy while its primary function
is unavailable.

**Hard requirement:**

```
n_ctx_total  ≥  2 × 10 500  =  21 000      (2 slots: extraction + chat)
n_ctx_slot   ≥  12 288                        (extraction slot, headroom)
```

The planner treats the context floor as a **constraint, not a preference**.
A model/quantization/cache combination that cannot satisfy it is rejected
and the cascade continues.

Consequence: cards below ~5 GB cannot run this service at all with a
4B-class model, because the KV cache required for extraction does not fit
alongside the weights. The planner reports this explicitly rather than
starting a service that cannot store memories.

### 4.8 Two models are almost never worth it

The obvious idea — a small model for extraction, a large one for chat —
costs more than it saves. Every additional `llama-server` process costs:

| Overhead | MiB |
|---|---|
| CUDA context (per process) | ~320 |
| Compute buffer (per process) | ~392 |
| Recurrent state (hybrid model, per process) | ~100 |
| **Total per extra process** | **~812** |

Measured combination results (extraction model with a fixed 12 288-token
slot, chat model at `Q4_K_M`, KV `q8_0`):

| Card | 9B chat + 4B extraction | 9B chat alone |
|---|---|---|
| 12 GB | 58 112 (58 112/slot) | 62 976 (62 976/slot) |
| 8 GB | does not fit | 17 920 (17 920/slot) |
| 6 GB | does not fit | 39 168 (39 168/slot) |

Splitting makes the 8 GB and 6 GB tiers impossible: a second
llama-server needs its own context, its own compute buffer and - for a
hybrid model - its own recurrent state, so the card has to hold two
models at once. On a 12 GB card the split still fits and even leaves
slightly more context for the chat model, because the extraction model
is small and needs only its 12 288-token slot.

Since a 4B extraction model is documented to be within ~6 % of a 9B for
fact extraction, the split brings no quality gain. The single-model
design wins on every axis that matters, and it is the only one that
works below 12 GB.

**Decision: one LLM serves both roles.** The planner may still *consider*
a split (it fits on a 24 GB+ card for a user who wants a large chat model
plus a fast extractor), but it is not the default and it is never required.

### 4.9 Reference results

Computed with the shipped planner against measured constants (RTX 3060
12 GB, embedder resident, 2 slots, KV `q8_0`, context floor 21 000
enforced). Regenerate with `scripts/regenerate_tiers.py`:

| VRAM | Selected configuration | `n_ctx` | Per slot |
|---|---|---|---|
| 12 GB | Qwen3.5-9B Q8_0 | 62 976 | 62 976 |
| 8 GB | Qwen3.5-9B Q5_K_M | 17 920 | 17 920 |
| 6 GB | Qwen3.5-9B IQ3_XXS | 39 168 | 39 168 |
| 5 GB | Qwen3-4B-Instruct-2507 Q5_K_M | 13 056 | 13 056 |
| 4 GB | Qwen3-4B-Instruct-2507 IQ3_XXS | 15 104 | 15 104 |

The tiers are monotonic: more VRAM yields a better model, not merely a
larger context. A card that cannot hold a model's weights never receives
that model.

The 4 GB tier is reachable, but only with a coarse quantisation
(`IQ3_XXS`). Whether that quantisation extracts facts well enough is
the open measurement of §17.4 — the planner computes the fit, not the
quality.

### 4.10 What `-c` means — the KV pool decides

`--kv-unified` makes the KV cache one shared pool. That changes what the
`-c` flag means, and getting it wrong makes the plan wrong in a way that
only shows up on real hardware.

| Flags | `-c` means | VRAM for the KV cache |
|---|---|---|
| with `--kv-unified` | context **per slot**, pool shared | `n_ctx` |
| without | total, split per slot | `slots × n_ctx` |

Measured on llama.cpp 0.7.0-dev (build 11798), with `-c 67122
--parallel 2`:

- with `--kv-unified`: the server reported `n_ctx_slot = 67328`
- without: the server reported `n_ctx_slot = 33792`

The planner therefore treats `-c` as the per-slot limit and budgets one
pool of `n_ctx` tokens — not `n_ctx / slots`. Two slots share the pool;
neither gets half of it. This is what makes two slots affordable: the
alternative doubles the KV cost.

Two details the planner reproduces:

- **The server rounds the context up** to a multiple of 256 tokens,
  because that is the allocation block of the KV tensors. A planned
  67 122 becomes 67 328 on the server. The planner rounds down to the
  same block size, so the value it prints is the value the server uses
  and the VRAM is never exceeded.
- **The context floor applies to one slot**, not to the total. The
  extraction prompt needs about 10 500 tokens in a single slot.

The verification run on 2026-10-24 found the first version of this
wrong: it planned 33 561 per slot and the server ran 67 328. The VRAM
prediction was correct to within 4 MiB (9 775 MiB measured against
9 811 MiB predicted), so the budget model was sound — only the meaning
of `-c` was not.

### 4.11 The recurrent state of a hybrid model

A hybrid architecture mixes attention layers with state-space layers.
Qwen3.5 does: 32 layers, every fourth one carries a KV cache, the
other 24 are state-space layers. Their state lives in its own buffer:

```
llama_memory_recurrent:      CUDA0 RS buffer size =   100.50 MiB
```

That buffer is **not** part of the KV cache. It does not scale with the
context, and it is not affected by the cache type. It is a fixed cost
that a budget without it understates by exactly that much.

The GGUF header says which models have this buffer. The keys are
`<arch>.ssm.state_size`, and a model without them has no recurrent
state:

| Model | Architecture | SSM keys | Recurrent state |
|---|---|---|---|
| Qwen3.5-9B | `qwen35` | yes | 100.50 MiB (measured) |
| Qwen3.5-4B | `qwen35` | yes | not measured yet |
| Qwen3-8B | `qwen3` | no | none |
| Qwen3-4B-Instruct-2507 | `qwen3` | no | none |
| Llama-3.1-8B-Instruct | `llama` | no | none |
| gemma-4-E2B-it | `gemma4` | no | none |

**The size is not derivable.** The attention geometry does not produce
it: the state of a state-space layer does not follow the same formula
as a KV tensor, and none of the combinations of state size, inner size
and group count reproduce the measured 100.50 MiB. So it is stored in
the catalogue as a measured value, and a hybrid model whose size is
unknown is **not plannable** — the planner reports that it needs a
calibration run rather than guessing.

The same rule that applies to the KV constants applies here: a number
that was not measured is not used.

**Why it matters.** With the state and the compute buffer counted, the
12 GB tier changed from 67 072 tokens to 62 976. The earlier plan
claimed 12 287 of 12 288 MiB; the measured reality needed 12 448. The
difference was exactly the two terms that were missing.

### 4.12 The compute buffer grows with the context

The CUDA scratch buffer holds the activations of the batch, so it
grows with the context. Three measurements of the same model:

| Model | Context | Measured |
|---|---|---|
| Qwen3.5-9B-Q8_0 | 54 784 | 331.79 MiB |
| Qwen3.5-9B-Q8_0 | 61 696 | 365.54 MiB |
| Qwen3.5-9B-Q8_0 | 67 072 | 391.79 MiB |

The three points lie on a straight line:

    compute = 0.00488281 * n_ctx + 64.29

which reproduces every measurement to within 0.01 MiB. The planner
uses the formula, so the context and the buffer are solved together:
the buffer depends on the context, and the context depends on the
buffer. The planner iterates until both settle.

A constant was wrong at every context but one. The first version used
290 MiB, taken from an earlier run at a smaller context, and the 12 GB
card was 160 MiB too optimistic as a result.

This also freed the 4 GB tier. With the buffer planned at a constant
of 392 MiB, that card missed the context floor by 253 MiB and was
refused. With the formula it needs 130 MiB at its own context, and the
tier works: 15 104 tokens per slot, 4078 of 4096 MiB used.

### 4.13 The cascade order

Two questions the cascade has to answer: which model, and which
quantisation. The rule is that the better model wins, and within a
model the better quantisation wins - but only if the result is
viable, that is, if it clears the context floor.

That leaves a case the rule does not settle: several quantisations of
the same model are viable, and they give very different contexts. On
an 8 GB card:

| Quantisation | File | Context | VRAM |
|---|---|---|---|
| Q5_K_M | 6273 MiB | 17 920 | 6877 MiB |
| Q4_K_M | 5417 MiB | 53 504 | 6880 MiB |
| IQ4_XS | 4929 MiB | 73 728 | 6880 MiB |
| IQ3_XXS | 3830 MiB | 119 296 | 6881 MiB |

All four use the same VRAM. The difference is how much of it sits in
the KV cache instead of in the weights. The choice is therefore
between better weights and more context - the same VRAM either way.

**Context above the floor is chat buffer, not memory capacity.**

The extraction prompt is 10 449 tokens. A slot holds it from 12 288
tokens upwards. Everything above that serves the chat side.

The extraction quality depends on the weights, not on how much room
the slot has beyond what the prompt needs. A slot with 17 920 tokens
extracts exactly as well as one with 119 296, because the prompt is
the same size either way.

So the cascade takes Q5_K_M: the better quantisation, and the context
that is left over. This is the same principle as the model hierarchy
- the better model wins, and the context is what is left over. A user
who wants the other trade asks for the quantisation by name.

**Resolved:** the better quantisation wins. The context is not a
quality dimension above the floor, so it does not compete with the
weights.

The cache type is degraded **last**, not first. The cascade tries every
model at `q8_0`, then every model at `f16`, then every model at `q4_0`.

Walking model → quantisation → cache type instead would reach a
configuration that fits in VRAM but misses the context floor, while a
viable one exists. On an 8 GB card that ordering produced the 9B with a
`q4_0` cache at 8 472 tokens per slot — below the floor — while the 4B
with a `q8_0` cache reached 92 273.

The arithmetic is not the only constraint. Even when the budget fits,
the context floor is a hard gate: a configuration that loads, answers
chat requests and cannot store a single memory is not a service. The
tool refuses those and reports the shortfall.

---

## 5. Embedder strategy

### 5.1 Two backends, one interface

| Backend | Process | VRAM | Speed (1 fact) | Vector compatible |
|---|---|---|---|---|
| `llamacpp` | separate `llama-server` | 188 MiB | 4.2 ms | reference |
| `fastembed` | in-process (ONNX) | **0 MiB** | 21.3 ms | **yes** |

Both produce 768-dimensional nomic vectors.

### 5.2 Store compatibility between backends — verified

A store written with the `llamacpp` embedder was queried with the
`fastembed` embedder:

| Query | llama.cpp score | fastembed score | Top-1 result |
|---|---|---|---|
| "welche grafikkarte nutze ich" | 0.4869 | 0.4131 | **identical** |
| "womit laeuft der store" | 0.4958 | 0.4403 | **identical** |
| "was entwickelt das projekt" | 0.5766 | 0.5241 | **identical** |

Identical top-1 results in every case. The score difference (~7 %) is the
quantization difference (Q4_K_M vs FP32 ONNX). The reason it works: mem0 and
Qdrant normalise internally, so the differing raw vector norms
(llama.cpp |v| = 1.0, fastembed |v| = 21.4) cancel out.

**Consequence:** switching the embedder backend does **not** require
rebuilding the store. This is what makes the CPU fallback safe.

### 5.3 Model choice

`nomic-ai/nomic-embed-text-v1.5` (768 dim) is the default. Cross-language
retrieval benchmark (documents in DE and EN, queries in DE/EN/FR):

| Model | dim | Size | Top-1 | Margin | ms/query |
|---|---|---|---|---|---|
| **nomic-embed-text-v1.5** | 768 | 0.52 GB | **3/4** | **0.1124** | 17.0 |
| BAAI/bge-small-en-v1.5 | 384 | 0.067 GB | 3/4 | 0.0519 | **3.2** |
| jina-embeddings-v2-base-de | 768 | 0.64 GB | 2/4 | 0.0269 | 15.8 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.22 GB | 1/4 | 0.0384 | 3.7 |

nomic wins on separation margin (2× the runner-up) and its 768 dimensions
match the `llamacpp` backend, so both backends can share one store.

### 5.4 Task prefixes — measured, not implemented

`nomic-embed-text` documents task prefixes (`search_query: ` /
`search_document: `). mem0 2.2.1 already threads the action through the
embedder API:

```python
mem0/memory/main.py:927   query_embedding = self.embedding_model.embed(parsed_messages, "search")
mem0/memory/main.py:1980  embeddings = self.embedding_model.embed(data, memory_action="add")
mem0/memory/main.py:689   vec = self.embedding_model.embed(entity_text, "update")
```

So the mechanism to apply prefixes per action exists. The question is
whether it helps.

**Measured on a 30-document corpus with 46 queries in 5 languages**
(English, German, French, Spanish, Italian), paired comparison with
bootstrap confidence intervals:

| Configuration | Top-1 | MRR | Margin |
|---|---|---|---|
| Without prefix | 31/46 | 0.7675 | 0.1008 |
| Doc + query prefix | 33/46 | 0.7798 | 0.0844 |
| Query prefix only | 29/46 | 0.7457 | 0.0937 |
| Doc prefix only | 34/46 | 0.7934 | 0.0876 |

Paired MRR differences (prefix − no prefix), 95 % bootstrap CI:

| Configuration | Δ MRR | 95 % CI | Verdict |
|---|---|---|---|
| Doc + query | +0.0123 | [−0.0200, +0.0501] | no significant difference |
| Query only | −0.0219 | [−0.0566, +0.0051] | no significant difference |
| Doc only | +0.0259 | [−0.0090, +0.0682] | no significant difference |

**Every confidence interval includes zero.** The effect is not
distinguishable from noise on this corpus. A control case confirms the
test has power: a deliberate mismatch (documents prefixed, queries not)
still scores 34/46 — the prefix does not break retrieval either way.

**Decision: no prefix support.** The measured benefit is not there, and
adding it would create a store-compatibility hazard (a store written with
prefixes is not readable by a build without them) in exchange for an
unproven gain.

The `memory_action` plumbing remains useful for other purposes — it is the
hook a future reranker or a different embedder family would use.

---

## 6. Vector store

**Decision: Qdrant embedded (`path=`).** Not configurable in v1.

| Criterion | pgvector | **Qdrant embedded** | ChromaDB |
|---|---|---|---|
| Server required | PostgreSQL service | **none** | none |
| Docker required | no (but service install) | **no** | no |
| Dependency status | needs PostgreSQL | **`qdrant-client` is a mem0 dependency** | not a mem0 dependency |
| Setup | service + password + port 5432 | **a directory** | a directory |
| Cross-process access | yes | **no (exclusive lock)** | yes |

**Verified working end-to-end** with mem0 2.2.1: `add()` extracted facts,
`search()` returned correct top-1 for German and English queries,
`get_all()` listed them. Store size 28.6 KiB for 4 memories.

The exclusive-lock limitation is acceptable because of the process model in
§3.1: one daemon owns the store.

Postgres remains reachable as a migration source for existing installations
(`cpt-mem0 migrate --from pgvector`).

---

## 7. Binary management

### 7.1 Version policy

`llama.cpp` publishes two kinds of releases:

| Kind | Tag pattern | Assets | Frequency |
|---|---|---|---|
| Tagged release | `v0.7.0` | **1** (`nightly-tag.txt` → points at a b-build) | per release |
| Rolling build | `b11838` | **36** (all platforms, prebuilt CUDA binaries) | daily |

The tagged release carries no binaries itself — it points at the rolling
build that was current when it was cut. **All prebuilt CUDA binaries live in
the b-build releases.**

**Decision: pin an exact b-build. Never track "latest".**

Rationale:
- b-build releases are **permanent** — `b11669` from 2026-10-15 still serves
  all 36 assets. A pinned build is therefore permanently reproducible.
- A floating "latest" would change under the user's feet: a new nightly can
  change model output, KV layout or flag semantics, breaking the planner's
  measurements and the store's embeddings without any user action.
- Pinned builds still allow opt-in updates (P5): the user runs
  `cpt-mem0 update` and reviews the diff.

The pin lives in `config.yaml` as `llama.cpp.build: "b11838"`.

### 7.2 CUDA runtime bundling — the DLL trap

`llama.cpp` ships two Windows/Linux CUDA variants:

| Asset prefix | Contains | Size (win-x64, CUDA 13.4) |
|---|---|---|
| `llama-…-bin-win-cuda-13.4-x64.zip` | binary only | 152.5 MB |
| **`cudart-llama-…-bin-win-cuda-13.4-x64.zip`** | binary **+ bundled CUDA runtime** | **403.9 MB** |

**Decision: always use the `cudart-` variant.**

This matters more than it looks. When the CUDA runtime DLLs are missing from
the binary directory, `llama.cpp` falls back to CPU **silently** — no error,
the port opens, `/v1/models` answers 200, and inference runs at ~6 tok/s
instead of ~47 tok/s. This failure has been observed repeatedly in this
project and is invisible to any check that does not read VRAM.

The `cudart-` bundles remove the failure mode at its source. The planner
still verifies the outcome by measuring VRAM (P2).

### 7.3 Platform matrix

| Platform | Asset | CUDA variant |
|---|---|---|
| Windows x64 | `cudart-llama-bin-win-cuda-13.4-x64.zip` | `cudart-` |
| Windows ARM64 | `cudart-llama-bin-win-cuda-13.4-arm64.zip` | `cudart-` |
| Linux x64 | `cudart-llama-bin-ubuntu-cuda-13.4-x64.tar.gz` | `cudart-` |
| Linux ARM64 | `cudart-llama-bin-ubuntu-cuda-13.4-arm64.tar.gz` | `cudart-` |
| macOS (Apple Silicon) | `llama-bin-macos-arm64.tar.gz` | Metal, no CUDA |
| CPU-only fallback | `llama-bin-win-cpu-x64.zip`, `llama-bin-ubuntu-x64.tar.gz` | — |

CUDA 12.4 variants exist for older drivers. The installer selects by
detecting the installed driver version.

### 7.4 Update procedure

```
cpt-mem0 update                 # show what is available, then apply
cpt-mem0 update --check         # report only, no changes
cpt-mem0 update --build b11900  # pin a specific build
cpt-mem0 update --rollback      # return to the previous build
```

Safety rules:
- The new build is downloaded and unpacked **beside** the current one, never
  over it.
- The daemon is stopped, the symlink/pointer is switched, the daemon is
  started, and a health check runs.
- On failure the previous build is restored automatically and the daemon is
  restarted with it.
- The previous build is retained for rollback.
- **Model files and the vector store are never touched by an update.**
  (The store is only affected if the embedder model changes — the updater
  checks this and warns.)

---

## 8. Directory layout

XDG-based on Linux/macOS, `%LOCALAPPDATA%`-based on Windows.

```
~/.local/share/cpt-mem0/            (Windows: %LOCALAPPDATA%\cpt-mem0)
├── config.yaml
├── models/
│   ├── catalogue.json              # model metadata + calibration cache
│   ├── llm/
│   │   └── Qwen3.5-9B-Q4_K_M.gguf
│   └── embedders/
│       └── nomic-embed-text-v1.5.Q4_K_M.gguf
├── runtime/
│   └── llama/
│       ├── b11838/                 # extracted build
│       └── current -> b11838       # pointer
├── logs/
│   ├── daemon.log
│   ├── llama-llm.log
│   └── llama-embed.log
└── keys/
    └── clients.json                # per-client API keys (0600)

~/.config/cpt-mem0/                 (Windows: %APPDATA%\cpt-mem0)
└── config.yaml
```

State that must survive an upgrade lives outside the runtime directory.

---

## 9. The context floor — a service that cannot store memories

This emerged from planning the low-VRAM tiers and is the single most
important constraint in the document.

### 9.1 The measurement

mem0's fact-extraction prompt, measured on this machine:

```
prompt eval time = 6110.85 ms / 10449 tokens (1709.91 tokens per second)
n_gen = 150, tg = 49.66 t/s
```

The extraction pass needs a **single slot with ~10 500 tokens**. The
instruction text alone (fact extraction rules, existing-memory
instructions, output format specification) is several thousand tokens
before any conversation content is added.

### 9.2 The failure mode

A configuration with `n_ctx = 4096` — which `--fit` produces by default
when the model does not fit — behaves like this:

1. The model loads. VRAM is allocated. The port opens.
2. `/v1/models` answers 200.
3. Chat requests work.
4. `mem0.add()` sends the extraction prompt. The prompt is **truncated**.
5. The model returns malformed JSON, or facts extracted from a fragment.
6. Either the write fails, or — worse — **wrong facts are stored**.

Nothing reports an error. `status` says healthy. The user's memory quietly
fills with garbage or stays empty. This is the exact failure the product
exists to prevent, one level above the `--fit` problem.

### 9.3 The floors

```
CONTEXT_FLOOR_TOTAL = 21 000     # 2 slots × 10 500
CONTEXT_FLOOR_SLOT  = 12 288     # extraction slot with headroom
```

These are **hard constraints** in the planner, checked alongside the VRAM
budget. A candidate configuration that fits in VRAM but violates the floor
is rejected and the cascade continues.

### 9.4 The consequence: the card tiers

With the floors enforced, the viable card tiers are:

| Card | Viable | Notes |
|---|---|---|
| 12 GB | ✅ Qwen3.5-9B Q8_0, n_ctx 62 976 | comfortable |
| 8 GB | ✅ Qwen3.5-9B Q5_K_M, n_ctx 17 920 | |
| 6 GB | ✅ Qwen3.5-9B IQ3_XXS, n_ctx 39 168 | coarse quantisation |
| 5 GB | ✅ Qwen3-4B-Instruct-2507 Q5_K_M, n_ctx 13 056 | |
| 4 GB | ✅ Qwen3-4B-Instruct-2507 IQ3_XXS, n_ctx 15 104 | fits, but at `IQ3_XXS` |

At 4 GB the smallest configuration that reaches the context floor needs:

| Item | MiB |
|---|---|
| Qwen3.5-4B IQ3_XXS weights | 1654 |
| KV cache, 12 288 tokens | 108 |
| Process overhead (compute + CUDA context + embedder) | 990 |
| Reserve + headroom (10 % + 6 %) | 654 |
| **Total** | **3406** |
| **Available** | **4096** |
| **Shortfall** | **0** |

The floor is still real: a configuration below the context floor is
rejected, and the planner reports how many MiB are missing. The 4 GB
card fits at a coarse quantisation; whether that quantisation extracts
valid facts is the open measurement of §17.4.

### 9.5 Why two models do not solve this

The obvious idea — a small model for extraction, a large one for chat —
was evaluated and rejected on measurement.

**Cost of a second `llama-server` process:**

| Overhead | MiB | Source |
|---|---|---|
| CUDA context | ~320 | measured (188 MiB observed delta + process baseline) |
| Compute buffer | ~392 | server log |
| Recurrent state (hybrid) | ~100 | server log |
| **Total** | **~812** | |

**Resulting configurations (KV `q8_0`, 2 slots):**

| Card | 9B chat + 4B extraction | 9B chat alone |
|---|---|---|
| 12 GB | 58 112 (58 112/slot) | **62 976 (62 976/slot)** |
| 8 GB | does not fit | **17 920 (17 920/slot)** |
| 6 GB | does not fit | **39 168 (39 168/slot)** |
| 5 GB | does not fit | does not fit |
| 4 GB | does not fit | does not fit |

Splitting eliminates the 8 GB and 6 GB tiers entirely: a second
llama-server has to fit on the card as well. The quality argument for splitting is weak:
a 4B extraction model is documented to be within ~6 % of a 9B for fact
extraction, and the mem0-ollama-hybrid project rates the 3–4B range as the
sweet spot.

**Decision: one LLM serves both chat and extraction.** A split remains
possible for users with 24 GB+ who want a large chat model plus a
dedicated fast extractor, but it is never the default and never required.

### 9.6 What the tool says when it cannot fit

The planner does not silently degrade. It reports the binding constraint
and the alternatives:

```
No viable configuration for 4096 MiB of VRAM.

The smallest model that can extract facts (Qwen3.5-4B IQ3_XXS) needs:
  weights              1654 MiB
  KV cache (12288 tok)  108 MiB
  process overhead      990 MiB
  reserve + headroom    654 MiB
  --------------------------------
  total               3406 MiB

Your card is 0 MiB short.

Options:
  - use a card with 5 GB or more VRAM
  - set planner.context_floor_total lower, if your mem0 configuration
    uses a shorter extraction prompt
  - set planner.allow_degraded: true to start anyway, accepting that
    fact extraction may fail
```

---

## 10. Configuration

```yaml
# config.yaml
version: 1

identity:
  user_id: "${USER}"          # default: OS user; override for shared installs
  agent_id: "cpt-mem0"

server:
  host: "127.0.0.1"           # localhost only by default
  port: 8765
  api_keys:
    - name: "local"
      key: ""                 # generated on first run if empty

llm:
  model: null                 # null = let the planner choose
  quantization: null          # null = let the planner choose
  context: null               # null = planner fills the remaining budget
  parallel: 2
  kv_cache_type: "q8_0"       # f16 | q8_0 | q4_0
  temperature: 0.0
  max_tokens: 2000

embedder:
  backend: "auto"             # auto | llamacpp | fastembed
  model: "nomic-ai/nomic-embed-text-v1.5"
  dimensions: 768

store:
  provider: "qdrant"
  path: "~/.local/share/cpt-mem0/store"
  collection: "mem0"

llama_cpp:
  build: "b11838"             # pinned b-build, never "latest"
  variant: "auto"             # auto | cudart | plain
  gpu_layers: 99

planner:
  reserve_mib: null           # null = 10 % of total VRAM
  headroom_mib: null          # null = 6 % of total VRAM
  context_floor_total: 21000  # min n_ctx for the service to extract facts
  context_floor_slot: 12288   # min per-slot context for extraction
  kv_cache_type_order: [q8_0, f16, q4_0]
  allow_degraded: false       # true = permit sub-floor configurations (see SPEC 17.3)
  calibration: "auto"         # auto | always | never

update:
  check_on_start: false       # P5: never update silently
```

Every field has a working default. A fresh install runs with an empty
config.

---

## 11. CLI

```
cpt-mem0 init                 # interactive setup: detect GPU, fetch binary, pick model
cpt-mem0 plan                 # show the computed configuration, start nothing
cpt-mem0 start                # start the daemon
cpt-mem0 stop
cpt-mem0 status               # processes, VRAM, store size, uptime
cpt-mem0 update [--check]     # binary update
cpt-mem0 calibrate            # re-derive KV constants for the active model
cpt-mem0 models list          # catalogue with sizes and suitability
cpt-mem0 models add <model>   # download a GGUF
cpt-mem0 migrate --from pgvector
```

`plan` output example:

```
GPU     RTX 3060                          12288 MiB total
LLM     Qwen3.5-9B Q8_0                   9086 MiB
        weights 8087 MiB  KV 1114 MiB (17408 B/tok, q8_0)
        n_ctx 62976  (62976 per slot, 2 slots)
        --fit off  --flash-attn on  --reasoning-format none  --kv-unified
        recurrent state     100 MiB (hybrid model, fixed)
        ranking basis: unmeasured
EMBED   llamacpp                          on GPU
TOTAL   10351 MiB used  +  1200 reserve  +  737 headroom  =  12288 / 12288
```

---

## 12. API surface

Full contracts, error envelopes, exit codes and the MCP tool list are in
**[docs/api.md](docs/api.md)**. Summary:

REST on `127.0.0.1:8765`, plus an MCP server on the same port.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/memories` | add facts from a conversation turn |
| `POST` | `/v1/memories/search` | semantic search |
| `GET` | `/v1/memories` | list memories (`?user_id=`) |
| `PATCH` | `/v1/memories/{id}` | update |
| `DELETE` | `/v1/memories/{id}` | delete |
| `GET` | `/v1/health` | liveness + GPU state |
| `GET` | `/v1/status` | full runtime status |
| `GET` | `/v1/plan` | the planner result for this hardware |
| `GET` | `/v1/chat/completions` | OpenAI-compatible passthrough to the local LLM |

Authentication: per-client API keys in `keys/clients.json`. The daemon
generates one on first run. Keys are never written to logs.

Multi-tenancy: `user_id` is a request parameter. A client may be scoped to
a single `user_id`; the daemon enforces the scope. This is a mem0-native
concept, not an agent-framework feature.

### 12.1 MCP tools

**Raw** — `memory_add`, `memory_search`, `memory_list`, `memory_update`,
`memory_delete`.

**Convenience** — `remember` (one statement → extracted fact) and `recall`
(search → formatted bullet list ready to inject into a prompt).

Both layers exist because they serve different callers: an agent that
wants to paste context into a prompt uses `recall`; an application that
needs structured data uses the raw tools.

### 12.2 Compatibility surface

The mem0 engine is wrapped, not exposed. The daemon translates to whatever
mem0's API requires — including version differences such as
`filters={"user_id": ...}` versus a top-level `user_id` parameter — so a
mem0 upgrade cannot break a client's contract.

---

## 13. Health checks and failure handling

| Condition | Detection | Response |
|---|---|---|
| Model fell back to CPU | VRAM below threshold while the server is up | Warn loudly in `status`; re-run the planner |
| `llama-server` crashed | Process exit | Restart with exponential backoff (max 5 tries) |
| Store locked by another process | `RuntimeError` on open | Fail with a clear message naming the lock |
| Plan does not fit any model | Planner exhausts the cascade | Report the binding constraint and suggest a smaller card requirement |
| GPU removed / driver reset | `nvidia-smi` fails | Stop child processes, wait, retry |
| Disk full during model download | Write error | Clean partial file, report |

The VRAM threshold for "model is on the GPU" is a platform constant, not a
guess. It is measured once during `init` on the actual machine.

---

## 14. Verification data

Every number in this specification was measured, not estimated. The full
record — logs, derivations, method and caveats — is in
**[docs/verification.md](docs/verification.md)**.

Summary of the test environment: RTX 3060 12 GB, llama.cpp build `b11798`,
mem0 2.2.1, qdrant-client 1.19/1.20, Windows 11.

| Measurement | Value | Source |
|---|---|---|
| KV bytes/token, Qwen3.5-9B | 32 768 (f16) | 8 server logs, linear fit, exact |
| KV bytes/token, nomic-embed | floor, not proportional | 12 MiB at n_ctx 2048, identical for 2 different models |
| Model buffer, Q8_0 | 8045 MiB GPU / 1031 MiB host | server log |
| Model buffer, Q4_K_M | 4861 MiB GPU / 546 MiB host | server log |
| Embed latency GPU | 4.2 ms/fact | benchmark |
| Embed latency CPU | 21.3 ms/fact | benchmark |
| Embed batch (6) | 94 ms GPU / 138 ms CPU | benchmark |
| Embedder VRAM delta | 188 MiB GPU / 118 MiB CPU | nvidia-smi |
| Cross-backend store compat | identical top-1, 3/3 queries | mem0 + Qdrant test |
| mem0 + Qdrant embedded | add / search / get_all all pass | end-to-end test |
| Store size | 28.6 KiB for 4 memories | filesystem |
| llama.cpp b-build assets | 36 per build, permanent | GitHub releases API |

---

## 15. Implementation phases

| Phase | Scope | Estimate |
|---|---|---|
| 1 | Planner + GGUF reader + calibration (pure logic, testable) | 1 day |
| 2 | Binary provisioning + update + platform detection | 1 day |
| 3 | Daemon: process supervision, mem0 engine, Qdrant store | 1.5 days |
| 4 | REST + MCP API, key management | 1 day |
| 5 | CLI, init flow, status | 1 day |
| 6 | Cross-platform hardening (Windows/Linux/macOS), docs | 1.5 days |

Phase 1 has no dependencies on the rest and can be validated independently
against the data in §13.

---

## 16. Licensing and third-party components

The project is MIT licensed (`LICENSE`).

### 16.1 Runtime dependencies

| Component | License | Implication |
|---|---|---|
| `mem0ai/mem0` | Apache-2.0 | Permissive; NOTICE/attribution applies when redistributed |
| `qdrant-client` | Apache-2.0 | Permissive; NOTICE/attribution applies when redistributed |
| `ggml-org/llama.cpp` | MIT | Permissive; attribution required in distribution |
| `fastembed` (Qdrant) | Apache-2.0 | Permissive |
| Model weights (e.g. `nomic-embed-text-v1.5`, Qwen3.5) | Apache-2.0 | Per-model license; surfaced at download time |

MIT as the project license accepts all of these as inputs.

### 16.2 Obligations when shipping a bundled environment

The app builds and ships its own Python environment, which means it
redistributes Apache-2.0 licensed packages. Required at packaging time:

- Include the Apache-2.0 license text for each redistributed package.
- Carry over any `NOTICE` file present in the package.
- List the third-party components and their licenses in the distribution
  (target: an auto-generated `THIRD_PARTY_LICENSES.md` produced during
  packaging, not maintained by hand).

Model weights are **downloaded at runtime, never redistributed** by the
app. Their licenses are surfaced when a model is added, and the catalogue
records the license per model.

### 16.3 What the app downloads at runtime

| Artifact | Source | License |
|---|---|---|
| `llama.cpp` binary | GitHub releases (pinned b-build) | MIT |
| CUDA runtime | bundled in the `cudart-` archive | NVIDIA EULA |
| GGUF models | Hugging Face | per model |
| ONNX embedder models | Hugging Face | per model |

None of these are vendored into the source repository.

---

## 17. Open questions

### 17.1 Resolved

| # | Question | Resolution |
|---|---|---|
| 1 | Model catalogue scope | **Curated list + on-demand extension.** The catalogue ships in `catalogue.json` (never fetched from the network — a tool that needs internet to start is not self-contained). `cpt-mem0 models add` resolves any GGUF from Hugging Face at runtime and stores it in a local overlay. See **[docs/catalogue.md](docs/catalogue.md)**. |
| 2 | MCP tool surface | **Both.** Raw CRUD plus `remember` and `recall` as convenience tools. See §12.1 and `docs/api.md`. |
| 3 | Nomic task prefixes | **No.** Measured over 46 queries in 5 languages: no significant difference, all 95 % CIs include zero. Adding them would create a store-compatibility hazard for an unproven gain. See §5.4. |
| 4 | Multi-process store access | **Not needed.** Qdrant embedded forbids it (§3.1); the daemon owns the store and serves all clients. A Qdrant server mode remains an escape hatch, not a feature. |
| 5 | The 4–5 GB tier | **Refuse by default, allow explicit opt-in.** The planner rejects any configuration below the context floor and reports the shortfall. `planner.allow_degraded: true` overrides it with a loud warning in `plan`, in `status` and in the daemon log. The refusal is the default because a service that silently stores nothing is worse than one that does not start. See §17.3. |
| 6 | Chat vs. extraction weighting | **No separate weighting. One ranking, and it is not parameter count.** Measured extraction quality (§17.4) inverts the assumed order: the 2B model extracts better than the 9B. The catalogue therefore ranks by **measured extraction capability**, with chat quality as a tiebreaker. |
| 7 | Context above the floor | **Chat buffer, not memory capacity.** The extraction prompt is 10 449 tokens; a slot holds it from 12 288 tokens upwards. Everything above that serves the chat side, and the extraction quality depends on the weights, not on the room left over. So when two quantisations of the same model are viable, the better one wins even if the coarser one would give three times the context - both use the same VRAM. See §4.13. |

### 17.2 Still open

**7. Reranking.** The catalogue currently relies on the embedder's own
similarity ranking. A cross-encoder reranker would improve Top-1 on
larger corpora, at the cost of another model (~100–500 MiB VRAM or CPU
time) and ~10–30 ms per query. Given that the VRAM budget is already
tight, a CPU-based reranker might be the better trade — but only if
measurements show a meaningful gain on a realistic corpus. Deferred
until the service runs end to end.

**8. Automatic model re-selection.** If the user later adds a larger
model to the catalogue, should the daemon notice and offer an upgrade?
Recommendation: no automatic switching — the active model is a running
service, changing it under a live client would invalidate context
caches. Offer it as `cpt-mem0 plan --upgrade` instead.

**9. Extraction quality beyond the current corpus.** The measurement in
§17.4 uses 10 short conversation turns. Real usage involves longer
histories, multi-turn context and more varied fact types. The catalogue
ranking should be re-derived from a larger corpus before it is treated as
settled.

### 17.3 The degraded-mode contract

```yaml
planner:
  allow_degraded: false     # true = permit sub-floor configurations
```

When `allow_degraded: false` (the default) and no configuration satisfies
the context floor, `plan` exits with code 3 and prints the binding
constraint (§4.4). The daemon does not start.

When `allow_degraded: true`, the planner selects the best configuration
that fits the VRAM budget, even if it violates the context floor, and:

- `plan` prints a `WARNING` block naming the shortfall.
- `status` reports `extraction_capable: false`.
- The daemon log records `WARN extraction_context_below_floor` on startup.
- `/v1/health` returns `status: degraded` rather than `healthy`.

The degradation is never silent. A user who opts in knows exactly what
they traded away.

### 17.4 Measured extraction capability — and why the ranking changed

The catalogue originally ranked models by parameter count. That assumption
was wrong, and the correction came from measurement.

**Test:** the real `USER_MEMORY_EXTRACTION_PROMPT` from mem0 2.2.1
(4220 characters) with the conversation appended, 10 conversation turns in
German and English, 2 seeds each, 20 runs per model. Scored on JSON
validity, expected-fact recall, atomicity of the produced facts, language
fidelity and hallucinations.

| Model | JSON valid | Empty | Recall | Atomic facts | Language | Halluc. | Time/run |
|---|---|---|---|---|---|---|---|
| **gemma-4-E2B-it** (2B) | 20/20 | 0 | **50/52 = 96 %** | 26/26 | 16/20 | 0 | 358 ms |
| Qwen3.5-9B | 20/20 | **2** | 46/52 = 88 % | 20/22 | 16/20 | 0 | 810 ms |
| Qwen3-4B-Instruct | 20/20 | 0 | 44/52 = 85 % | 24/24 | 8/20 | 0 | 255 ms |

Three findings:

1. **The smallest model extracts best.** gemma-4-E2B reaches 96 % recall
   against the 9B's 88 %, with zero hallucinations and perfect atomicity.
   It is also 2.3× faster.
2. **The 9B fails differently — it returns empty results twice.** An empty
   extraction means nothing is stored, silently. This is the same failure
   class as the context floor (§9): a service that appears healthy while
   its primary function does nothing.
3. **The 4B breaks language fidelity.** Only 8 of 20 answers came back in
   the language of the input, against 16/20 for both others. For an
   international audience this matters — mem0's prompt explicitly asks for
   facts in the input's language.

**Consequences for the design:**

- The catalogue ranking becomes **measured extraction capability**, not
  parameter count. A 2B model that extracts better and uses half the VRAM
  wins on the axis that determines whether the service works at all.
- Chat quality cannot be the deciding factor, because the service's
  primary function is storage. Chat remains available through the
  passthrough endpoint (§12), and users who want a better chat model can
  pin one with `--model`.
- The catalogue therefore gains a **measured** field per model:

```json
{
  "id": "gemma-4-E2B-it",
  "extraction": {
    "measured": true,
    "recall": 0.96,
    "json_validity": 1.0,
    "atomicity": 1.0,
    "language_fidelity": 0.8,
    "empty_rate": 0.0,
    "corpus": "10 turns x 2 seeds, de+en",
    "date": "2026-10-24"
  }
}
```

- Models without a measurement are ranked **below** measured ones, and the
  `plan` output labels the basis. An unmeasured model is a guess; a
  measured one is evidence.

This also resolves the apparent tension in §9.5: the split model idea was
rejected because a second process costs ~812 MiB. With gemma-4-E2B at
2637 MiB of weights, a dedicated extractor on top of a 9B chat model fits
on a 24 GB card for ~9300 MiB — but the measurement gives no reason to do
it, since the 2B alone already extracts best.

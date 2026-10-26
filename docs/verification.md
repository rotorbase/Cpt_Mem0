# Verification Data

Every number in [SPEC.md](../SPEC.md) traces to a measurement recorded here.
No value in the specification is estimated where a measurement was possible.

**Test machine**

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB (12288 MiB) |
| CPU | 8 threads available to llama.cpp |
| llama.cpp | build `b11798` (`0.7.0-dev`), Unsloth build |
| mem0 | 2.2.1 |
| qdrant-client | 1.19.1 (installed) / 1.20.0 (test venv) |
| OS | Windows 11, Git Bash / MSYS |

---

## 1. KV cache size per token

### 1.1 Method

`llama.cpp` writes the allocated VRAM buffer to its log:

```
llama_kv_cache:      CUDA0 KV buffer size =   3328.00 MiB
```

Collecting this line across logs with different `n_ctx` values and fitting a
line gives bytes/token directly. 14 logs contained usable measurements.

### 1.2 Qwen3.5-9B-Q4_K_M — 8 measurements

| `n_ctx` | KV (MiB) | Model buffer (MiB) | Compute (MiB) |
|---|---|---|---|
| 102 912 | 3216.0 | 4861.28 | 378.81 |
| 105 472 | 3296.0 | 4861.28 | 381.31 |
| 105 472 | 3296.0 | 4861.28 | 381.31 |
| 105 728 | 3304.0 | 4861.28 | 381.56 |
| 106 496 | 3328.0 | 4861.28 | 382.31 |
| 106 496 | 3328.0 | 4861.28 | 382.31 |
| 108 032 | 3376.0 | 4861.28 | 383.81 |
| 109 568 | 3424.0 | 4861.28 | 385.31 |

**Linear fit: 32 768 bytes/token (f16), zero intercept.** All eight points
lie exactly on the line.

### 1.3 Qwen3.5-9B-Q8_0 — 3 measurements

| `n_ctx` | KV (MiB) | Model buffer (MiB) | Compute (MiB) |
|---|---|---|---|
| 8192 | 256.0 | 8045.05 | 286.31 |
| 8192 | 256.0 | 8045.05 | 286.31 |
| 8192 | 256.0 | 8045.05 | 286.31 |

8192 × 32 768 bytes = 256 MiB exactly. Consistent with §1.2.

### 1.4 Why the naive formula fails

GGUF metadata for `Qwen3.5-9B`:

```
qwen35.architecture              = qwen35
qwen35.block_count               = 32
qwen35.attention.head_count      = 16
qwen35.attention.head_count_kv   = 4
qwen35.attention.key_length      = 256
qwen35.attention.value_length    = 256
qwen35.full_attention_interval   = 4        <-- hybrid: SSM layers
```

Naive formula (all 32 layers carry KV):

```
32 × (4×256 + 4×256) × 2 bytes = 131 072 bytes/token
```

Measured: **32 768 bytes/token** — exactly ¼.

Corrected formula with `kv_layers = block_count ÷ full_attention_interval`:

```
(32 ÷ 4) × (4×256 + 4×256) × 2 bytes = 32 768 bytes/token   ✓
```

Qwen3.5 is a hybrid architecture: only every fourth layer has full
attention, the rest are Mamba/SSM layers with constant-size recurrent state.

### 1.5 Measured geometry of every catalogue model

The KV constant of each model follows from its GGUF header. All values
below were read with the tool's own reader (`src/cpt_mem0/gguf.py`),
either from a local file or through an HTTP range request against the
HuggingFace repository — no model was downloaded and no GPU was involved.

| Model | arch | `block_count` | `full_attention_interval` | `kv_layers` | `head_count_kv` | key/value length | **KV f16 (B/token)** |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | `qwen35` | 32 | 4 | 8 | 4 | 256 / 256 | **32 768** |
| Qwen3.5-4B | `qwen35` | 32 | 4 | 8 | 4 | 256 / 256 | **32 768** |
| Qwen3-8B | `qwen3` | 36 | 1 | 36 | 8 | 128 / 128 | **147 456** |
| Qwen3-4B-Instruct-2507 | `qwen3` | 36 | 1 | 36 | 8 | 128 / 128 | **147 456** |
| Llama-3.1-8B-Instruct | `llama` | 32 | 1 | 32 | 8 | 128 / 128 | **131 072** |
| gemma-4-E2B-it | `gemma4` | 35 | 1 | 35 | 1 | 512 / 512 | *(not reliable)* |

`gemma-4-E2B-it` carries `attention.sliding_window = 512` and
`attention.shared_kv_layers = 20`, so the closed form does not apply.
The catalogue marks it `calibrate` instead of guessing, and the planner
reports it as needing calibration.

### 1.6 Quantised KV cache sizes

The `q8_0` and `q4_0` factors are llama.cpp's block layouts, read from
`ggml/src/ggml-common.h`:

| Type | Block | Bytes per 32 values | Bytes per value | Share of f16 |
|---|---|---|---|---|
| `f16` | 2 B half | 64 | 2.0 | 100 % |
| `q8_0` | 2 B half scale + 32 B int8 | 34 | 1.0625 | 53.125 % |
| `q4_0` | 2 B half scale + 16 B nibbles | 18 | 0.5625 | 28.125 % |

`llama_kv_cache` allocates K and V as `ggml_new_tensor_3d(type,
n_embd_gqa, kv_size, n_stream)`; `ggml_row_size` then gives
`type_size × ne ÷ blck_size`, with no further padding for
`n_embd_gqa` divisible by 32.

> **Correction (2026-10-24).** The factor for `q8_0` was previously
> estimated at 1.125 bytes per value — 5.9 % too pessimistic, which made
> every context figure in §9.2 too small. The correct value is 1.0625.

### 1.7 BERT architectures do not follow the per-layer formula

`nomic-embed-text-v1.5` (nomic-bert, 12 layers) and `gemma-4-E2B-it`
(gemma4, 35 layers, sliding window) both report:

```
CUDA0 KV buffer size = 12.00 MiB   at n_ctx 2048
```

Two structurally unrelated models, identical allocation. This is a
**floor**, not a slope — the per-layer formula does not apply. Such models
must be calibrated, not computed.

---

## 2. VRAM decomposition

Measured on the running stack (2026-10-23):

| Component | MiB | Source |
|---|---|---|
| Qwen3.5-9B-Q4_K_M GPU buffer | 4861 | server log |
| Qwen3.5-9B-Q4_K_M host buffer | 546 | server log |
| Qwen3.5-9B-Q8_0 GPU buffer | 8045 | server log |
| Qwen3.5-9B-Q8_0 host buffer | 1031 | server log |
| Qwen3.5-9B compute buffer | 286–385 | server log |
| nomic-embed GPU buffer | 78 | server log |
| nomic-embed KV + compute | 12 + 122 | server log |
| **Total measured (both servers + desktop)** | **7910** | `nvidia-smi` |

Host fraction: 10 % (Q4_K_M) and 11 % (Q8_0) of the file size. The planner
uses **11 %** as the conservative constant.

Per-process CUDA context overhead is measured at ~320 MiB. This is the cost
of a second `llama-server` — relevant when deciding whether the embedder
belongs on the GPU.

---

## 3. Embedder: GPU vs CPU

Same model (`nomic-embed-text-v1.5.Q4_K_M`), same texts, realistic mem0
fact lengths (30–60 tokens).

| | GPU (`-ngl 99`) | CPU (`-ngl 0`) |
|---|---|---|
| 1 fact | 3.9 ms | 17.6 ms |
| 1 fact (repeat) | 3.7 ms | 17.1 ms |
| 6 facts (batch) | 94.2 ms | 97.3 ms |
| VRAM delta | 188 MiB | 118 MiB |

The CPU penalty nearly vanishes for batches. Neither is the bottleneck:
mem0's fact extraction takes 5–8 s of LLM time per write.

A llama.cpp server on CPU still costs 118 MiB of VRAM (CUDA context only).
The **fastembed** path (ONNX, in-process) costs **0 MiB** — no server, no
CUDA context at all.

---

## 4. Embedder backends produce compatible stores

Store built with the `llamacpp` embedder, queried with `fastembed`:

| Query | llama.cpp | fastembed | Top-1 result |
|---|---|---|---|
| "welche grafikkarte nutze ich" | 0.4869 | 0.4131 | identical |
| "womit laeuft der store" | 0.4958 | 0.4403 | identical |
| "was entwickelt das projekt" | 0.5766 | 0.5241 | identical |

Why it works: the raw vectors differ in norm.

| | Vector norm |
|---|---|
| llama.cpp | \|v\| = 1.0000 (normalised) |
| fastembed | \|v\| = 21.4003 (not normalised) |

mem0 and Qdrant normalise internally, so the norm difference cancels.
Cosine between the two backends for the same text: **0.92** (different
quantisation: Q4_K_M vs FP32 ONNX), which shifts scores by ~7 % but does not
change the ranking.

**Conclusion: the embedder backend can be switched at runtime without
rebuilding the store.**

---

## 5. Cross-language retrieval quality

Corpus: 8 documents, 4 German and 4 English. Queries in German, English and
French. Metric: does the correct document land on position 1?

| Model | dim | Size | Top-1 | Margin | ms/query |
|---|---|---|---|---|---|
| nomic-ai/nomic-embed-text-v1.5 | 768 | 0.520 GB | 3/4 | **0.1124** | 17.0 |
| BAAI/bge-small-en-v1.5 | 384 | 0.067 GB | 3/4 | 0.0519 | **3.2** |
| jinaai/jina-embeddings-v2-base-de | 768 | 0.640 GB | 2/4 | 0.0269 | 15.8 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.220 GB | 1/4 | 0.0384 | 3.7 |

`nomic-embed-text-v1.5` has roughly double the separation margin of the
runner-up and matches the llama.cpp backend's dimensionality (768), which is
why it is the default.

Not available in fastembed at the time of measurement:
`intfloat/multilingual-e5-small`.

### 5.1 Nomic task prefixes

Measured cosine between prefixed query and prefixed document: **0.9359**
(not 1.0). The prefixes change the embedding space, so omitting them costs
retrieval quality. Neither backend applies them today.

---

## 6. mem0 with Qdrant embedded

mem0 2.2.1, `vector_store.provider = qdrant`, `path = <dir>`,
`embedding_model_dims = 768`:

```
Memory.from_config: OK
add(): OK -> 2 Ergebnisse
  [ADD] User utilizes an NVIDIA RTX 3060 graphics card with 12 GB of VRAM...
  [ADD] User configured their Vector-Store to run in embedded mode...
search(): OK -> 2 Treffer
get_all(): OK -> 2 Memories
Store-Groesse: 28.6 KiB (4 Memories)
```

A second test with the `fastembed` provider, the same llama.cpp LLM and
Qdrant embedded:

```
add(): OK in 6.9s -> 4 Fakten
search("welche grafikkarte nutze ich"):    30 ms  -> RTX 3060 (correct)
search("womit laeuft der store"):          31 ms  -> Qdrant embedded (correct)
search("which gpu does the user have"):    30 ms  -> RTX 3060 (correct)
```

### 6.1 Qdrant embedded locks its storage directory

```
RuntimeError: Storage folder ... is already accessed by another instance of
Qdrant client. If you require concurrent access, use Qdrant server instead.
```

Reproduced with two `Memory` instances on the same path. The lock is
filesystem-level (`portalocker`). A second client on a *different* path for
the same data does not see the first client's collection.

**Design consequence:** one daemon process owns the store; clients use the
API.

---

## 7. llama.cpp release assets

`https://api.github.com/repos/ggml-org/llama.cpp/releases`

### 7.1 Two kinds of release

| Kind | Tag | Assets | Frequency |
|---|---|---|---|
| Tagged release | `v0.7.0` | **1** (`nightly-tag.txt`) | per release |
| Rolling build | `b11838` | **36** | daily |

`v0.7.0`'s only asset is `nightly-tag.txt` containing `b11669` — the
rolling build that was current when the release was cut. **All prebuilt CUDA
binaries live in the b-build releases.**

### 7.2 b-builds are permanent

`b11669` (published 2026-10-15) still serves all 36 assets, including:

```
cudart-llama-bin-win-cuda-12.4-x64.zip              373.3 MB
cudart-llama-bin-win-cuda-13.4-x64.zip              403.9 MB
llama-b11838-bin-win-cuda-13.4-x64.zip              152.5 MB
llama-b11838-bin-ubuntu-x64.tar.gz                   21.3 MB
llama-b11838-bin-macos-arm64.tar.gz                  15.4 MB
```

Pinning an exact b-build is therefore permanently reproducible.

### 7.3 The `cudart-` prefix matters

| Variant | Size (win-x64 CUDA 13.4) | Contents |
|---|---|---|
| `llama-…-bin-win-cuda-13.4-x64.zip` | 152.5 MB | binary only |
| `cudart-llama-…-bin-win-cuda-13.4-x64.zip` | 403.9 MB | binary + bundled CUDA runtime |

When the CUDA runtime DLLs are absent from the binary directory, llama.cpp
falls back to CPU **silently** — the port opens, `/v1/models` answers 200,
inference runs at ~6 tok/s instead of ~47 tok/s. The `cudart-` bundles
remove this failure mode at its source.

---

## 8. Planner self-check

Recomputing the running configuration with the planner:

| Quantity | Predicted | Measured | |
|---|---|---|---|
| KV cache (Q4_K_M, n_ctx 16384, f16) | 512 MiB | 512 MiB | ✓ |
| Model weights (Q4_K_M) | 4821 MiB | 4861 MiB | 0.8 % off |

The weight prediction uses the 11 % host-fraction constant; the residual
difference is that constant's tolerance.

---

## 8a. Hardware verification against a live llama-server

### 8a.1 The run of 2026-10-24

RTX 3060 12 GB, Qwen3.5-9B-Q8_0, KV cache `q8_0`, two slots, on an
otherwise idle card.

Server flags:

```
-c 67122 --parallel 2 -ngl 99 --fit off --flash-attn on
--reasoning-format none --no-context-shift --kv-unified
--cache-type-k q8_0 --cache-type-v q8_0
```

| Quantity | Predicted | Measured | |
|---|---|---|---|
| Model weights | 8 087 MiB | — | — |
| KV cache | 1 114 MiB | — | — |
| Compute buffer | 290 MiB | — | — |
| CUDA context | 320 MiB | — | — |
| Embedder | 540 MiB | — | — |
| **Total VRAM delta** | **9 811 MiB** | **9 775 MiB** | **36 MiB off (0.4 %)** |
| Slot context | 33 561 | 67 328 | ✗ |

The VRAM budget model is accurate: the prediction was within 0.4 % of
the measured delta. The context prediction was not.

### 8a.2 What the run found

**`-c` was planned as the total, but the server used it per slot.**

With `--kv-unified` the KV cache is one shared pool. The `-c` flag is
the limit per slot, and the pool holds that many tokens in total. The
planner had assumed the opposite: it treated `-c` as the total budget
and divided it by the slot count to get the per-slot context.

Measured on the same build, with `-c 67122 --parallel 2`:

| Flags | Server reported `n_ctx_slot` |
|---|---|
| with `--kv-unified` | 67 328 |
| without | 33 792 |

The fix moves the division out of the planner: `-c` is now the per-slot
limit, and the KV pool is budgeted once.

**The server rounds the context up to a multiple of 256.**

The planned 67 122 arrived as 67 328 — 206 tokens more, because the KV
tensors are allocated in blocks of 256. The planner now rounds down to
the same block size, so the printed value is the value the server runs
with, and the VRAM is never exceeded.

**The cascade degraded the cache type too early.**

Walking model → quantisation → cache type reached a configuration that
fits but misses the context floor, while a viable one existed. On an
8 GB card the 9B with a `q4_0` cache reached 8 472 tokens per slot —
below the floor of 12 288 — while the 4B with a `q8_0` cache reached
92 273. The cache type is now the outer loop: every model at `q8_0`
first, then every model at `f16`, then every model at `q4_0`.

### 8a.3 What the run found

**The buffer lines name a device, and the host line comes first.**

The log carries one line per device:

```
load_tensors:   CPU_Mapped model buffer size = 1030.62 MiB
load_tensors:        CUDA0 model buffer size = 8055.38 MiB
llama_kv_cache:        CUDA0 KV buffer size = 1113.50 MiB
sched_reserve:      CUDA0 compute buffer size =   391.79 MiB
```

A pattern that takes the first match finds the `CPU_Mapped` line. The
memory-mapped part of the model is **host RAM**, not VRAM. The first
version of the script did exactly that and reported 1030 MiB of
weights where the card held 8087 MiB.

The two numbers add up exactly: 1030.62 + 8055.38 = 9086.00 MiB, the
size of Qwen3.5-9B-Q8_0. The model was fully loaded all along.

**The plan was 160 MiB too optimistic, and the reason was two missing
terms.**

The measured buffers of the 9B-Q8_0 run:

```
load_tensors:        CUDA0 model buffer size =  8045.05 MiB
llama_kv_cache:      CUDA0 KV buffer size =  1113.50 MiB
llama_memory_recurrent: CUDA0 RS buffer size =   100.50 MiB
sched_reserve:      CUDA0 compute buffer size =   391.79 MiB
```

| Term | Planner | Measured | |
|---|---|---|---|
| Weights (CUDA0) | 8087 MiB | 8045.05 MiB | ✓ |
| KV cache | 1114 MiB | 1113.50 MiB | ✓ exact |
| Compute buffer | 290 MiB | 391.79 MiB | ✗ too small |
| Recurrent state | — | 100.50 MiB | ✗ missing |
| CUDA context | 320 MiB | — | |
| Embedder | 540 MiB | — | |

The recurrent state is the buffer of the state-space layers. Qwen3.5
is a hybrid architecture: 32 layers, every fourth one carries a KV
cache, the other 24 are state-space layers. Their state lives in its
own buffer, which does not scale with the context and is not affected
by the cache type.

The GGUF header says which models have it - the keys are
`<arch>.ssm.state_size`. Read over an HTTP range request, without
downloading the model:

| Model | Architecture | SSM keys | Recurrent state |
|---|---|---|---|
| Qwen3.5-9B | `qwen35` | yes | 100.50 MiB (measured) |
| Qwen3.5-4B | `qwen35` | yes | not measured yet |
| Qwen3-8B | `qwen3` | no | none |
| Qwen3-4B-Instruct-2507 | `qwen3` | no | none |
| Llama-3.1-8B-Instruct | `llama` | no | none |
| gemma-4-E2B-it | `gemma4` | no | none |

The size is **not derivable** from the attention geometry: none of the
combinations of state size (128), inner size (4096), group count (16)
and layer count reproduce the measured 100.50 MiB. It is therefore a
measured catalogue value, and a hybrid model without one is not
plannable - the planner asks for a calibration run instead of
guessing.

**The compute buffer grows with the context, and it is a line.**

Three measurements of the same model at three contexts:

| `n_ctx` | compute buffer |
|---|---|
| 54 784 | 331.79 MiB |
| 61 696 | 365.54 MiB |
| 67 072 | 391.79 MiB |

The three points lie on a straight line:

    compute = 0.00488281 * n_ctx + 64.29

It reproduces every measurement to within 0.01 MiB. The planner uses
the formula, which means the context and the buffer have to be solved
together - the buffer depends on the context, and the context depends
on the buffer. The planner iterates; it settles after three or four
rounds.

**The 4 GB tier was rejected because of this.** With the buffer planned
as a constant of 392 MiB - a value measured at a context of 67 072 -
the 4 GB card missed the context floor by 253 MiB and was refused. With
the formula it needs 130 MiB at its own context, and the tier works:
15 104 tokens per slot, 4078 of 4096 MiB used.

**The embedder has no KV cache.** The nomic embedder is a pure
encoder; llama.cpp allocates no KV state for it. The plan counted
12 MiB that do not exist.

**One VRAM reading is a moment, not the allocation.**

The measurement took a single `nvidia-smi` sample while the server was
still allocating and reported a delta of 3804 MiB. The card actually
held about 9560 MiB. The measurement now samples repeatedly and keeps
the highest value.

**The script never wrote a log.**

The numbers were printed and then lost. The next run could not be
compared against the previous one, and the first run with a wrong
figure could not be diagnosed. The server log now goes to
`results/server.log`.

### 8a.4 The re-check

The same checks, after the fix, with the 4B model on the GPU:

```
  n_ctx_slot         16384 (the -c value, per slot)
  layers to GPU      37 / 37
  model buffer       2375.91 MiB (CUDA0)
  model buffer       304.28 MiB (CPU_Mapped, host RAM - not VRAM)
  compute buffer     116.09 MiB
  KV buffer          1224.00 MiB
  GPU usage delta    3830 MiB (7175 -> 11005, measured while the server ran)

KV delta: 1224 measured vs 1224 predicted (0 MiB)
weights delta: 2376 measured vs 2453 predicted (77 MiB)
n_ctx_slot matches the plan (16384)
inference: the server generated a response

The plan matches the measurement.
```

| Quantity | Predicted | Measured | |
|---|---|---|---|
| KV cache | 1224 MiB | 1224 MiB | ✓ exact |
| Weights (CUDA0) | 2453 MiB | 2376 MiB | 3 % off |
| Slot context | 16384 | 16384 | ✓ exact |
| Layers to the GPU | 37 / 37 | 37 / 37 | ✓ |
| Inference | — | response received | ✓ |

The run also reports how many layers reached the GPU. A partial
offload means the card did not have room for the planned configuration,
and the budget cannot be verified in that case — the script says so
instead of producing a misleading comparison.

---

## 9. Context floor

### 9.1 Extraction prompt size

Measured on the running stack during a `mem0_add`:

```
prompt eval time = 6110.85 ms / 10449 tokens (1709.91 tokens per second)
n_gen = 150, tg = 49.66 t/s, tg_3s = 49.99 t/s
```

The fact-extraction pass requires a slot with ~10 500 tokens. This is the
basis for `CONTEXT_FLOOR_SLOT = 12288` and `CONTEXT_FLOOR_TOTAL = 21000`
in the specification.

### 9.2 Card tiers with the floor enforced

**Regenerated from the planner** (`scripts/regenerate_tiers.py`), so these
numbers come out of the shipped code rather than a scratch script.

Cascade order per SPEC section 4.4: best model that fits, then best
quantisation, then best KV cache type. Embedder on GPU, 2 slots:

| VRAM | Configuration | `n_ctx` | Per slot |
|---|---|---|---|
| 12 GB | Qwen3.5-9B-Q8_0 | 62 976 | 62 976 |
| 8 GB | Qwen3.5-9B-Q5_K_M | 17 920 | 17 920 |
| 6 GB | Qwen3.5-9B-IQ3_XXS | 39 168 | 39 168 |
| 5 GB | Qwen3-4B-Instruct-2507-Q5_K_M | 13 056 | 13 056 |
| 4 GB | Qwen3-4B-Instruct-2507-IQ3_XXS | 15 104 | 15 104 |

The hierarchy is monotonic: more VRAM yields a better model, not merely
a larger context. `tests/test_planner.py::TestCardHierarchy` asserts it.

> **Correction (2026-10-24).** The earlier version of this table was
> computed with a scratch script and listed a different configuration per
> card. Two things were wrong with it:
>
> 1. The `q8_0` KV factor was estimated at 1.125 bytes per value. The
>    real value is 1.0625 (34 bytes per 32 values, read from
>    `ggml/src/ggml-common.h`), so every context figure was ~5.9 % too
>    small.
> 2. The script did not consider that a model which does not fit on the
>    card cannot be its choice at all. Without that rule the 4 GB card
>    received the same small model as the 12 GB card.
>
> The table above is the first one produced by the shipped planner.

### 9.3 Two-model comparison

Same budget model, KV `q8_0`, 2 slots:

| VRAM | 9B + 4B (split) | 9B alone |
|---|---|---|
| 12 GB | 93 862 (46 931/slot) | 249 146 (124 573/slot) |
| 8 GB | does not fit (10 011 MiB needed) | 51 798 (25 899/slot) |
| 6 GB | does not fit (9 683 MiB needed) | does not fit |

The second process costs ~812 MiB (320 CUDA context + 392 compute buffer
+ 100 recurrent state for a hybrid model),
which is more than the entire KV cache of the 9B at a usable context size.

### 9.4 Minimum configuration breakdown (4 GB)

The 4 GB card is viable with a 4B model at `IQ3_XXS`, because the
weights are small enough to leave room for the KV cache:

| Item | MiB |
|---|---|
| Qwen3.5-4B IQ3_XXS weights | 1654 |
| KV cache, 12 288 tokens | 108 |
| Process overhead | 990 |
| Reserve (10 %) + headroom (6 %) | 654 |
| **Total required** | **3406** |
| **Available** | **4096** |
| **Shortfall** | **0** |

With the GPU embedder removed (CPU embedder, 540 MiB saved) a 4B model
at `Q4_K_M` also fits, at the cost of a smaller context.

`IQ3_XXS` is a coarse quantisation. Whether it extracts facts well
enough is an open measurement (§17.4) — the planner does not yet know,
and it does not pretend to.

---

## 10. Task prefixes (§5.4 of the specification)

### 10.1 Test setup

30 realistic memory facts, 46 queries across 5 languages (en, de, fr, es,
it). Paired comparison: every query embedded with and without the prefix,
the rank difference evaluated per query, bootstrap confidence interval
over 10 000 resamples.

### 10.2 Results

| Configuration | Top-1 | MRR | Margin |
|---|---|---|---|
| Without prefix | 31/46 | 0.7675 | 0.1008 |
| Doc + query prefix | 33/46 | 0.7798 | 0.0844 |
| Query prefix only | 29/46 | 0.7457 | 0.0937 |
| Doc prefix only | 34/46 | 0.7934 | 0.0876 |

Paired Δ MRR (prefix − no prefix), 95 % CI:

| Configuration | Δ MRR | 95 % CI | Affected queries |
|---|---|---|---|
| Doc + query | +0.0123 | [−0.0200, +0.0501] | 12/46 |
| Query only | −0.0219 | [−0.0566, +0.0051] | 9/46 |
| Doc only | +0.0259 | [−0.0090, +0.0682] | 12/46 |

**All three confidence intervals include zero.** No configuration is
distinguishable from no prefix.

### 10.3 Control

A deliberate mismatch — documents prefixed, queries not — scores 34/46,
essentially the same as the consistent prefixed case (33/46) and slightly
*better* than the unprefixed baseline (31/46). The prefix neither helps
nor breaks retrieval, so the test is measuring a real (if absent) effect
rather than a broken setup.

### 10.4 Conclusion

No prefix support is implemented. The measured benefit is absent, and
prefix support would introduce a store-compatibility hazard: a store
written by a prefixed build cannot be read by an unprefixed one.

### 10.5 The mem0 hook

mem0 2.2.1 already threads the action through the embedder API, so the
mechanism exists if a future embedder family needs it:

```
mem0/memory/main.py:927   embed(parsed_messages, "search")
mem0/memory/main.py:1980  embed(data, memory_action="add")
mem0/memory/main.py:689   embed(entity_text, "update")
```

---

## 11. Catalogue file sizes

Read from the Hugging Face API (`?blobs=true`) on 2026-10-24. VRAM = file
size × 0.89 (measured host fraction).

| Model | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | IQ4_XS | IQ3_XXS |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 9086 | 7113 | 6273 | 5417 | 4929 | 3830 |
| Qwen3-14B | 14971 | 11560 | 10027 | 8585 | 7758 | – |
| Qwen3-8B | 8306 | 6414 | 5580 | 4795 | 4369 | 3252 |
| Llama-3.1-8B | – | 6290 | 5467 | 4693 | 4257 | 3168 |
| gemma-4-E2B-it | 4814 | 4293 | 3201 | 2963 | 2846 | 2263 |
| Qwen3.5-4B | 4275 | 3363 | 2998 | 2614 | 2362 | 1859 |
| Qwen3-4B-Instruct-2507 | 4082 | 3153 | 2756 | 2382 | 2166 | 1597 |

### 11.1 Qwen3-14B — the sizes and the KV constant

The sizes come from the Hugging Face API on 2026-10-26
(`unsloth/Qwen3-14B-GGUF`, `?blobs=true`).

**The KV constant is derived, not measured.** The model card gives
the geometry: 40 layers, 8 KV heads, head_dim 128. The f16 size per
token follows from it:

```
2 (K and V) × 40 layers × 8 KV heads × 128 head_dim × 2 bytes
    = 163 840 bytes per token
```

That is the number the catalogue carries, marked `kv_source:
"formula"`. It has not been measured on this build - a measurement
needs the model in the VRAM, and the card was busy. The models with
a measured constant are the ones in the table above.

**The 14B on a 16 GB card.** The plan for 16 384 MiB picks
Qwen3-14B Q8_0 with 13 568 tokens of context; Qwen3.5-9B Q8_0 fits
on the same card with 250 624 tokens. For mem0 the context matters
as much as the size - the extraction takes the existing memories with
it, and a model with 13 568 tokens cannot hold a long conversation
that a model with 250 624 can. The planner therefore prefers the
measured model by default; `prefer_larger` in the planner config
turns it the other way.

**Resolution pitfall:** `unsloth/gemma-4-E2B-it-GGUF` contains
`MTP/mtp-gemma-4-E2B-it-Q8_0.gguf` (93 MiB) — a speculative-decoding draft
model. A naive "smallest file with this quantisation" search picks the
draft instead of the main model (4814 MiB). Catalogue resolution must
exclude `MTP/` and `mmproj` paths.

The catalogue in the repository carries these sizes as
`size_bytes` per quantisation. `cpt-mem0 validate-catalogue` checks the
catalogue's internal consistency; the sizes themselves are re-read from
the API whenever the catalogue is refreshed.

---

## 12. Extraction capability (SPEC §17.4)

### 12.1 Method

The real `USER_MEMORY_EXTRACTION_PROMPT` from mem0 2.2.1 (4220 characters)
with the conversation appended, run against each model through
`llama-server` with `--reasoning off`, temperature 0.

- 10 conversation turns (5 German, 5 English), 2 seeds each
- 20 runs per model
- Scored on: JSON validity, expected-fact recall, atomicity of the
  produced facts, language fidelity, hallucinations, wall time

The expected output format is `{"facts": [...]}` — a dict, not a bare
array. An earlier test run that looked for a top-level array reported
0 % recall for every model; the models were correct, the parser was wrong.

### 12.2 Results

| Model | JSON valid | Empty | Recall | Atomic facts | Language | Halluc. | Time/run |
|---|---|---|---|---|---|---|---|
| **gemma-4-E2B-it** | 20/20 | 0 | **50/52 = 96 %** | 26/26 | 16/20 | 0 | 358 ms |
| Qwen3.5-9B | 20/20 | **2** | 46/52 = 88 % | 20/22 | 16/20 | 0 | 810 ms |
| Qwen3-4B-Instruct | 20/20 | 0 | 44/52 = 85 % | 24/24 | 8/20 | 0 | 255 ms |

### 12.3 What the models produced

gemma-4-E2B, German input — correct language, atomic facts, no
hallucination:

```
- Nutzt eine NVIDIA RTX 3060 mit 12 GB VRAM für lokale Modelle
- Mem0 läuft komplett lokal, ohne Cloud
- Der Vektorstore ist Qdrant im embedded Modus
```

Qwen3.5-9B, same input — English output for a German question, and one
fact merged into another:

```
- Uses NVIDIA RTX 3060 with 12 GB VRAM for local models
- Mem0 runs completely locally without cloud
- Vectorstore is Qdrant in embedded mode
```

### 12.4 Findings

1. **The smallest model extracts best.** 96 % recall against the 9B's
   88 %, zero hallucinations, perfect atomicity, 2.3× faster.
2. **The 9B has a silent failure mode.** It returned an empty fact list in
   2 of 20 runs. Nothing is stored, nothing errors. Same failure class as
   the context floor: a healthy-looking service that does not work.
3. **The 4B breaks language fidelity.** Only 8/20 answers came back in the
   input's language, against 16/20 for the other two. mem0's prompt
   explicitly asks for facts in the input's language.

### 12.5 Conclusion

The catalogue ranks by measured extraction capability, not parameter count.
Chat quality is a tiebreaker only — the service's primary function is
storage. A user who wants a stronger chat model pins one explicitly.

## The daemon against real llama.cpp — 25.10.2026

The first hardware run of the daemon. The servers came up correctly
and the failure was in the extraction.

**Measured on the card:**

| | |
|---|---|
| `n_ctx_slot` | 61696 |
| layers on the GPU | 33 / 33 |
| weights (CUDA0) | 8045.05 MiB |
| KV buffer | 1024.25 MiB |
| recurrent state | 100.50 MiB |

**The failure:**

```
LLMError: LLM extraction failed: Error code: 400 - {'error': {'code': 400,
'message': 'Failed to initialize samplers: Unexpected empty grammar stack
after accepting piece: <think> (248068)',
'type': 'invalid_request_error'}}
```

**The cause, from the server log:**

```
D common_sampler_init: prefill token: 248068 = <think>
E common_sampler_init: error initializing grammar sampler for grammar:
    root ::= "<|im_start|>assistant\n" space response-format
    space ::= | " " | "\n"{1,2} [ \t]{0,20}
```

mem0 asks for `response_format: {"type": "json_object"}`. llama.cpp
builds a JSON grammar from it. The chat template puts an open
`<think>` into the prompt because the thinking mode is on, and the
grammar breaks. The failure is in the prompt, not in the answer.

**The fix:** `--reasoning off`. It sets `enable_thinking` to false
(`common/arg.cpp`), so the template closes the block and the prompt
stays valid whitespace for the grammar.

`--reasoning-format none` is not enough - it controls the parsing of
the answer, not the thinking mode. The project used it and the
extraction failed.

**Also measured:** the per-process reading from `nvidia-smi` returns
N/A for every process on this machine. The card is managed by the
display driver (WDDM), not by the NVIDIA driver. The GPU proof
therefore comes from the server log, which reports the device, the
number of layers on the GPU and the size of every buffer.

## The full daemon run — 25.10.2026, 08:56

The run after the reasoning fix. **Everything worked.**

```json
{"mode": "full", "problems": [], "embedder_dimensions": 768,
 "collection_dimensions": 768, "memories": 3}
```

**Both servers on the GPU, read from the server logs:**

| | Embedder | LLM (Qwen3.5-9B Q8_0) |
|---|---|---|
| device | CUDA0 | CUDA0 |
| layers on the GPU | 13 / 13 | 33 / 33 |
| weights | 66.92 MiB | 8045.05 MiB |
| CPU mapped (host RAM) | 12.58 MiB | 1030.62 MiB |
| KV buffer | — | 1045.50 MiB |
| recurrent state | — | 100.50 MiB |
| compute buffer | 24.51 MiB | 371.79 MiB |

**The extraction, through the real LLM:**

| Input | Facts | Time |
|---|---|---|
| "The user rides a bicycle to the office every day." | 1 | 6 s |
| "The build server has 64 GB of RAM." | 1 | 2 s |
| "The coffee machine in the office is fully automatic." | 1 | 2 s |

**The recall, through the real embedder:** every question found the
right fact first, in 0 s each. The search is a pure embedding plus a
vector comparison - no LLM call, which is why it is instant.

**Why the first extraction took 6 s and the others 2 s:** the first
one processed **8224 prompt tokens** (the extraction prompt plus the
memories stored so far), the later ones only 120 and 155. The prompt
is cached, so the second and third call are fast.

That 8224-token figure is exactly the context floor from §4.7: below
it, mem0 cannot extract at all. The measured run confirms the floor
from the other side.

**Measured generation speed:** 34.39 tokens per second. That is GPU
speed - the CPU fallback is around 6.

**The VRAM came back:** 1021 MiB before the run, 1021 MiB after it.
Both servers were stopped cleanly.

**The whole run took 17 seconds**, including loading both models.

## The first extraction benchmark — 25.10.2026, 09:00

The benchmark ran, but **the benchmark was wrong, not the model.**

```json
{"sentences": 22, "json_validity": 0.3636, "recall": 0.3182,
 "empty_rate": 0.6364, "language_fidelity": 0.8750,
 "facts_stored": 8}
```

**What the log shows:**

| Sentences | Language | Facts | Time |
|---|---|---|---|
| 1–8 | English | 1 each | 2.0–5.9 s |
| 9–22 | German, Spanish, French | **0 each** | 0.6 s each |

**The cause:** the corpus had every non-English sentence as a
translation of an English one. Sentences 9–16 were the German
translations of sentences 1–8, 17–19 the Spanish of 1–3, 20–22 the
French of 1–3.

The extraction prompt asks the model to deduplicate:

> "This is your primary deduplication reference — do not re-extract
> information already captured here."

The model did exactly that: 8 unique facts stored, 14 translations
recognised and suppressed. The `empty_rate` metric counted the
correct behaviour as a 64 % failure rate.

**This is a good result, measured wrongly.** A service that stores
the same fact three times in three languages is worse than one that
misses a fact: the store fills with repetitions and every search
returns near-duplicates.

**The corpus is now built differently:**

| | Count |
|---|---|
| unique English facts | 8 |
| unique German facts | 8 (different content, not translations) |
| unique Spanish facts | 4 |
| unique French facts | 4 |
| deliberate duplicates (cross-language) | 6 |

**The metrics are split apart:**

| Metric | What it means |
|---|---|
| `extraction_rate` | of the new facts, how many produced a fact |
| `dedup_precision` | of the duplicates, how many produced **nothing** |
| `false_empty` | new facts that produced nothing — a real failure |
| `false_duplicate` | duplicates that produced a fact — a real failure |
| `recall_same_language` | the question and the fact in the same language |
| `recall_cross_language` | the question in one language, the fact in another |

The cross-language recall is the interesting one: it says whether a
German question finds a fact that was stored in English.

**Also fixed:** the first version deleted the server logs together
with the work directory. The logs are the evidence for the buffer
lines, so they are now copied next to the result file.

## The extraction benchmark — 25.10.2026, the second run

The corrected corpus: 24 unique facts in four languages and 6
deliberate duplicates. No problems, exit 0.

```json
{"sentences": 30, "new_facts": 24, "duplicates": 6,
 "extraction_rate": 0.958, "dedup_precision": 0.833,
 "false_empty": 0, "false_duplicate": 1, "atomicity": 1.000,
 "recall": 0.567, "recall_same_language": 0.857,
 "recall_cross_language": 0.647,
 "language_preservation": 0.292, "facts_stored": 24,
 "average_extraction_seconds": 2.071}
```

### What the numbers mean

| Metric | Value | Reading |
|---|---|---|
| `extraction_rate` | 0.958 | 23 of 24 new facts produced a fact |
| `dedup_precision` | 0.833 | 5 of 6 duplicates produced **nothing** |
| `atomicity` | 1.000 | every fact is one fact |
| `recall_same_language` | 0.857 | the question and the fact in one language |
| `recall_cross_language` | 0.647 | the question in one language, the fact in another |
| `language_preservation` | 0.292 | **mem0 rewrites the facts into English** |

### The findings

**1. mem0 normalises every fact into English.** The extraction prompt
is English and the model follows it:

| Input | Stored fact |
|---|---|
| "Der Drucker im Büro ist ein Laserdrucker." | "The printer in the user's office is a laser printer." |
| "Die Firewall blockiert eingehende Verbindungen aus dem Internet." | "The office firewall blocks incoming connections from the internet." |
| "L'utilisateur écrit la documentation en français." | "User writes documentation in French." |

**That is not a failure.** It is what makes the cross-language recall
work: a German question finds a fact that was stored in English,
because everything is in the same language. The metric was renamed
from `language_fidelity` to `language_preservation`, and the
counterpart `language_normalisation` says how much was rewritten.

**2. The cross-language deduplication works, and it is not perfect.**
Five of the six translations were recognised and suppressed. The one
that slipped through was the loosest paraphrase:

| Duplicate | Original | Result |
|---|---|---|
| "Der Benutzer fährt jeden Tag mit dem Fahrrad zur Arbeit." | "The user rides a bicycle to the office every day." | **stored again** |
| "Der Build-Server hat 64 GB RAM." | "The build server has 64 GB of RAM." | suppressed |
| "Die Kaffeemaschine im Büro ist vollautomatisch." | "The coffee machine in the office is fully automatic." | suppressed |
| "Das Büro befindet sich im dritten Stock." | "The office is on the third floor." | suppressed |
| "El usuario va al trabajo en bicicleta todos los días." | "The user rides a bicycle to the office every day." | suppressed |
| "La machine à café du bureau est entièrement automatique." | "The coffee machine in the office is fully automatic." | suppressed |

The one that leaked is the one whose wording differs the most: "zur
Arbeit" against "to the office". The closer the wording, the more
reliably the duplicate is recognised.

**3. One new fact was not stored.** "Der Benutzer trinkt morgens
Kaffee aus einer Tasse." produced nothing. That is the model's
judgement: a coffee cup in the morning is not worth a memory. It is
the one `false_empty` in the run.

**4. The recall was measured wrongly.** The first version of the
metric looked for the fact under the sentence that was searched. A
duplicate that was correctly deduplicated has no fact of its own, so
the question about it has to be answered by the fact from the
original. Without that the metric reported a failure even when the
right fact was on the first place - which is what happened with
"Wie viel RAM hat der Build-Server?". The corpus now names the
original for every duplicate.

### The hardware

| | |
|---|---|
| VRAM free before | 11101 MiB |
| VRAM used after | 1015 MiB |
| average extraction | 2.07 s per sentence |


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

### 1.5 BERT architectures do not follow the per-layer formula

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

Computed with the planner's budget model, KV `q8_0`, 2 slots, embedder on GPU:

| VRAM | Configuration | `n_ctx` | Per slot |
|---|---|---|---|
| 12 GB | Qwen3.5-9B-Q4_K_M | 249 146 | 124 573 |
| 12 GB | Qwen3.5-9B-Q8_0 | 63 392 | 31 696 |
| 8 GB | Qwen3.5-9B-Q4_K_M | 51 798 | 25 899 |
| 8 GB | gemma-4-E2B-Q4_XL | 78 774 | 39 387 |
| 6 GB | gemma-4-E2B-Q4_XL | 34 044 | 17 022 |
| 6 GB | Qwen3-4B-Instruct-Q4 | 23 922 | 11 961 |
| 5 GB | Qwen3-4B-Instruct-Q4, KV `q4_0` | 26 076 | 13 038 |
| 4 GB | **no viable configuration** | — | — |

Without the floor, the same planner would have proposed `n_ctx = 2178` for
the 4 GB card — a configuration that loads, answers chat requests, and
cannot store a single memory.

### 9.3 Two-model comparison

Same budget model, KV `q8_0`, 2 slots:

| VRAM | 9B + 4B (split) | 9B alone |
|---|---|---|
| 12 GB | 93 862 (46 931/slot) | 249 146 (124 573/slot) |
| 8 GB | does not fit (10 011 MiB needed) | 51 798 (25 899/slot) |
| 6 GB | does not fit (9 683 MiB needed) | does not fit |

The second process costs ~610 MiB (320 CUDA context + 290 compute buffer),
which is more than the entire KV cache of the 9B at a usable context size.

### 9.4 Minimum configuration breakdown (4 GB attempt)

| Item | MiB |
|---|---|
| Qwen3-4B-Instruct Q4_K_M weights | 2120 |
| KV cache, 12 288 tokens | 1570 |
| Process overhead | 610 |
| Reserve (10 %) + headroom (6 %) | 654 |
| **Total required** | **4954** |
| **Available** | **4096** |
| **Shortfall** | **858** |

With the embedder on CPU the shortfall drops to ~318 MiB, which makes the
4 GB tier marginal rather than impossible for users willing to reduce the
KV cache type to `q4_0`.


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
| 12 GB | Qwen3.5-9B-Q8_0 | 67 122 | 33 561 |
| 8 GB | Qwen3.5-9B-Q4_K_M | 54 846 | 27 423 |
| 6 GB | Qwen3.5-9B-IQ3_XXS | 36 330 | 18 165 |
| 5 GB | Qwen3.5-4B-Q5_K_M | 29 078 | 14 539 |
| 4 GB | Qwen3.5-4B-IQ3_XXS | 48 050 | 24 025 |

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

The second process costs ~610 MiB (320 CUDA context + 290 compute buffer),
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
| Qwen3-8B | 8306 | 6414 | 5580 | 4795 | 4369 | 3252 |
| Llama-3.1-8B | – | 6290 | 5467 | 4693 | 4257 | 3168 |
| gemma-4-E2B-it | 4814 | 4293 | 3201 | 2963 | 2846 | 2263 |
| Qwen3.5-4B | 4275 | 3363 | 2998 | 2614 | 2362 | 1859 |
| Qwen3-4B-Instruct-2507 | 4082 | 3153 | 2756 | 2382 | 2166 | 1597 |

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




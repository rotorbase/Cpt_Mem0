# Model Catalogue

The catalogue maps a GPU to a concrete GGUF file. It is the data the
planner reasons over: file size, architecture, KV constants, license.

All file sizes below were read from the Hugging Face API
(`?blobs=true`) on 2026-10-24. VRAM figures are the file size minus the
measured 11 % host fraction (§2 of `verification.md`).

---

## 1. Selection algorithm

```
1. Determine VRAM total and the usable budget
   usable = total - reserve(10%) - headroom(6%)
2. Sort candidates by EXTRACTION CAPABILITY (measured, see §1.1),
   not by parameter count
3. For each candidate model (best extraction first):
     for each quantisation (best first):
       weights_vram = file_size * 0.89
       if weights_vram + OVERHEAD > usable: continue
       compute n_ctx from the remaining budget
       if n_ctx < CONTEXT_FLOOR_TOTAL: continue
       -> selected
```

`OVERHEAD` = 812 MiB (process) + embedder cost (540 MiB GPU / 0 CPU) +
KV floor (1570 MiB).

### 1.1 Why the ranking is extraction capability, not parameter count

The original ranking by parameter count was wrong. Measured against the
real mem0 extraction prompt (SPEC §17.4):

| Model | Params | Extraction recall | JSON valid | Empty results | Halluc. |
|---|---|---|---|---|---|
| **gemma-4-E2B-it** | 2B | **96 %** | 20/20 | 0 | 0 |
| Qwen3.5-9B | 9B | 88 % | 20/20 | **2** | 0 |
| Qwen3-4B-Instruct | 4B | 85 % | 20/20 | 0 | 0 |

The 2B model extracts better than the 9B, uses less than half the VRAM and
runs 2.3× faster. The 9B's failure mode is the dangerous one: it returns
an empty fact list twice out of twenty runs — a service that appears
healthy while storing nothing.

Extraction capability is the axis that determines whether the service
works at all, so it decides the ranking. Chat quality is a tiebreaker
only, and a user who wants a stronger chat model pins one explicitly
(`--model`).

Models without a measurement rank **below** measured ones, and `plan`
labels the basis. An unmeasured model is a guess; a measured one is
evidence.

---

## 2. Catalogue contents

### 2.1 Models

| Model | Params | Architecture | Note |
|---|---|---|---|
| Qwen3.5-9B | 9B | `qwen35` hybrid | Verified on this machine |
| Qwen3.5-4B | 4B | `qwen35` hybrid | |
| Qwen3-8B | 8B | `qwen3` standard | Wide ecosystem support |
| Llama-3.1-8B-Instruct | 8B | `llama` standard | |
| Qwen3-4B-Instruct-2507 | 4B | `qwen3` standard | Smallest viable model (§9) |
| gemma-4-E2B-it | ~2B | `gemma4` hybrid + sliding window | Smallest; KV needs calibration |

### 2.2 Quantisations and file sizes (MiB on disk / MiB in VRAM)

| Model | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | IQ4_XS | IQ3_XXS | Q3_K_M | Q2_K |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 9086 / 8087 | 7113 / 6330 | 6273 / 5583 | 5417 / 4821 | 4929 / 4387 | 3830 / 3409 | 4457 / 3967 | 3931 / 3498 |
| Qwen3-8B | 8306 / 7392 | 6414 / 5709 | 5580 / 4966 | 4795 / 4267 | 4369 / 3888 | 3252 / 2895 | 3933 / 3500 | 3130 / 2785 |
| Llama-3.1-8B | – | 6290 / 5598 | 5467 / 4866 | 4693 / 4177 | 4257 / 3789 | 3168 / 2819 | 3833 / 3411 | 3032 / 2698 |
| gemma-4-E2B-it | 4815 / 4285 | 4293 / 3821 | 3201 / 2849 | 2963 / 2637 | 2846 / 2533 | 2263 / 2014 | 2419 / 2153 | 2292 / 2040 |
| Qwen3.5-4B | 4275 / 3805 | 3363 / 2993 | 2998 / 2668 | 2614 / 2326 | 2362 / 2102 | 1859 / 1654 | 2187 / 1947 | 1851 / 1647 |
| Qwen3-4B-Instruct-2507 | 4082 / 3633 | 3153 / 2806 | 2756 / 2453 | 2382 / 2120 | 2166 / 1927 | 1597 / 1421 | 1979 / 1762 | 1592 / 1417 |

Source repositories:

| Model | Repository |
|---|---|
| Qwen3.5-9B | `unsloth/Qwen3.5-9B-GGUF` |
| Qwen3.5-4B | `unsloth/Qwen3.5-4B-GGUF` |
| Qwen3-8B | `unsloth/Qwen3-8B-GGUF` |
| Llama-3.1-8B-Instruct | `unsloth/Llama-3.1-8B-Instruct-GGUF` |
| Qwen3-4B-Instruct-2507 | `unsloth/Qwen3-4B-Instruct-2507-GGUF` |
| gemma-4-E2B-it | `unsloth/gemma-4-E2B-it-GGUF` |

**Note on gemma-4-E2B-it Q8_0:** the repository also contains
`MTP/mtp-gemma-4-E2B-it-Q8_0.gguf` (93 MiB) — a speculative-decoding draft
model, not the main model. Catalogue resolution must exclude `MTP/` and
`mmproj` paths.

---

## 3. KV constants per model

Every constant was read from the model's own GGUF header with this
project's reader (`src/cpt_mem0/gguf.py`) — from a local file or through
an HTTP range request against the HuggingFace repository. None of it
required a GPU.

| Model | `full_attention_interval` | KV layers | Bytes/token (f16) | Source |
|---|---|---|---|---|
| Qwen3.5-9B | 4 | 8 of 32 | **32 768** | **measured** (8 logs + header) |
| Qwen3.5-4B | 4 | 8 of 32 | **32 768** | **measured** (header) |
| Qwen3-8B | 1 | 36 of 36 | **147 456** | **measured** (header) |
| Llama-3.1-8B-Instruct | 1 | 32 of 32 | **131 072** | **measured** (header) |
| Qwen3-4B-Instruct-2507 | 1 | 36 of 36 | **147 456** | **measured** (header) |
| gemma-4-E2B-it | 1 + sliding window | shared | **calibrate** | not reliable |

The `gemma4` architecture uses `attention.shared_kv_layers = 20` and
`attention.sliding_window = 512`, so the per-layer formula does not apply.
Measured behaviour of a related model: 12 MiB at `n_ctx` 2048, identical
to a completely different architecture — a floor, not a slope.

**Calibration** (§4.3 of SPEC): load the model at two `n_ctx` values, read
`CUDA0 KV buffer size` from the log, derive the slope. Costs one model
load (~45 s for a 9B) and is cached in `models.json`.

### 3.1 Quantised cache sizes

| Cache type | Bytes per 32 values | Bytes per value | Share of f16 |
|---|---|---|---|
| `f16` | 64 | 2.0 | 100 % |
| `q8_0` | 34 | 1.0625 | 53.125 % |
| `q4_0` | 18 | 0.5625 | 28.125 % |

From llama.cpp's block layouts (`ggml/src/ggml-common.h`). The `q8_0`
value was previously estimated at 1.125 bytes per value, 5.9 % too
pessimistic.

---

## 4. Resulting selections

**Generated from the planner** (`scripts/regenerate_tiers.py`). KV
`q8_0`, 2 slots, embedder on GPU, context floor 21 000 enforced.

### 12 GB (12288 MiB)

| Rank | Model | Quant | `n_ctx` | Per slot | Weights |
|---|---|---|---|---|---|
| 1 | Qwen3.5-9B | Q8_0 | 62 976 | 62 976 | 8087 MiB |

### 8 GB (8192 MiB)

| Rank | Model | Quant | `n_ctx` | Per slot | Weights |
|---|---|---|---|---|---|
| 1 | Qwen3.5-9B | Q5_K_M | 17 920 | 17 920 | 5603 MiB |

### 6 GB (6144 MiB)

| Rank | Model | Quant | `n_ctx` | Per slot | Weights |
|---|---|---|---|---|---|
| 1 | Qwen3.5-9B | IQ3_XXS | 39 168 | 39 168 | 3409 MiB |

### 5 GB (5120 MiB)

| Rank | Model | Quant | `n_ctx` | Per slot | Weights |
|---|---|---|---|---|---|
| 1 | Qwen3-4B-Instruct-2507 | Q5_K_M | 13 056 | 13 056 | 2133 MiB |

### 4 GB (4096 MiB)

| Rank | Model | Quant | `n_ctx` | Per slot | Weights |
|---|---|---|---|---|---|
| 1 | Qwen3-4B-Instruct-2507 | IQ3_XXS | 15 104 | 15 104 | 1654 MiB |

### 4.1 Largest reachable context

The cascade prefers quality; this table is the opposite extreme — the
largest context any configuration on the card can reach. Both are valid
answers to different questions.

| VRAM | Configuration | `n_ctx` | Per slot |
|---|---|---|---|
| 12 GB | Qwen3.5-4B-IQ3_XXS, KV `q4_0` | 876 850 | 438 425 |
| 8 GB | Qwen3.5-4B-IQ3_XXS, KV `q4_0` | 482 156 | 241 078 |
| 6 GB | Qwen3.5-4B-IQ3_XXS, KV `q4_0` | 286 458 | 143 229 |
| 5 GB | Qwen3.5-4B-IQ3_XXS, KV `q4_0` | 188 496 | 94 248 |
| 4 GB | Qwen3.5-4B-IQ3_XXS, KV `q4_0` | 90 760 | 45 380 |

`IQ3_XXS` is a coarse quantisation. Whether it extracts facts well
enough is an open measurement (SPEC §17.4) — the planner does not know
yet, and does not pretend to.

---

## 5. The catalogue as a file

The catalogue ships as `catalogue.json` in the repository and is read at
runtime. It is **not** fetched from the network — a tool that cannot start
without internet access is not self-contained.

```json
{
  "version": 1,
  "updated": "2026-10-24",
  "models": [
    {
      "id": "Qwen3.5-9B",
      "repo": "unsloth/Qwen3.5-9B-GGUF",
      "params": 9000000000,
      "architecture": "qwen35",
      "kv_bytes_per_token_f16": 32768,
      "kv_source": "measured",
      "license": "apache-2.0",
      "quantisations": [
        {"name": "Q8_0", "file": "Qwen3.5-9B-Q8_0.gguf",
         "size_bytes": 9527502048},
        {"name": "Q4_K_M", "file": "Qwen3.5-9B-Q4_K_M.gguf",
         "size_bytes": 5680522464}
      ]
    }
  ]
}
```

Files are downloaded from Hugging Face at runtime and verified against
`size_bytes` plus the GGUF header. They are never vendored into the
repository.

---

## 6. Adding a model

Two paths:

**Curated** — a pull request to `catalogue.json` with the file size read
from the Hugging Face API and the KV constant either measured or marked
`formula`.

**Ad-hoc** — `cpt-mem0 models add <repo>:<quant>` resolves the file from
Hugging Face at runtime, reads the GGUF header for the architecture and
KV constants, and stores the result in the local `models.json` overlay.
If the architecture is not one the formula handles, the planner calibrates
on first use.

The overlay is layered on top of the shipped catalogue, so an ad-hoc
addition never modifies the repository file.

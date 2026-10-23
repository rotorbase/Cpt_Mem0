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
        for cache_type in [f16, q8_0, q4_0]:
            for embedder_mode in [gpu, cpu]:
                cfg = compute(model, quant, cache_type, embedder_mode)
                if cfg.fits(reserve, headroom):
                    return cfg
return "no viable configuration"   # with an explanation of the binding constraint
```

### 4.5 KV cache quantization

`--cache-type-k` / `--cache-type-v` are the single strongest lever and are
**never set by `llama.cpp` by default**:

| Cache type | KV size (16 384 tokens) | Note |
|---|---|---|
| `f16` | 512 MiB | default |
| `q8_0` | 288 MiB | minimal quality loss — **recommended default** |
| `q4_0` | 144 MiB | larger loss on the value cache |

Halving the KV cost doubles the context for the same VRAM.

### 4.6 `--fit` must be disabled

The launch configuration always contains `--fit off`. With `--fit` enabled,
`llama.cpp` may reduce `n_ctx` and offload layers at any time, including
after the planner computed a valid configuration. The planner's result must
be the final word.

### 4.7 Reference results

Computed with the verified planner against measured constants
(RTX 3060 12 GB, embedder resident, 2 slots, reserve 10 % + headroom 6 %):

| VRAM | Selected configuration | `n_ctx` | Per slot |
|---|---|---|---|
| 12 GB | Qwen3.5-9B Q8_0, KV `q8_0` | 67 520 | 33 760 |
| 12 GB | Qwen3.5-9B Q4_K_M, KV `q8_0` | 264 200 | 132 100 |
| 8 GB | gemma-4-E2B Q4_K_XL, KV `q8_0` | 83 590 | 41 795 |
| 8 GB | Qwen3.5-9B Q4_K_M, KV `q8_0` | 55 244 | 27 622 |
| 6 GB | gemma-4-E2B Q4_K_XL, KV `q8_0` | 36 228 | 18 114 |
| 4 GB | Qwen3-4B-Instruct Q4_K_M, KV `q8_0` | 2 394 | 1 197 |

The 4 GB row is the failure mode this product exists to avoid: a model that
loads but has no usable context.

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

### 5.4 Known quality note

`nomic-embed-text` expects task prefixes (`search_query: ` /
`search_document: `). Measured cosine between prefixed query and document
is 0.9359 — omitting the prefix costs retrieval quality. Neither backend
applies it today. Adding prefix support is a planned enhancement.

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

## 9. Configuration

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
  calibration: "auto"         # auto | always | never

update:
  check_on_start: false       # P5: never update silently
```

Every field has a working default. A fresh install runs with an empty
config.

---

## 10. CLI

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
GPU     NVIDIA GeForce RTX 3060          12288 MiB total
LLM     Qwen3.5-9B-Q8_0.gguf             9086 MiB
        weights 8045 MiB  KV 1058 MiB (32768 B/tok, f16→q8_0)
        n_ctx 67520  (33760 per slot, 2 slots)
EMBED   nomic-embed-text-v1.5            llamacpp, GPU
        weights 78 MiB  KV 12 MiB
STORE   qdrant embedded  ~/.local/share/cpt-mem0/store
TOTAL   9841 MiB model  +  1200 reserve  +  737 headroom  =  11778 / 12288
```

---

## 11. API surface

REST on `127.0.0.1:8765`, plus an MCP server on the same port.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/memories` | add facts from a conversation turn |
| `GET` | `/v1/memories` | list memories (`?user_id=`) |
| `POST` | `/v1/memories/search` | semantic search |
| `PATCH` | `/v1/memories/{id}` | update |
| `DELETE` | `/v1/memories/{id}` | delete |
| `GET` | `/v1/health` | liveness + GPU state |
| `GET` | `/v1/status` | full runtime status |
| `GET` | `/v1/chat/completions` | OpenAI-compatible passthrough to the local LLM |

Authentication: per-client API keys in `keys/clients.json`. The daemon
generates one on first run. Keys are never written to logs.

Multi-tenancy: `user_id` is a request parameter. A client may be scoped to a
single `user_id`; the daemon enforces the scope. This is a mem0-native
concept, not an agent-framework feature.

---

## 12. Health checks and failure handling

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

## 13. Verification data

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

## 14. Implementation phases

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

## 15. Licensing and third-party components

The project is MIT licensed (`LICENSE`).

### 15.1 Runtime dependencies

| Component | License | Implication |
|---|---|---|
| `mem0ai/mem0` | Apache-2.0 | Permissive; NOTICE/attribution applies when redistributed |
| `qdrant-client` | Apache-2.0 | Permissive; NOTICE/attribution applies when redistributed |
| `ggml-org/llama.cpp` | MIT | Permissive; attribution required in distribution |
| `fastembed` (Qdrant) | Apache-2.0 | Permissive |
| Model weights (e.g. `nomic-embed-text-v1.5`, Qwen3.5) | Apache-2.0 | Per-model license; surfaced at download time |

MIT as the project license accepts all of these as inputs.

### 15.2 Obligations when shipping a bundled environment

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

### 15.3 What the app downloads at runtime

| Artifact | Source | License |
|---|---|---|
| `llama.cpp` binary | GitHub releases (pinned b-build) | MIT |
| CUDA runtime | bundled in the `cudart-` archive | NVIDIA EULA |
| GGUF models | Hugging Face | per model |
| ONNX embedder models | Hugging Face | per model |

None of these are vendored into the source repository.

---

## 16. Open questions

1. **Model catalogue scope.** Which models ship in the default catalogue?
   Options: a curated set of 3–4 popular GGUF architectures, or fetch
   metadata from Hugging Face on demand. Recommendation: curated set with
   on-demand extension.
2. **MCP tool surface.** Should the MCP server expose raw CRUD, or a
   higher-level "remember this conversation / recall about X" interface?
   Recommendation: raw CRUD plus two convenience tools.
3. **Prompt-prefix support** for nomic embeddings (§5.4) — implement now or
   defer? It measurably improves retrieval.
4. **Multi-process store access.** Qdrant embedded forbids it (§3.1). If
   demand appears for direct client access to the store, a Qdrant server
   mode becomes necessary — at the cost of the zero-setup property.

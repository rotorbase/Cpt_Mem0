# Cpt_Mem0

Self-contained local [mem0](https://github.com/mem0ai/mem0) memory service for
consumer GPUs.

Runs a language model, an embedder and a vector store on one graphics card.
No cloud, no Docker, no dependency on any other application.

## The problem it solves

`llama.cpp`'s `--fit` mechanism is **on by default**. When a model doesn't
fit your GPU, it silently reduces the context window toward its floor
(4096 tokens) and pushes layers into host RAM. The model loads, answers, and
has almost no context — with no error message anywhere.

Cpt_Mem0 computes the configuration **before** launch: which model, at which
quantization, with how much context, with the embedder where. `--fit` is
turned off.

```
$ cpt-mem0 plan

GPU     RTX 3060                          12288 MiB total
LLM     Qwen3.5-9B Q8_0                   9086 MiB
        weights 8087 MiB  KV 1114 MiB (17408 B/tok, q8_0)
        n_ctx 67072  (67072 per slot, 2 slots)
        --fit off  --flash-attn on  --reasoning-format none  --kv-unified
        ranking basis: unmeasured
EMBED   llamacpp                          on GPU
TOTAL   10350 MiB used  +  1200 reserve  +  737 headroom  =  12287 / 12288
```

`plan` starts nothing and touches nothing. It prints the configuration
that `start` would use.

## What it does

- **Solves the VRAM budget.** Model, quantization, KV cache type and context
  size are computed from your actual card — including hybrid architectures,
  where the naive KV formula is off by a factor of four.
- **Ships its own `llama.cpp`.** Pinned to an exact build, with the CUDA
  runtime bundled. Nothing to install system-wide.
- **Keeps the embedder on the CPU when the GPU is tight.** The store stays
  compatible either way — verified, not assumed.
- **Runs the store embedded.** Qdrant in-process: no server, no Docker, no
  database daemon.
- **Stays local.** No telemetry, no outbound requests.

## Install

```
cpt-mem0 init
```

That detects your GPU, fetches the pinned `llama.cpp` build with CUDA
runtime bundled, selects a model and quantisation that fit, and writes a
configuration. It does not start anything until you say so.

## Usage

```
cpt-mem0 plan       # show the computed configuration
cpt-mem0 start      # run the daemon (REST + MCP on 127.0.0.1:8765)
cpt-mem0 status     # processes, VRAM, store size
cpt-mem0 update     # update the llama.cpp build (never automatic)
```

## Status

**Phase 1 — the planner is implemented and tested.**

| Component | State |
|---|---|
| GGUF header reader | done, 12 tests |
| VRAM planner | done, 48 tests |
| Model catalogue | done, 6 models / 35 quantisations, all sizes and KV constants measured |
| CLI (`plan`, `models`, `validate-catalogue`) | done |
| llama.cpp bootstrap and self-update | not started |
| Daemon, REST and MCP server | not started |
| Store integration | not started |

The test suite is pure logic: it reads no GPU and starts no model.

```
$ pytest
84 passed
```

There is a second suite that does need a free card — it starts one
`llama-server` with the planned flags, asks it to generate something,
and compares the allocation against the prediction. Step-by-step
commands are in [docs/testing.md](docs/testing.md).

See [SPEC.md](SPEC.md) for the full design and the open questions.

## Documents

| Document | Content |
|---|---|
| [SPEC.md](SPEC.md) | Technical specification, 17 sections |
| [docs/verification.md](docs/verification.md) | Every measurement behind the spec's numbers |
| [docs/catalogue.md](docs/catalogue.md) | Model catalogue, file sizes, KV constants, resulting selections |
| [docs/api.md](docs/api.md) | CLI, HTTP and MCP contracts |
| [docs/testing.md](docs/testing.md) | How to run the tests, and what needs a free GPU |

## Key findings

- **The `--fit` trap.** `llama.cpp`'s `--fit` is on by default and silently
  reduces context toward a 4096-token floor when a model does not fit.
  The planner computes the configuration before launch and turns `--fit`
  off.
- **Hybrid architectures.** Qwen3.5 keeps a KV cache on only every fourth
  layer (`full_attention_interval = 4`). The naive formula overestimates
  the KV cache fourfold. Corrected, it matches the measurement exactly.
- **A service that cannot remember.** mem0's extraction prompt needs
  ~10 500 tokens in one slot. Below that, the service loads, answers chat,
  and silently stores nothing. The planner enforces a context floor and
  reports the shortfall rather than starting a broken service.
- **One model, not two.** A second `llama-server` costs ~610 MiB of CUDA
  context and compute buffer — more than the entire KV cache of the 9B at
  a usable context size.
- **A larger card should get a better model.** Ranking the catalogue only
  by quality would hand a 4 GB card the same small model as a 12 GB card.
  The planner therefore ranks by what actually fits, and the card tiers
  come out monotonic: 12 GB and 8 GB get the 9B, 6 GB gets the 9B at
  `IQ3_XXS`, 5 GB and 4 GB get the 4B.
- **The quantised KV sizes are not guesses.** `q8_0` stores 32 values in
  34 bytes, `q4_0` in 18 — llama.cpp's block layouts. An estimate of
  1.125 bytes per value for `q8_0` was 5.9 % off, and made every context
  figure in the documentation too small.

## License

MIT — see [LICENSE](LICENSE).

Third-party components and their licenses (mem0, qdrant-client, llama.cpp,
fastembed, model weights) are documented in
[SPEC.md §16](SPEC.md#16-licensing-and-third-party-components). Model
weights are downloaded at runtime, never redistributed by the project.

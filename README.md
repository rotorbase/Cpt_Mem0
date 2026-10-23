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

GPU     NVIDIA GeForce RTX 3060          12288 MiB total
LLM     Qwen3.5-9B-Q8_0.gguf             9086 MiB
        weights 8045 MiB  KV 1058 MiB (32768 B/tok, q8_0)
        n_ctx 67520  (33760 per slot, 2 slots)
EMBED   nomic-embed-text-v1.5            llamacpp, GPU
STORE   qdrant embedded
TOTAL   9841 MiB model  +  1200 reserve  +  737 headroom  =  11778 / 12288
```

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

Specification phase. See [SPEC.md](SPEC.md) for the full design, the
verification data behind every number in it, and the open questions.

## License

MIT

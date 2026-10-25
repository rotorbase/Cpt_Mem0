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
        weights 8087 MiB  KV 1046 MiB (17408 B/tok, q8_0)
        n_ctx 62976  (62976 per slot, 2 slots)
        --fit off  --flash-attn on  --reasoning-format none  --kv-unified
        recurrent state     100 MiB (hybrid model, fixed)
        ranking basis: unmeasured
EMBED   llamacpp                          on GPU
TOTAL   10348 MiB used  +  1200 reserve  +  737 headroom  =  12285 / 12288
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
- **Has a window.** A native window that the daemon serves: a big
  on/off switch for the models, the topic buttons filled from the
  store, and every memory findable and deletable.
- **Can speak.** The voice output uses the speech engine Windows
  already has - no model, no download, no VRAM.
- **Finds everything.** mem0's `get_all` returns 20 memories by default —
  the rest are unfindable. The limit is passed explicitly.

## Install

**Windows:** download the package from the
[releases](https://github.com/rotorbase/Cpt_Mem0/releases), unpack it
anywhere, and run `Cpt_Mem0.exe`. The package is a folder with the
`.exe` in it - see [docs/install-windows.md](docs/install-windows.md).

From the source tree:

```
cpt-mem0 init
```

That installs the pinned `llama.cpp` build — the binaries plus the CUDA
runtime — into its own directory. It does not change any system
setting, no driver, no PATH. The system llama.cpp stays untouched.

```
cpt-mem0 runtime    # what is installed
```

## Usage

```
cpt-mem0 init       # install the pinned llama.cpp build
cpt-mem0 runtime    # show the installed build
cpt-mem0 plan       # show the computed configuration
cpt-mem0 start      # run the daemon (REST + MCP on 127.0.0.1:8765)
cpt-mem0 ui         # run the daemon and open the window in the browser
cpt-mem0 ui --suspend  # same, but the models stay off - the VRAM stays free
cpt-mem0 status     # processes, VRAM, store size
cpt-mem0 update     # update the llama.cpp build (never automatic)
```

### The window

`cpt-mem0 ui` opens a browser window that the daemon serves itself. It
is a client - the store has an exclusive lock and only the daemon may
hold it.

The big switch starts and stops the models. The daemon and the store
stay up: the models are child processes, the memory lives in the
daemon. With the switch off the VRAM is free, the memories are still
there, and the search says so instead of failing with a connection
error.

The topic buttons are filled from the store: the facts are grouped by
the vectors that are already in Qdrant, and the local LLM gives every
group a name of two or three words.

See [docs/ui.md](docs/ui.md).

## Status

**Phase 1 (planner) and Phase 2 (runtime bootstrap) are implemented.**

| Component | State |
|---|---|
| GGUF header reader | done |
| VRAM planner | done |
| Model catalogue | done, 6 models / 35 quantisations, all sizes and KV constants measured |
| CLI (`plan`, `models`, `validate-catalogue`, `runtime`, `init`, `update`, `start`, `stop`, `status`, `ui`) | done |
| The window (HTML, CSS, JavaScript) | done, served by the daemon |
| The on/off switch for the models | done, the store stays up |
| The topic buttons (clustering over the stored vectors) | done |
| Full memory CRUD (find, change, delete one, delete all, reset) | done, verified against the real mem0 |
| The native window (pywebview on WebView2) | done, the .exe opens it |
| The Windows package (PyInstaller, onedir) | done, 89 MB, the .exe is 17 MB |
| The GitHub Actions build | done, the package is attached to every release |
| The voice output (Windows SAPI) | done, no VRAM, German voice preferred |
| llama.cpp bootstrap and self-update | done, installed and verified on this machine |
| Daemon, REST and MCP server | done |
| Store integration | done, Qdrant embedded with the mem0 engine |

The test suite is pure logic: it reads no GPU and starts no model.

```
$ pytest
249 passed
```

The daemon has not run end-to-end yet - that needs the runtime
dependencies (`mem0ai`, `qdrant-client`) in the project environment.

**Verified on this machine:** `cpt-mem0 init` installed build 11838 with
the CUDA runtime, and `--list-devices` reports
`CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB)`. The details are in
[docs/runtime.md](docs/runtime.md).

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
| [docs/ui.md](docs/ui.md) | The window: the switch, the topic buttons, the routes |
| [docs/install-windows.md](docs/install-windows.md) | The Windows package, the autostart, the uninstall |
| [docs/daemon.md](docs/daemon.md) | The daemon: the order of startup and shutdown |
| [docs/runtime.md](docs/runtime.md) | The llama.cpp runtime, the update and the rollback |
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
- **One model, not two.** A second `llama-server` costs ~812 MiB of CUDA
  context, compute buffer and recurrent state — more than the entire KV
  cache of the 9B at a usable context size.
- **A larger card should get a better model.** Ranking the catalogue only
  by quality would hand a 4 GB card the same small model as a 12 GB card.
  The planner therefore ranks by what actually fits, and the card tiers
  come out monotonic: 12 GB and 8 GB get the 9B, 6 GB gets the 9B at
  `IQ3_XXS`, 5 GB and 4 GB get the 4B.
- **The context is planned per slot, not per card.** With `--kv-unified`
  the KV cache is one shared pool, so `-c` is the limit per slot and the
  pool is not multiplied by the slot count. Getting this backwards makes
  every context figure wrong by a factor of two.
- **The server rounds the context up** to a multiple of 256 tokens, so
  the planner plans a value that is already a multiple. The printed
  number is then the number the server runs with.
- **Loading a model is not using it.** The verification suite asks the
  server to generate something. A server that answers `/health` has not
  proven that it can answer a prompt.
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

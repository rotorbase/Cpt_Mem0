# Cpt_Mem0

Self-contained local mem0 memory service for consumer GPUs.

Full technical specification: [SPEC.md](SPEC.md)

## Development status

Phase 1 — the planner is implemented and tested. 57 tests, none of
which touch a GPU.

| Component | State |
|---|---|
| GGUF header reader | done |
| VRAM planner | done |
| Model catalogue | done, 6 models / 35 quantisations |
| CLI (`plan`, `models`, `validate-catalogue`) | done |
| llama.cpp bootstrap and self-update | not started |
| Daemon, REST and MCP server | not started |
| Store integration | not started |

## Layout

```
SPEC.md              technical specification (17 sections)
README.md            overview
catalogue.json       model catalogue: sizes, KV constants, geometry
pyproject.toml       package metadata and dependencies
src/cpt_mem0/
  gguf.py            GGUF header reader
  planner.py         VRAM budget, decision cascade, context floor
  catalogue.py       catalogue data and loading
  hardware.py        GPU detection (read-only)
  cli.py             command line interface
scripts/
  run_tests.py       test entry point
  verify_plan.py     hardware verification against a real server
  regenerate_tiers.py  regenerates the reference tables
test.sh              test launcher, works from any directory
tests/               the test suite
docs/
  verification.md    measurement data behind the spec's numbers
  catalogue.md       model catalogue, file sizes, KV constants
  api.md             CLI, HTTP and MCP contracts
  testing.md         how to run the tests
```

## Conventions

- Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`)
- Markdownlint (MD013 line length disabled, MD047 newline-at-EOF enabled)
- Every claim about GPU or runtime state must cite a measurement

## Rules for this repository

- **Measure, don't assume.** GPU state is read from `nvidia-smi`, model
  sizes from the Hugging Face API, KV constants from server logs or from
  the GGUF header. Every number in a document states its source.
- **Never assert a KV constant for an architecture the formula does not
  handle.** BERT-style and sliding-window models need calibration
  (SPEC §4.3). Mark the source as `measured`, `formula` or `calibrate`.
- **A configuration that cannot extract facts is not viable.** The context
  floor (SPEC §9) is a hard constraint, not a preference.
- **No secrets in the repository.** API keys, paths and hostnames are
  redacted. Run the leak scan before every push to a public repo.
- **The pinned llama.cpp build is never "latest".** Exact b-build only
  (SPEC §7.1); old b-builds remain available forever.
- **Never write a launch flag from memory.** Check it against
  `llama-server --help`. A flag that does not exist makes the server
  abort before it loads anything, which looks like a hang.
- **A readiness check must not depend on a single log line.** The
  server only reports the buffer sizes at debug level, so waiting for
  `KV buffer size` means waiting for a line this build never prints.
  Readiness is an HTTP answer plus a state, not a string match.
- **Loading a model is not using it.** A verification that only checks
  `/health` proves that the server is up, not that it generates.
  `scripts/verify_plan.py` sends a completion request and reports the
  answer; an embedding-only server is reported as "not applicable",
  not as a failure.
- **Log lines name a device — read the right one.** llama.cpp prints
  one buffer line per device, and the host line comes first:
  `CPU_Mapped model buffer size` before `CUDA0 model buffer size`.
  A pattern that takes the first match reports host RAM as VRAM. The
  two add up to the file size, which is the check.
- **One VRAM reading is a moment, not the allocation.** The server
  keeps allocating after it reports the model as loaded. Sample
  repeatedly and keep the highest value.
- **Check how many layers reached the GPU.** A partial offload means
  the card did not have room for the planned configuration, and the
  budget cannot be verified in that case.
- **A measurement that is not written down is lost.** The verification
  writes its server log to `results/server.log`, so a later run can be
  compared against an earlier one and a wrong figure can be traced.
- **Count every buffer the framework allocates, not only the known
  ones.** A hybrid architecture keeps a recurrent state next to the KV
  cache — a fixed cost that does not scale with the context. The GGUF
  header says which models have it (`<arch>.ssm.state_size`). A size
  that cannot be derived is stored as a measured catalogue value; a
  hybrid model without one is not plannable. See SPEC §4.11.
- **A buffer that scales with the context needs a value per operating
  point.** The compute buffer measured 116 MiB at 16 384 tokens and
  392 MiB at 67 072. A constant from a small run is too small.
- **Before a GPU run, compute the need and compare it with the free
  VRAM.** Starting the 9B-Q8_0 on a card with 4927 MiB free took the
  card to 275 MiB and pushed the user's own stack out of VRAM. The
  need is the file size plus the KV plus the overhead — check it
  first.
- **The KV cache layout decides what `-c` means.** With
  `--kv-unified` it is the context per slot and the pool is shared;
  without it, it is the total and each slot gets its own pool. See
  SPEC §4.10 — getting this wrong makes every context number wrong by
  a factor of the slot count.
- **The server rounds the context up** to a multiple of 256 tokens.
  Plan a value that is already a multiple, so the printed value is the
  value the server uses.
- **Degrade the cache type last.** The cascade tries every model at
  `q8_0` before it tries any model at `q4_0`; otherwise it settles for
  a configuration that misses the context floor while a viable one
  exists. See SPEC §4.11.
- **Commands in documentation are plain text with absolute paths.**
  Markdown fences and relative paths both break copy-paste: the fences
  become an unbalanced command substitution, and a relative path fails
  as soon as the working directory is not the repository.
- **Verify the logic on the CPU.** A check that needs a model can run
  llama-server with `-ngl 0` and never touch the GPU. That keeps the
  user's graphics card free while the code is still proven.

## Running the tests

`bash test.sh` works from any directory. See
[docs/testing.md](docs/testing.md) for the step-by-step commands and
what each suite needs.

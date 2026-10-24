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
- **Commands in documentation are plain text with absolute paths.**
  Markdown fences and relative paths both break copy-paste: the fences
  become an unbalanced command substitution, and a relative path fails
  as soon as the working directory is not the repository.

## Running the tests

`bash test.sh` works from any directory. See
[docs/testing.md](docs/testing.md) for the step-by-step commands and
what each suite needs.

# Cpt_Mem0

Self-contained local mem0 memory service for consumer GPUs.

Full technical specification: [SPEC.md](SPEC.md)

## Development status

Phase 0 — specification. No implementation code yet.

## Layout

```
SPEC.md              technical specification (17 sections)
README.md            overview
docs/
  verification.md    measurement data behind the spec's numbers
  catalogue.md       model catalogue, file sizes, KV constants
  api.md             CLI, HTTP and MCP contracts
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

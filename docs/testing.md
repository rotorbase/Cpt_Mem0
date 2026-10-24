# Testing

Two suites, separated by what they need.

## Logic tests — no GPU, no network, no model

```
python scripts/run_tests.py
```

or directly:

```
pytest
```

49 tests, about a tenth of a second. They cover:

| Area | What is checked |
|---|---|
| KV cache scaling | the `f16` / `q8_0` / `q4_0` factors against llama.cpp's block layouts |
| Embedder cost | GPU vs. CPU embedder VRAM |
| Candidate arithmetic | host fraction, budget, slot division, `--fit off` in the flags |
| Context floor | the 4 GB card is rejected below the floor; degraded mode is opt-in |
| Cascade order | better quantisation first; measured models before unmeasured; calibration required, never guessed |
| Catalogue integrity | sizes against the documented table; no MTP or mmproj files |
| Catalogue geometry | every KV constant follows from the stored GGUF geometry |
| Card hierarchy | more VRAM never yields a smaller model |
| GGUF reader | hybrid detection, cache types, error handling |

These run anywhere. They read no GPU and start no model, so they are
safe to run while a memory service is active.

## Hardware verification — needs a free GPU and a model file

This is the check that validates the budget model against reality. It
starts one `llama-server` with the planned flags, reads the VRAM the
server actually allocated, and compares that against the prediction.

```
python scripts/run_tests.py --hardware --vram 12288 \
    --model-file ~/.cache/huggingface/hub/Qwen3.5-9B-Q4_K_M.gguf
```

or directly:

```
python scripts/verify_plan.py --vram 12288 \
    --model-file ~/.cache/huggingface/hub/Qwen3.5-9B-Q4_K_M.gguf
```

It reports:

```
planned: Qwen3.5-9B Q8_0, n_ctx 67122, KV q8_0, embedder llamacpp
...
measured from the server log:
  n_ctx              67122
  model buffer       8045.05 MiB
  compute buffer     286.31 MiB
  KV buffer          1114.00 MiB
  GPU usage delta    9445 MiB (6904 -> 16349)

KV delta: 1114 measured vs 1114 predicted (0 MiB)
weights delta: 8045 measured vs 8087 predicted (42 MiB)
n_ctx matches the plan (67122)

The plan matches the measurement.
```

### Before you run it

**Stop any other `llama-server` first.** The check compares total GPU
usage, so a second model on the same card shows up as a delta and the
comparison becomes meaningless.

This is not a formality. A previous run of the extraction benchmark
started three additional servers alongside the running stack, the 9B
could not get its planned VRAM, `--fit` silently reduced the context,
and the measurement was worthless. The symptom was a model that
"answered" but extracted almost nothing.

The same applies to anything else on the card — a browser with hardware
acceleration, a compositor. The script prints the GPU usage before and
after, so the delta is visible.

### Options

| Option | Meaning |
|---|---|
| `--vram` | plan for a card with this much VRAM (MiB); omit to detect |
| `--model` | restrict the plan to one model id from the catalogue |
| `--quant` | restrict to one quantisation |
| `--model-file` | path to the GGUF file to load |
| `--tolerance-mib` | allowed difference between prediction and measurement (default 300) |

The tolerance exists because two terms are estimates: the host fraction
(11 %, measured to 0.8 % on the weights) and the process overhead
(compute buffer and CUDA context, which vary with the build and the
model size).

## Regenerating the reference tables

The card tiers in SPEC.md, docs/catalogue.md and docs/verification.md
are generated, not hand-written:

```
python scripts/regenerate_tiers.py
```

If the planner changes, run this and update the tables in the same
commit. The test suite asserts that the catalogue still matches the
documented sizes and geometry, so drift is caught.

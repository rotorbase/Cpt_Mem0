# Testing

Two suites, separated by what they need.

## Run it — step by step

### Step 1 — close Hermes

The watchdog stops the memory stack within about three seconds of Hermes
closing. Wait until the card is free.

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

Expected: `used` well under 1000 MiB. If it is still high, the stack has
not shut down yet — wait a moment and check again.

### Step 2 — the logic tests

No GPU, no model, no network. Safe at any time.

```bash
cd ~/Documents/Cpt_Mem0
.venv/Scripts/python.exe scripts/run_tests.py
```

Expected: `49 passed`, total under one second.

### Step 3 — the hardware verification

**Only with a free card.** This loads a model into VRAM. With the
memory stack running, the two compete for the card and the measurement
is meaningless.

With the 4B model:

```bash
cd ~/Documents/Cpt_Mem0
.venv/Scripts/python.exe scripts/verify_plan.py \
  --vram 5120 \
  --model Qwen3-4B-Instruct-2507 \
  --quant Q4_K_M \
  --model-file "C:/Users/willow/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  --port 8799
```

With the 9B:

```bash
.venv/Scripts/python.exe scripts/verify_plan.py \
  --vram 12288 \
  --model Qwen3.5-9B \
  --quant Q4_K_M \
  --model-file "C:/Users/willow/.cache/huggingface/hub/Qwen3.5-9B-Q4_K_M.gguf" \
  --port 8799
```

Takes about 45 seconds — that is the model load.

Both suites in one go, with the results written to `results/`:

```bash
.venv/Scripts/python.exe scripts/run_tests.py --hardware --vram 12288 \
  --model-file "C:/Users/willow/.cache/huggingface/hub/Qwen3.5-9B-Q4_K_M.gguf"
```

### Step 4 — check that nothing is left behind

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

Expected: back to roughly the starting value.

### Stopping a run

| Method | What happens |
|---|---|
| **Ctrl+C** | The runner terminates the child process and exits. The result file is still written, with the phase marked `interrupted`. |
| **`--timeout 120`** | The phase is aborted after 120 seconds. The child is terminated. |
| **Closing the terminal** | The Windows job object takes the whole process tree down with it. |

The job object matters. `CTRL_BREAK_EVENT` kills the Python process
hard — the `finally` block never runs — so without it a `llama-server`
would survive in VRAM. With `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, the
operating system terminates the entire child tree when the handle
closes, whether the parent exits cleanly or is killed.

Verified: the parent was killed with `Process.kill()` while a child was
sleeping. Afterwards no child process survived.

The output is read on a separate thread. Reading it in the main thread
would block in `readline()`, and the timeout would only be checked
between output lines — a process that hangs *silently* would then never
be interrupted. Measured: a silent hang is cut off after 3,08 s with a
3 s limit, not after the 120 s the process wanted to run.

## Logic tests — no GPU, no network, no model

```
python scripts/run_tests.py
```

or directly:

```
pytest
```

49 tests, about half a second. They cover:

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

### How long

Measured on this machine, three runs each:

| Command | Duration |
|---|---|
| `pytest tests/ -q` | 0,41 – 0,44 s |
| `python scripts/run_tests.py` | 0,48 – 0,49 s |
| `python scripts/regenerate_tiers.py` | 0,09 s |
| `cpt-mem0 plan` | 0,04 – 0,05 s |
| `cpt-mem0 validate-catalogue` | 0,04 – 0,05 s |

The hardware verification takes as long as the model needs to load —
45 s for a 9B on this card, plus a few seconds of measurement.

### What gets written

Everything lands in `results/` inside the repository (git-ignored):

| File | Content |
|---|---|
| `results/pytest-junit.xml` | JUnit XML, the format CI systems read |
| `results/results.json` | machine-readable summary: status, duration, pass counts |
| `results/logic.log` | full console output of the logic suite |
| `results/hardware.log` | full console output of the hardware check |

Plus `.pytest_cache/` in the repository root, written by pytest itself.

**Nothing outside the repository is touched.** No configuration file, no
service, no environment variable, no registry entry. The runner adds
`src/` to `PYTHONPATH` *for its own child processes* and leaves the
caller's environment alone.

Verified by hash: `mem0.json`, `gateway.pid` and `.env` in the Hermes
profile were byte-identical before and after a run, both running
`llama-server` processes kept their PIDs, and VRAM returned to its
starting value.

## Hardware verification — needs a free GPU and a model file

This is the check that validates the budget model against reality. It
starts one `llama-server` with the planned flags, reads the VRAM the
server actually allocated, and compares that against the prediction.
The commands are in **Run it — step by step** above.

### Why the card has to be free

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
| `--timeout` | seconds per phase (0 = no limit) |
| `--results-dir` | where to write the result files (default `<repo>/results`) |

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

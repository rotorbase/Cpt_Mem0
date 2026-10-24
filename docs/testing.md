# Testing

Two suites, separated by what they need.

## Run it — step by step

**Copy the commands as plain text.** The fences around a command in
this file are markdown, not part of the command. If you paste them
together with the command, bash sees the backticks as a
command substitution: three backticks are an unbalanced one, and bash
waits for a closing backtick instead of running anything. The symptom
is a terminal that accepts input and does nothing.

**Use absolute paths.** The commands below work from any directory.
A relative path like `.venv/Scripts/python.exe` only works after a
`cd` into the repository — from your home directory it does not exist.

### Step 1 — close Hermes

The watchdog stops the memory stack within about three seconds of Hermes
closing. Wait until the card is free.

    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

Expected: `used` well under 1000 MiB. If it is still high, the stack has
not shut down yet — wait a moment and check again.

### Step 2 — the logic tests

No GPU, no model, no network. Safe at any time, from any directory.

    bash "C:/Users/willow/Documents/Cpt_Mem0/test.sh"

Expected: `92 passed`, total under fifteen seconds.

The same thing with the full path to the interpreter, if you prefer:

    "C:/Users/willow/Documents/Cpt_Mem0/.venv/Scripts/python.exe" "C:/Users/willow/Documents/Cpt_Mem0/scripts/run_tests.py"

### Step 3 — the hardware verification

**Only with a free card.** This loads a model into VRAM. With the
memory stack running, the two compete for the card and the measurement
is meaningless.

### What you should see

The plan, then the server command, then the model loading, then the
measured values and the comparison. The script flushes its output as it
goes, so you see progress rather than a blank screen.

If it produces **no output at all**, the server aborted while parsing
its arguments — before it loaded anything. That is a flag problem, not
a hardware problem: check the flags against `llama-server --help`. The
test suite does this automatically (`tests/test_launch_flags.py`), so a
flag that llama.cpp does not know fails the tests.

The script detects the running server by asking it over HTTP
(`/health`), not by matching a log line — the wording of that line
differs between llama.cpp builds, and a mismatch looks exactly like a
hang.

**Which model to use:** the buffer sizes (`KV buffer size`, `model
buffer size`) are only printed for a chat model. An embedding model
uses a different architecture with no CUDA KV buffer, so the comparison
has nothing to measure. Use the 4B or the 9B for the hardware check.

With the 4B model — one line, no line breaks:

    "C:/Users/willow/Documents/Cpt_Mem0/.venv/Scripts/python.exe" "C:/Users/willow/Documents/Cpt_Mem0/scripts/verify_plan.py" --vram 5120 --model Qwen3-4B-Instruct-2507 --quant Q4_K_M --model-file "C:/Users/willow/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" --port 8799

With the 9B in Q8_0 — the configuration the planner picks for a 12 GB
card:

    "C:/Users/willow/Documents/Cpt_Mem0/.venv/Scripts/python.exe" "C:/Users/willow/Documents/Cpt_Mem0/scripts/verify_plan.py" --vram 12288 --model Qwen3.5-9B --quant Q8_0 --model-file "C:/Users/willow/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q8_0.gguf" --port 8799

With the 9B in Q4_K_M — smaller, loads faster, still on the GPU:

    "C:/Users/willow/Documents/Cpt_Mem0/.venv/Scripts/python.exe" "C:/Users/willow/Documents/Cpt_Mem0/scripts/verify_plan.py" --vram 12288 --model Qwen3.5-9B --quant Q4_K_M --model-file "C:/Users/willow/.cache/huggingface/hub/Qwen3.5-9B-Q4_K_M.gguf" --port 8799

Takes about 45 seconds — that is the model load.

### What the run has to show

A successful run reports four things:

    planned: Qwen3.5-9B Q8_0, n_ctx 67072, KV q8_0, embedder llamacpp
    inference: OK - ready.
    ...
    n_ctx_slot matches the plan (67072)
    inference: the server generated a response

The `inference` line is the one that proves the server is usable, not
merely loaded. The check sends a real completion request; a server that
answers `/health` but generates nothing is reported as a failure.

If the run ends with `timeout after 90s`, the server came up but the
check never reached its end condition. Send the output — the three
causes are a flag the build does not know, a log line this build does
not print, and a model file that is not where the command says.

Both suites in one go, with the results written to `results/`:

    "C:/Users/willow/Documents/Cpt_Mem0/.venv/Scripts/python.exe" "C:/Users/willow/Documents/Cpt_Mem0/scripts/run_tests.py" --hardware --vram 12288 --model Qwen3.5-9B --quant Q8_0 --model-file "C:/Users/willow/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q8_0.gguf"

### Step 4 — check that nothing is left behind

    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

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

    bash "C:/Users/willow/Documents/Cpt_Mem0/test.sh"

or directly:

    "C:/Users/willow/Documents/Cpt_Mem0/.venv/Scripts/python.exe" -m pytest "C:/Users/willow/Documents/Cpt_Mem0/tests" -q

92 tests, about ten seconds. They cover:

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
| KV pool semantics | `-c` is the per-slot limit with `--kv-unified`; the pool is not multiplied |
| Context rounding | the planned value is a multiple of 256, so the server uses it unchanged |
| Cascade order | the cache type is degraded last; every tier reaches the context floor |
| Verification script | the buffer patterns match the real format; the run ends when the server is up |
| GGUF reader | hybrid detection, cache types, error handling |

These run anywhere. They read no GPU and start no model, so they are
safe to run while a memory service is active.

### How long

Measured on this machine, three runs each:

| Command | Duration |
|---|---|
| `pytest tests/ -q` | 10,3 s (92 tests) |
| `python scripts/run_tests.py` | 0,48 – 0,49 s |
| `python scripts/regenerate_tiers.py` | 0,09 s |
| `cpt-mem0 plan` | 0,04 – 0,05 s |
| `cpt-mem0 validate-catalogue` | 0,04 – 0,05 s |

The hardware verification takes as long as the model needs to load —
45 s for a 9B on this card, plus a few seconds of measurement.

The logic suite grew from 0,4 s to 10,3 s because the verification tests
now start a real llama-server on the CPU (`-ngl 0`) to prove the
liveness check works. That keeps the graphics card free.

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

    "C:/Users/willow/Documents/Cpt_Mem0/.venv/Scripts/python.exe" "C:/Users/willow/Documents/Cpt_Mem0/scripts/regenerate_tiers.py"

If the planner changes, run this and update the tables in the same
commit. The test suite asserts that the catalogue still matches the
documented sizes and geometry, so drift is caught.

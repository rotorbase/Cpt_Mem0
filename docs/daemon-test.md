# The daemon test

Two verifications, both against the real thing. The first uses stub
endpoints for the LLM and the embedder, the second starts real
llama-server processes.

| Script | What it proves | GPU |
|---|---|---|
| `scripts/verify_engine.py` | the mem0 integration: the store, the lock, the dimension count | none |
| `scripts/verify_daemon.py --embedder-only` | llama.cpp serves the embeddings the way mem0 expects | ~420 MiB |
| `scripts/verify_daemon.py --full` | the whole thing: the LLM extracts, the embedder stores, the search finds | the whole card |

**The full run needs the card.** Close everything that uses the GPU
first - the script measures the free VRAM before it starts and refuses
to run when there is not enough room. A run that starts into a full
card fails in a way that looks like a bug.

## The console command

```
cd <Pfad zum Repository> && ./.venv/Scripts/python.exe scripts/verify_daemon.py --full --model-file "<Modellverzeichnis>/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q8_0.gguf" --embedder-file "<Modellverzeichnis>/huggingface/embed-models/nomic-embed-text-v1.5.Q4_K_M.gguf" --free-mib 10000
```

## What it writes

| File | What is in it |
|---|---|
| `results/daemon-verification.log` | everything the run printed, in order |
| `results/daemon-verification.json` | the measurements: the buffers, the dimensions, the facts |
| `results/llm-server.log` | the server log of the LLM |
| `results/embedder-server.log` | the server log of the embedder |

The console and the log file get the same lines. The file is the
evidence - the console scrolls away.

## The exit codes

| Code | What it means |
|---|---|
| 0 | everything worked |
| 1 | something did not work - the PROBLEMS section says what |
| 2 | a file is missing: the model, the embedder or the runtime |
| 3 | not enough VRAM - the run did not start |

## What the full run checks

1. **Both servers come up.** The embedder first, then the LLM. Both
   stay resident at the same time - that is what the app does.
2. **Both are on the GPU.** From the server log: the device, the
   number of layers on the GPU, and the size of each buffer.
3. **The embedder answers** with 768 dimensions.
4. **The store opens** with the right dimension count and takes the
   exclusive lock.
5. **The LLM extracts** three facts from three sentences.
6. **The search finds them** again - three different questions.
7. **The store is closed** and the path is free again.

## Why the log and not nvidia-smi

The per-process reading from `nvidia-smi` returns N/A for every
process on Windows with the WDDM driver - the card is managed by the
display driver there, not by the NVIDIA driver. The server log has the
truth on every system: it says which device the server uses, how many
layers went to the GPU and how big each buffer is.

That is also why the servers always run with `--verbose`: without it
the buffer lines are not printed at all.

## The extraction benchmark

The catalogue ranks the models by extraction quality. That ranking has
to come from a measurement.

```
.venv/Scripts/python.exe scripts/benchmark_extraction.py --model-file "<Modellverzeichnis>/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF/snapshots/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q8_0.gguf" --embedder-file "<Modellverzeichnis>/huggingface/embed-models/nomic-embed-text-v1.5.Q4_K_M.gguf" --free-mib 10000
```

The corpus has 30 sentences: **24 unique facts** in four languages
(8 English, 8 German, 4 Spanish, 4 French) and **6 deliberate
duplicates** - translations of English facts that the extractor has
to recognise. They are neutral: no project, no person, nothing that
identifies the user.

**The metrics:**

| Metric | What it means |
|---|---|
| `extraction_rate` | of the new facts, how many produced a fact |
| `dedup_precision` | of the duplicates, how many produced **nothing** |
| `false_empty` | new facts that produced nothing - a real failure |
| `false_duplicate` | duplicates that produced a fact - a real failure |
| `atomicity` | one fact per entry, not a paragraph |
| `recall` | the fact is found again by a search - in the top three |
| `recall_same_language` | the question and the fact in the same language |
| `recall_cross_language` | the question in one language, the fact in another |
| `language_preservation` | the answer is in the language of the input |
| `language_normalisation` | how much was rewritten into English |

**mem0 rewrites every fact into English.** The extraction prompt is
English and the model follows it. That is not a failure - it is what
makes the cross-language recall work: a German question finds a
fact that was stored in English, because everything is in the same
language.

**Why the duplicates are there.** The first version of the corpus
had every non-English sentence as a translation of an English one
and counted an empty result as a failure. The model had deduplicated
all fourteen translations perfectly, and the metric called it a 64 %
failure rate. A service that stores the same fact three times in
three languages is worse than one that misses a fact.

The recall check needs the embedder: a fact that cannot be found
again is a fact the service does not have.

**The files:**

| File | What is in it |
|---|---|
| `results/extraction-benchmark.log` | every sentence, every extracted fact, every recall |
| `results/extraction-benchmark.json` | the summary and the server measurements |
| `results/llm-benchmark.log` | the server log of the LLM |
| `results/embedder-benchmark.log` | the server log of the embedder |

**The expected duration:** 30 sentences with the extraction and the
recall. The first extraction costs about 6 s (8224 prompt tokens),
the later ones about 2 s each because the prompt is cached. The
duplicates are faster - the model sees them and answers at once. The
recall is instant - it is a vector comparison, no LLM call.

## The catalog update

The benchmark result goes into `catalogue.json` as the
`extraction` measurement of the model. That is what makes the
ranking measured instead of assumed.

The corpus is small on purpose: it is a first measurement, not a
benchmark suite. A model with a bad `recall` is out. A model with a
good `recall` still has to be measured against the others before the
ranking means anything.

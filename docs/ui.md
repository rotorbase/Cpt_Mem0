# The window

The window is served by the daemon. It is a **client**: it never
opens the store itself. Qdrant's embedded mode allows one client per
directory, so the store has an exclusive lock - and the daemon is the
only one that may hold it.

```
cpt-mem0 ui
```

From the source tree, without installing:

```
.venv/Scripts/python.exe -m cpt_mem0.cli ui
```

With `--suspend` the daemon comes up with the models off. The API is
reachable, the memory is there, and the VRAM stays free.

## Why a browser and not a native window

One code base, every system. The window is three files - HTML, CSS
and JavaScript - that the daemon delivers from its own address. There
is no installer, no runtime that has to be shipped, and no build for
four platforms.

The look is a dark panel with glowing labels and big coloured
buttons: a Winamp window and a MIDI keyboard. The topic buttons are
the equalizer faders.

If a native window is wanted later, the same code goes into a Tauri
wrapper. Nothing in the window changes.

## The big switch

The switch starts and stops the models. **The daemon and the store
stay up.** The models are child processes, the store lives in the
daemon process.

| State | What it means |
|---|---|
| `AN` | the models are loaded, the VRAM is used, search and write work |
| `AUS` | the models are stopped, the VRAM is free, the memory is still there |

With the models off:

- the list of the memories works
- a single memory can be changed and deleted
- the search and the write refuse **with a clear message**, not with a
  connection error: `the models are suspended - the search needs them.
  Switch them on first.`

Nothing restarts the models on its own. The watcher in the daemon
does not touch a suspended runtime - the switch is the only way.

## The topic buttons

The buttons are the topics of the store, three to six of them.

**Where the facts come from:** the store, over `get_all` with an
explicitly high `top_k`. mem0's default is 20, and a store with more
than 20 memories would lose the rest silently.

**Where the grouping comes from:** the vectors that are already in
Qdrant. Asking the embedder again would cost a round trip per memory
and might produce a different geometry than the one the vectors were
written with.

The grouping is k-means over the cosine similarity - the vectors are
normalised, so the cosine similarity is the dot product. The start
configuration is the vectors that are farthest from each other, so
the result is the same on every reload and the buttons do not
shuffle.

**Where the names come from:** the local LLM. It gets the facts of a
cluster and answers with two or three words. A cluster of facts about
the build server and the backup job is called `Infrastructure`. When
the models are off, or the answer is not usable, the first fact of the
cluster stays as the name - a wrong name is worse than a long one.

**The colours** are spread over the hue circle with the same
lightness. The colour says which topic it is, not which one is more
important.

## The settings

| Setting | What it does |
|---|---|
| `MODELL` | every model that fits on the card, with its size. The choice is used for the next start. |
| `AUTOSTART (WINDOWS)` | writes a scheduled task that starts the daemon at the logon. **The models stay off** - the VRAM is free until the switch is pressed. |
| `SPRACHAUSGABE` | reads the facts out. The voice is the one Windows already has, so it costs no VRAM. |
| `WINAMP-LOOK` | the quiet look without the neon. |
| `THEMEN` | how many topic buttons, three to six. |
| `STORE LEEREN` | deletes everything. It asks first. |

## The voice

**It costs no VRAM.** The voice is SAPI, the speech API that has been
in Windows since XP. There is no model to load, nothing to download,
and nothing that takes a bite out of the card. A TTS model in
llama.cpp would be a few hundred megabytes of VRAM on a card that is
already full.

The German voice is preferred - the facts in the store are in
whatever language the user writes in, and a German voice reading
German is easier to hear than an English one. When there is no
German voice, the first one is used.

What is read out are the **facts**, not the whole answer: the first
few, each on its own line. A long list would take a minute to read,
and the user knows what they wrote.

The speaking runs in the background. The answer appears at the same
moment the voice starts, not when it ends.

On a machine without a speech engine the switch says so instead of
pretending to work.

## The routes

| Route | What it does |
|---|---|
| `GET /` | the window |
| `GET /runtime` | the state of the models and the VRAM |
| `POST /runtime/suspend` | the models off, the VRAM free |
| `POST /runtime/resume` | the models on again |
| `GET /topics?count=5` | the topic buttons with their colours |
| `GET /plan` | every model that fits on the card |
| `GET /settings` | the settings of the window |
| `POST /settings/{autostart,voice,theme,model}` | change one setting |
| `GET /status` | the state of the daemon |
| `GET /memories` | everything in the store |
| `POST /memories` | store a fact |
| `GET /search?q=...` | the search |
| `PATCH /memories/{id}` | change a memory |
| `DELETE /memories/{id}` | delete a memory |
| `DELETE /memories` | delete every memory of the user |
| `DELETE /memories/reset` | empty the store |

## The telemetry

mem0's telemetry is off, and it has to be off **before mem0 is
imported** - the flag is read once, when the module loads.

Two reasons, and the second one is not about privacy:

1. The app promises no outbound requests. With the telemetry on, mem0
   sends events to PostHog.

2. **The telemetry opens a second, global Qdrant client** under
   `~/.mem0/migrations_qdrant`. Qdrant's embedded mode allows one
   client per directory, so that path is an exclusive lock. With the
   telemetry on, Cpt Mem0 cannot start while **any other mem0
   process** on the machine holds it - including the Hermes stack.
   The error names a migrations folder and says nothing about the
   real cause.

The user's own mem0 stack keeps its telemetry as it is. This only
changes what this app does.

## What was measured

- The window is served: `/`, `/ui/style.css` and `/ui/app.js` answer
  with 200.
- The daemon starts **without a model**: no GPU is touched, the VRAM
  reading shows only what Hermes uses.
- The switch works: suspended, the search answers with 503 and the
  message about the models.
- The topics group correctly: four demo facts about the build server
  landed in one cluster, the two about the coffee machine in another.
- The count is right: ten demo facts, ten counted.
- The telemetry fix is tested against a second process that holds the
  global migrations store: the daemon starts anyway.

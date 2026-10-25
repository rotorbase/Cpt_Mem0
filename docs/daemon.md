# The daemon

One process owns the vector store, the llama-servers and the API.

**That is not a style choice.** Qdrant's embedded mode takes an
exclusive lock on its storage directory. A second client on the same
path fails with:

> `RuntimeError: Storage folder ... is already accessed by another instance
> of Qdrant client.`

Verified on qdrant-client 1.19/1.20, Windows and Linux. The daemon
serves all clients; clients never open the store themselves.

## The order of startup

The order matters, because each step depends on the previous one:

1. **The servers come up.** The engine needs their endpoints. A server
   that answers `/v1/models` is up - that does not say anything about
   the GPU.
2. **The store is opened.** The lock is taken here. It is the one
   resource that cannot be taken back, so it comes last.
3. **The API starts.** The health endpoint answers as soon as the
   store is open.

The reverse order on shutdown: the API stops accepting work, the store
is released, then the servers go down.

## The crash handling

A server that dies is restarted with an exponential backoff - 1, 2, 4,
8, 16 seconds - up to five times. After that the daemon leaves it down
and reports it. A crash loop does not become a restart loop that eats
the GPU.

## The CPU fallback

The daemon checks the VRAM of every server process. A server that
holds less than 1000 MiB is almost certainly computing on the CPU.

That failure is invisible from the outside: the port is open, the model
answers, and the speed is a sixth of what it should be. The check
catches it and says so in `status`.

## The API

| Route | Method | What it does |
|---|---|---|
| `/health` | GET | Liveness. Needs no key. |
| `/status` | GET | The state of the daemon. |
| `/memories` | GET | Everything in the store. |
| `/memories` | POST | Store a text. The LLM extracts the facts. |
| `/memories/<id>` | DELETE | Delete one memory. |
| `/search` | GET, POST | Search the store. |

The server binds `127.0.0.1` by default. A key is only needed when one
is configured; without a key the service is reachable from localhost
only. `/health` never needs a key - a health check that needs a
credential cannot be used by a health check.

## The MCP tools

| Tool | What it does |
|---|---|
| `remember` | Store a text. The LLM extracts the facts. |
| `recall` | Search the stored facts. |

The same operations as the REST endpoints. The raw CRUD is there for
the cases the two do not cover.

## The providers

mem0's `lmstudio` provider points at the local llama-servers. The name
is a leftover from mem0 - llama.cpp speaks the same OpenAI-compatible
protocol.

The dimension count has to match the embedder model: Nomic produces
768. A mismatch fails deep inside Qdrant with a shape error, so it is
checked in the configuration, not after the fact.

## The configuration

The config is YAML. Every field has a working default, and a missing
file is not an error - the defaults are a working configuration.

The writer and the reader are tested against PyYAML. Three things had
to be right for that:

- **A value can contain a colon.** `model: Qwen3.5-9B` is one value,
  not the key "Qwen3.5" with the value "9B".
- **A list of mappings keeps every item.** Not only the last one.
- **A Windows path is escaped.** A backslash is an escape character in
  YAML, so a path like `D:\Modelle` has to be written as
  `"D:\\Modelle"`. Without it PyYAML refuses the file.

## Commands

```
cpt-mem0 init --model-file <path> --embedder-file <path>
cpt-mem0 start          # detached
cpt-mem0 start --foreground
cpt-mem0 status
cpt-mem0 stop
```

`init` writes the configuration from the planner. The flags in it come
from the plan, never from hand: a wrong flag means a wrong context,
and the context is what decides whether the service can extract facts
at all.

`stop` reads the pid file, sends SIGTERM and waits. The daemon has to
release the store lock before it exits - a killed daemon leaves the
lock behind and the next start fails with a lock error.

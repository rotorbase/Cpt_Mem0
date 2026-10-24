# CLI and API Contracts

Concrete interfaces for the daemon, the CLI and the HTTP/MCP surface.
These are contracts, not implementation notes — they are what Phase 3, 4
and 5 build against.

---

## 1. CLI

```
cpt-mem0 <command> [options]
```

Global options:

| Option | Effect |
|---|---|
| `--config PATH` | Config file (default: platform config dir) |
| `--data-dir PATH` | Data directory (default: platform data dir) |
| `--json` | Machine-readable output on stdout |
| `--quiet` | Suppress progress output |
| `--verbose` | Include debug output |
| `--no-color` | Disable ANSI colour |

### 1.1 Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic failure |
| 2 | Invalid usage or configuration |
| 3 | No viable configuration for this hardware |
| 4 | A required resource is missing (binary, model, store) |
| 5 | A required resource is locked or in use by another process |
| 6 | Authentication failure |
| 7 | The service is already running (for `start`) or not running (for `stop`) |

Exit code 3 is distinct on purpose: "your card cannot run this" is a
different situation from "something broke", and scripts should be able to
tell them apart.

### 1.2 Commands

#### `cpt-mem0 init`

Interactive first-run setup. Detects the GPU, fetches the pinned
`llama.cpp` build, selects a model, writes the configuration.

```
cpt-mem0 init [--yes] [--force] [--llama-build BUILD]
```

| Flag | Effect |
|---|---|
| `--yes` | Accept all defaults, no prompts |
| `--force` | Overwrite an existing configuration |
| `--llama-build BUILD` | Pin a specific b-build instead of the default |

Steps performed:

1. Detect GPU (vendor, name, VRAM, driver).
2. Verify the CUDA runtime is available; report if not.
3. Download and unpack the pinned `llama.cpp` build (`cudart-` variant).
4. Verify the binary starts and reports a CUDA device.
5. Run the planner; print the selected configuration.
6. Download the selected model and the embedder.
7. Write `config.yaml` and the API key file.

#### `cpt-mem0 plan`

Compute and print the configuration. Starts nothing.

```
cpt-mem0 plan [--model ID] [--quant QUANT] [--embedder auto|gpu|cpu]
              [--context N] [--json]
```

Output (text mode):

```
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

Output (JSON mode) — the machine-readable contract:

```json
{
  "gpu": {"name": "RTX 3060", "vendor": "nvidia", "total_mib": 12288},
  "llm": {
    "model": "Qwen3.5-9B", "quantisation": "Q8_0",
    "file": "Qwen3.5-9B-Q8_0.gguf", "size_mib": 9086,
    "weights_vram_mib": 8087, "kv_vram_mib": 1114,
    "kv_bytes_per_token": 17408, "cache_type": "q8_0",
    "n_ctx": 62976, "n_ctx_slot": 62976, "parallel": 2,
    "rank_basis": "unmeasured",
    "flags": ["-c", "62976", "--parallel", "2", "-ngl", "99",
              "--fit", "off", "--flash-attn", "on",
              "--reasoning-format", "none",
              "--no-context-shift", "--kv-unified",
              "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
  },
  "embedder": {"backend": "llamacpp"},
  "budget": {"total_mib": 12288, "used_mib": 10348, "reserved_mib": 1200,
             "headroom_mib": 737, "free_mib": 1940},
  "viable": true
}
```

When nothing fits, `viable` is `false` and `diagnostics` explains why:

```json
{
  "viable": false,
  "diagnostics": {
    "reason": "context_floor_not_met",
    "required_mib": 3406, "available_mib": 4096, "shortfall_mib": 0,
    "detail": {"model": "Qwen3-4B-Instruct-2507", "n_ctx": 15104, "n_ctx_slot": 15104,
               "required_total": 21000, "required_slot": 12288},
    "calibration_required": {},
    "options": [
      "use a card with more VRAM",
      "set planner.context_floor_total lower if your mem0 configuration uses a shorter extraction prompt",
      "set planner.allow_degraded: true to run anyway, accepting that fact extraction may fail"
    ]
  }
}
```

#### `cpt-mem0 start` / `stop` / `restart`

```
cpt-mem0 start [--foreground] [--wait SECONDS]
cpt-mem0 stop [--timeout SECONDS]
cpt-mem0 restart
```

`start` returns once the daemon reports healthy. `--foreground` runs the
daemon in the terminal (for debugging; logs go to stderr). `--wait` bounds
how long `start` waits for health before giving up.

#### `cpt-mem0 status`

```
cpt-mem0 status [--json]
```

Reports: daemon state, uptime, PID, GPU utilisation, VRAM used by each
child process, store size, memory count, last error, active model and
context.

#### `cpt-mem0 update`

```
cpt-mem0 update [--check] [--build BUILD] [--rollback] [--json]
```

| Flag | Effect |
|---|---|
| `--check` | Report what would change, apply nothing |
| `--build BUILD` | Pin a specific b-build |
| `--rollback` | Return to the previous build |

Never runs without an explicit command. Never updates models or the
store.

#### `cpt-mem0 calibrate`

```
cpt-mem0 calibrate [--model ID] [--json]
```

Derives KV bytes/token by loading the model at two context sizes and
reading the server log. Caches the result in the local overlay.

#### `cpt-mem0 models`

```
cpt-mem0 models list [--json] [--fits]
cpt-mem0 models add <repo>[:<quant>] [--json]
cpt-mem0 models remove <id>
cpt-mem0 models path <id>
```

`--fits` filters the catalogue to models that fit the current GPU.

#### `cpt-mem0 migrate`

```
cpt-mem0 migrate --from pgvector [--source DSN] [--dry-run]
```

Reads from an external PostgreSQL/pgvector store and writes into the
embedded Qdrant store. Requires matching embedding dimensions; the command
refuses if they differ.

---

## 2. HTTP API

Base URL: `http://127.0.0.1:8765`

Authentication: `Authorization: Bearer <key>`. Keys are per-client and
stored in `keys/clients.json` (mode 0600).

All responses are JSON. Errors use the envelope in §2.3.

### 2.1 Memory endpoints

#### `POST /v1/memories`

Extract and store facts from a conversation.

```json
{
  "messages": [
    {"role": "user", "content": "Ich nutze eine RTX 3060 mit 12 GB."},
    {"role": "assistant", "content": "Verstanden."}
  ],
  "user_id": "default",
  "agent_id": "cpt-mem0",
  "infer": true,
  "metadata": {"source": "chat"}
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `messages` | array | required | Conversation turns |
| `user_id` | string | key's scope | Memory owner |
| `agent_id` | string | `"cpt-mem0"` | Sub-scope within the owner |
| `infer` | bool | `true` | Extract facts (`false` = store verbatim) |
| `metadata` | object | `{}` | Attached to every stored fact |

Response:

```json
{
  "results": [
    {"id": "mem_01H...", "event": "ADD",
     "memory": "User utilizes an NVIDIA RTX 3060 with 12 GB of VRAM",
     "user_id": "default", "agent_id": "cpt-mem0", "hash": "..."}
  ],
  "metadata": {"elapsed_ms": 6891, "model": "Qwen3.5-9B", "inferred": 2}
}
```

`event` is one of `ADD`, `UPDATE`, `DELETE`, `NONE` — the same vocabulary
mem0 uses.

#### `POST /v1/memories/search`

```json
{"query": "which gpu does the user have", "user_id": "default",
 "limit": 10, "threshold": 0.2}
```

Response:

```json
{
  "results": [
    {"id": "mem_01H...", "memory": "User utilizes an NVIDIA RTX 3060...",
     "score": 0.4869, "user_id": "default", "metadata": {"created": "..."}}
  ],
  "metadata": {"elapsed_ms": 31, "query_embedding_ms": 21}
}
```

Note: `mem0` 2.2.1 requires filters inside the search payload
(`filters={"user_id": ...}`), not as a top-level parameter. The API
accepts `user_id` at the top level for convenience and translates it.

#### `GET /v1/memories`

```json
{"user_id": "default", "limit": 100, "offset": 0}
```

#### `GET /v1/memories/{id}` · `PATCH /v1/memories/{id}` · `DELETE /v1/memories/{id}`

`PATCH` accepts `{"memory": "new text"}` and re-embedds.

#### `DELETE /v1/memories`

Bulk delete, `{"user_id": "default", "ids": [...]}` or
`{"user_id": "default", "all": true}`.

### 2.2 Service endpoints

#### `GET /v1/health`

```json
{
  "status": "healthy",
  "checks": {
    "daemon": "ok",
    "llm": {"status": "ok", "port": 50260, "model": "Qwen3.5-9B",
            "vram_mib": 4861, "inference_gpu": true},
    "embedder": {"status": "ok", "port": 50261, "vram_mib": 78},
    "store": {"status": "ok", "path": "...", "count": 128,
              "size_mib": 1.4}
  }
}
```

`inference_gpu` is **measured**, not inferred: the daemon compares VRAM
against the known weight footprint. A model that silently fell back to CPU
reports `inference_gpu: false` and `status: degraded`.

#### `GET /v1/status`

The full runtime state — same content as `cpt-mem0 status --json`.

#### `GET /v1/plan`

The planner result, same schema as `cpt-mem0 plan --json`. Lets a client
show the user what the hardware can do before committing to a setup.

#### `POST /v1/chat/completions`

OpenAI-compatible passthrough to the local LLM. This is what lets the
service double as a local inference endpoint.

Standard OpenAI request and response shape. Additional fields:

| Field | Meaning |
|---|---|
| `memories` | Optional. If present, retrieved facts are injected into the system prompt. |

With `memories` set, the daemon retrieves top-k for the conversation
content, formats them as a `[Persistent context]` block, and prepends it.
The number of retrieved facts and the similarity threshold come from the
client, not the daemon.

### 2.3 Error envelope

Every error response uses the same shape:

```json
{
  "error": {
    "code": "context_floor_not_met",
    "message": "The selected configuration cannot extract facts: 2178 tokens available, 10500 required.",
    "detail": {"required_tokens": 10500, "available_tokens": 2178},
    "request_id": "req_01H..."
  }
}
```

| HTTP | `code` | Meaning |
|---|---|---|
| 400 | `invalid_request` | Malformed body or missing field |
| 401 | `unauthorized` | Missing or invalid API key |
| 403 | `forbidden` | Key is not scoped to this `user_id` |
| 404 | `not_found` | No such memory or route |
| 409 | `store_locked` | Another process holds the store lock |
| 422 | `context_floor_not_met` | Configuration cannot perform the requested operation |
| 429 | `rate_limited` | Client exceeded its rate limit |
| 500 | `internal_error` | Unexpected failure; `request_id` correlates with the log |
| 503 | `model_unavailable` | The LLM or embedder is not responding |

### 2.4 Authentication and scoping

Key file format (`keys/clients.json`):

```json
{
  "clients": [
    {"name": "local", "key": "cpt_…", "user_id": "default",
     "created": "2026-10-24T10:00:00Z", "last_used": "..."}
  ]
}
```

- A key without an explicit `user_id` may access any `user_id`.
- A key with a `user_id` may only access that one; requests for another
  return 403.
- Keys are compared in constant time.
- Keys never appear in logs. The daemon logs the client *name*, never the
  key.

CLI to manage keys:

```
cpt-mem0 keys list
cpt-mem0 keys add --name NAME [--user-id ID]
cpt-mem0 keys remove NAME
```

---

## 3. MCP surface

The MCP server runs on the same port as the REST API, under `/mcp`.

### 3.1 Raw tools

| Tool | Input | Output |
|---|---|---|
| `memory_add` | `{messages, user_id?, metadata?, infer?}` | `{results, metadata}` |
| `memory_search` | `{query, user_id?, limit?, threshold?}` | `{results, metadata}` |
| `memory_list` | `{user_id?, limit?, offset?}` | `{results, metadata}` |
| `memory_update` | `{id, memory}` | `{id, event, memory}` |
| `memory_delete` | `{id}` | `{deleted: true}` |

### 3.2 Convenience tools

| Tool | Input | Behaviour |
|---|---|---|
| `remember` | `{content, user_id?}` | Adds a single statement. Internally: `memory_add` with one user message and `infer: true`. |
| `recall` | `{query, user_id?, limit?}` | Searches and returns the facts as a formatted bullet list, ready to paste into a prompt. Internally: `memory_search`. |

`recall` returns a human-readable string block, not a JSON array:

```
- User utilizes an NVIDIA RTX 3060 with 12 GB of VRAM (score 0.49)
- The user's project is named Hermes-VTOL (score 0.42)
```

This is the shape an agent wants to inject into a prompt. The raw tools
stay available for anything that needs the structured data.

### 3.3 MCP resources

| URI | Content |
|---|---|
| `cpt-mem0://status` | Daemon state, VRAM, model |
| `cpt-mem0://plan` | The planner result for this hardware |
| `cpt-mem0://memories/{user_id}` | All memories for a user |

---

## 4. Background processing

`memory_add` with `infer: true` costs one LLM round-trip (~5–8 s
measured). To keep clients responsive:

- `POST /v1/memories` accepts `"async": true`.
- With `async: true` the response is `202 Accepted` with a job id.
- `GET /v1/jobs/{id}` reports progress.
- The extraction runs in a background thread; the daemon stays responsive.

The default is synchronous, because a client that asks to store something
usually wants to know it was stored.

---

## 5. Rate limiting

Per key, per minute. Defaults:

| Setting | Value |
|---|---|
| Requests | 120/min |
| `memory_add` (LLM-backed) | 12/min |
| `memory_search` | 120/min |

Exceeding a limit returns 429 with `Retry-After`. The limit for
LLM-backed operations is lower because each one occupies the GPU.

Limits are configurable per client in `keys/clients.json`.

---

## 6. Logging

Structured JSON lines to `logs/daemon.log`.

| Field | Content |
|---|---|
| `ts` | ISO 8601 timestamp |
| `level` | `debug` / `info` / `warn` / `error` |
| `event` | Event name, e.g. `llm.request` |
| `request_id` | Correlates with the error envelope |
| `client` | Client **name**, never the key |
| `duration_ms` | For timed operations |
| `model` | Active model |
| `vram_mib` | VRAM after the operation |

Secrets are never logged: API keys, the internal llama-server API key and
the store path are redacted.

---

## 7. Compatibility surface

The mem0 engine is wrapped, not exposed. The daemon speaks HTTP and MCP,
and internally translates to whatever mem0's API requires — including
version differences such as `filters={"user_id": ...}` versus a top-level
`user_id` parameter.

This means a mem0 upgrade cannot break a client's contract. It does mean
the daemon must track mem0's API changes; the pinned dependency version in
`requirements.txt` is part of the release contract.

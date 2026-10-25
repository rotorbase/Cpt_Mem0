"""Verify the engine against the real mem0.

The tests use a fake mem0, which proves the code paths but not the
integration. This script runs the real mem0 2.2.1 with the real
Qdrant embedded store, against a stub LLM and a stub embedder.

**No GPU, no model, no VRAM.** The two OpenAI-compatible endpoints are
stubs that answer with fixed data. That is enough to prove:

- the mem0 configuration the daemon builds is accepted
- the Qdrant store opens, locks and searches
- the extraction flow works end to end
- the vectors that come back have the dimension count the store was
  created with

What it does **not** prove is that llama.cpp serves these endpoints
the way the stubs do. That needs the real hardware.

Usage:
    python scripts/verify_engine.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.engine import Engine, llama_endpoint  # noqa: E402

#: The dimension count of the embedder model.
DIMENSIONS = 768


class StubHandler(BaseHTTPRequestHandler):
    """A stub for the two OpenAI-compatible endpoints.

    The embedder returns a deterministic vector, so the search has
    something to match against. The LLM returns a fact in the format
    mem0 asks for.
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/models"):
            self._send({"data": [{"id": "stub"}]})
        else:
            self._send({})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8"))
        except ValueError:
            request = {}

        if "embeddings" in self.path:
            inputs = request.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            data = []
            for index, text in enumerate(inputs):
                data.append({
                    "index": index,
                    "embedding": _vector(str(text)),
                })
            self._send({"data": data, "model": request.get("model", "stub")})
            return

        if "chat/completions" in self.path:
            self._send({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": _fact()},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })
            return

        if "completions" in self.path:
            self._send({"content": _fact()})
            return

        self._send({})


def _vector(text: str) -> list[float]:
    """A deterministic vector for a text.

    The same text always gives the same vector, so a search finds the
    memory that was stored from the same text.
    """
    seed = sum(ord(char) for char in text)
    return [((seed * (index + 1)) % 1000) / 1000.0
            for index in range(DIMENSIONS)]


def _fact() -> str:
    """The fact the stub LLM extracts.

    The format is measured on mem0 2.2.1: the answer is a JSON
    object with a "memory" key, and the value is a list of objects
    with a "text" key. A list of plain strings raises
    ``AttributeError: 'str' object has no attribute 'get'`` deep
    inside mem0.

    The extraction prompt asks for these fields:

        {"memory": [{"id": "0", "text": "the fact"}]}
    """
    return json.dumps({
        "memory": [{
            "id": "0",
            "text": "The user rides a bicycle to the office every day.",
            "event": "ADD",
            "memory": "The user rides a bicycle to the office every day.",
        }],
    })


def main() -> int:
    print("=== the engine against the real mem0 ===", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"  stub endpoint on port {port}", flush=True)

    workdir = Path(tempfile.mkdtemp(prefix="cpt-mem0-engine-"))
    store_path = workdir / "store"
    problems: list[str] = []

    try:
        endpoint = llama_endpoint(port, "stub-model")
        engine = Engine(
            llm=endpoint,
            embedder=endpoint,
            store_path=store_path,
            dimensions=DIMENSIONS,
            user_id="willow")

        print("\n--- opening the store ---", flush=True)
        engine.open()
        print("  the store is open", flush=True)

        # The store must be locked. A second client on the same path
        # has to fail.
        print("\n--- the exclusive lock ---", flush=True)
        lock_check = subprocess.run(
            [sys.executable, "-c",
             "import sys;"
             "from qdrant_client import QdrantClient;"
             f"QdrantClient(path=r'{store_path}')"],
            capture_output=True, text=True, timeout=60)
        if lock_check.returncode == 0:
            problems.append(
                "a second client opened the same store - the lock does "
                "not work")
            print("  FEHLER: the second client got through", flush=True)
        else:
            print("  OK: the second client was refused", flush=True)

        # A write: the LLM extracts the facts.
        print("\n--- remember ---", flush=True)
        result = engine.remember(
            "The user rides a bicycle to the office every day.")
        stored = result.get("results") or []
        print(f"  stored {len(stored)} facts", flush=True)
        for item in stored:
            memory = item.get("memory") if isinstance(item, dict) else item
            print(f"    {memory}", flush=True)
        if not stored:
            problems.append("the write stored nothing")

        # A search: the embedder and the vector comparison.
        print("\n--- recall ---", flush=True)
        found = engine.recall("How does the user get to work?")
        print(f"  {len(found)} results", flush=True)
        for item in found:
            memory = item.get("memory") if isinstance(item, dict) else item
            print(f"    {memory}", flush=True)
        if not found:
            problems.append("the search found nothing")

        print("\n--- all ---", flush=True)
        everything = engine.all()
        print(f"  {len(everything)} memories in the store", flush=True)

        # The dimension count has to be the one the store was created
        # with. A mismatch would fail later, in the search.
        print("\n--- the dimension count ---", flush=True)
        client = engine._memory.vector_store.client
        collections = client.get_collections().collections
        for collection in collections:
            info = client.get_collection(collection.name)
            vectors = info.config.params.vectors
            size = getattr(vectors, "size", None)
            print(f"  {collection.name}: {size} dimensions", flush=True)
            if size != DIMENSIONS:
                problems.append(
                    f"the collection has {size} dimensions, expected "
                    f"{DIMENSIONS}")

        print("\n--- closing the store ---", flush=True)
        engine.close()
        print("  the store is closed", flush=True)

        # After closing, the path must be free again.
        reopened = subprocess.run(
            [sys.executable, "-c",
             "from qdrant_client import QdrantClient;"
             f"c=QdrantClient(path=r'{store_path}'); c.close()"],
            capture_output=True, text=True, timeout=60)
        if reopened.returncode != 0:
            problems.append(
                "the store path is still locked after the close - the "
                "next start would fail")
            print("  FEHLER: the path is still locked", flush=True)
        else:
            print("  OK: the path is free again", flush=True)

    except Exception as exc:
        import traceback
        print(f"\nERROR: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        problems.append(f"{type(exc).__name__}: {exc}")
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("The engine works against the real mem0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

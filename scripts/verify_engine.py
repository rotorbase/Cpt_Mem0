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


#: The texts the stub recognises in the extraction prompt.
#:
#: The stub has to answer with the text that was stored, and the
#: extraction prompt wraps the input into a longer instruction. The
#: stub therefore looks for the known texts in the prompt and answers
#: with the one it finds.
#:
#: Without this the stub always answers with the same fact, mem0
#: deduplicates all the calls, and a test of "everything is findable"
#: sees one memory instead of twenty-five.
KNOWN_INPUTS: list[str] = []


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
            # Only the user message carries the new input. The system
            # message is the extraction instruction, and it has a
            # "## New Messages" section of its own - with the
            # description of the roles.
            prompt = ""
            for message in request.get("messages", []):
                if message.get("role") == "user":
                    prompt = str(message.get("content", ""))
                    break
            self._send({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": _fact(prompt)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })
            return

        if "completions" in self.path:
            self._send({"content": _fact(str(request))})
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


def _new_messages(prompt: str) -> str:
    """The new message from the extraction prompt.

    The prompt has a section with the new messages:

        ## New Messages
        user: Fact number 7 about the system.

        ## Observation Date
        2026-10-25

    **Only that section.** The rest of the prompt grows with every
    stored memory - the "Existing Memories" section lists all of
    them. A search over the whole prompt finds the first memory that
    happens to be known, not the new message, and the stub answers
    with a fact that was stored long ago.
    """
    marker = "## New Messages"
    index = prompt.find(marker)
    if index < 0:
        return ""

    rest = prompt[index + len(marker):]
    # The section ends at the next heading.
    end = rest.find("\n## ")
    if end >= 0:
        rest = rest[:end]

    lines = []
    for line in rest.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("user:"):
            line = line[len("user:"):].strip()
        elif line.startswith("assistant:"):
            line = line[len("assistant:"):].strip()
        if line:
            lines.append(line)
    return " ".join(lines)


def _fact(prompt: str = "") -> str:
    """The fact the stub LLM extracts.

    The format is measured on mem0 2.2.1: the answer is a JSON
    object with a "memory" key, and the value is a list of objects
    with a "text" key. A list of plain strings raises
    ``AttributeError: 'str' object has no attribute 'get'`` deep
    inside mem0.

    The extraction prompt asks for these fields:

        {"memory": [{"id": "0", "text": "the fact"}]}

    **The fact is the new message.** The stub reads the "## New
    Messages" section and answers with it. Without that the stub
    always answers with the same fact, mem0 deduplicates every call
    through its hash check, and a test of "everything is findable"
    sees one memory instead of twenty-five.
    """
    text = _new_messages(prompt)

    if not text:
        # The section was not found. Fall back to the known inputs.
        for known in KNOWN_INPUTS:
            if known in prompt:
                text = known
                break

    if not text:
        text = "The user rides a bicycle to the office every day."

    return json.dumps({
        "memory": [{
            "id": "0",
            "text": text,
            "event": "ADD",
            "memory": text,
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

        # Everything has to be findable and deletable. That is the
        # requirement for the finished app.
        print("\n--- findable: everything is listed ---", flush=True)
        KNOWN_INPUTS.clear()
        KNOWN_INPUTS.append("The user rides a bicycle to the office every day.")
        for index in range(25):
            KNOWN_INPUTS.append(f"Fact number {index} about the system.")
        for index in range(25):
            engine.remember(f"Fact number {index} about the system.")
        everything = engine.all()
        print(f"  {len(everything)} memories in the store", flush=True)
        if len(everything) < 26:
            problems.append(
                f"the list returned {len(everything)} memories, 26 were "
                f"stored - mem0's default top_k of 20 truncates the list "
                f"silently")
        else:
            print("  the list is not truncated", flush=True)

        print("\n--- findable: every memory has an id ---", flush=True)
        without_id = [item for item in everything
                      if not (item.get("id") or item.get("memory_id"))]
        if without_id:
            problems.append(
                f"{len(without_id)} memories have no id - they cannot be "
                f"updated or deleted")
        else:
            print("  every memory carries an id", flush=True)

        print("\n--- deletable: one memory ---", flush=True)
        first = everything[0]
        memory_id = first.get("id") or first.get("memory_id")
        engine.delete(memory_id)
        after = engine.all()
        print(f"  {len(after)} memories left", flush=True)
        if len(after) != len(everything) - 1:
            problems.append(
                f"the delete left {len(after)} memories, "
                f"{len(everything) - 1} were expected")

        print("\n--- deletable: one memory, changed ---", flush=True)
        second = after[0]
        second_id = second.get("id") or second.get("memory_id")
        engine.update(second_id, "The changed fact.")
        changed = [item for item in engine.all()
                    if (item.get("id") or item.get("memory_id")) == second_id]
        if not changed or "changed fact" not in str(changed[0].get("memory")):
            problems.append(
                "the update did not change the memory")
        else:
            print("  the memory was changed in place", flush=True)

        print("\n--- deletable: all of one user ---", flush=True)
        engine.delete_all("willow")
        left = engine.all()
        print(f"  {len(left)} memories left", flush=True)
        if left:
            problems.append(
                f"the delete_all left {len(left)} memories behind")

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

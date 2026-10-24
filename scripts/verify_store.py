"""The full circle: embed, store, restart, recall.

This is the end-to-end check of the part that matters. It starts the
embedding server, turns a few sentences into vectors, writes them to
a store, then **starts the server again** and looks for them with a
different query.

The restart is the point. A vector store that lives in memory proves
nothing - it has to survive the process that wrote it.

Run this only with a free card. It loads the embedding model into VRAM.

Usage:
    python scripts/verify_store.py --embedder-file <path> --port 8800
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.store import VectorStore, cosine_similarity  # noqa: E402

#: The sentences that go in. They are about different things, so a
#: query has to actually discriminate to find the right one.
FACTS = [
    "The user flies a VTOL aircraft with a 3.2 m wingspan.",
    "The development board uses an Intel chipset with a faulty NIC.",
    "The build server runs on a machine with 64 GB of RAM.",
    "The coffee machine in the office is a fully automatic model.",
]

#: The query for each fact. It is a different wording, so the match
#: has to come from the meaning, not from the words.
QUERIES = [
    ("What aircraft does the user fly?",
     "The user flies a VTOL aircraft with a 3.2 m wingspan."),
    ("Which network card is broken?",
     "The development board uses an Intel chipset with a faulty NIC."),
    ("How much memory does the build server have?",
     "The build server runs on a machine with 64 GB of RAM."),
    ("What kind of coffee machine is in the office?",
     "The coffee machine in the office is a fully automatic model."),
]


def find_llama_server() -> str | None:
    import shutil
    exe = shutil.which("llama-server")
    if exe:
        return exe
    home = Path.home()
    for pattern in (
        ".unsloth/llama.cpp/build/bin/*/llama-server.exe",
        "llama.cpp/build/bin/*/llama-server.exe",
    ):
        for path in sorted(home.glob(pattern)):
            if path.is_file():
                return str(path)
    return None


def embed(exe: str, model: Path, port: int, texts: list[str],
          timeout: int = 180) -> list[list[float]]:
    """Start the embedding server, embed the texts, stop the server.

    The server is started here rather than kept running, because the
    second phase needs a fresh process: that is what proves the store
    survives.
    """
    flags = [
        "--parallel", "1", "-c", "2048", "-ngl", "99",
        "--embeddings", "--pooling", "mean", "--kv-unified",
        "--verbose",
    ]
    cmd = [exe, "-m", str(model), "--port", str(port), *flags]
    print("  $", " ".join(cmd), flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    log: list[str] = []

    def reader() -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    log.append(line)
        except (OSError, ValueError):
            pass

    threading.Thread(target=reader, daemon=True).start()

    try:
        # Wait until the server answers.
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                    resp.read(1)
                break
            except Exception:
                if proc.poll() is not None:
                    print("the server exited:", file=sys.stderr)
                    print("".join(log[-20:]), file=sys.stderr)
                    return []
                time.sleep(0.5)
        else:
            print("the server did not come up", file=sys.stderr)
            return []

        vectors = []
        for text in texts:
            vector = embed_one(port, text)
            if vector is None:
                return []
            vectors.append(vector)
            print(f"  embedded {len(vector)} dimensions: "
                  f"{text[:48]!r}", flush=True)
        return vectors
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


def embed_one(port: int, text: str, timeout: float = 60.0
              ) -> list[float] | None:
    """One embedding request against a running server."""
    payload = json.dumps({"input": text, "model": "nomic-embed"}).encode(
        "utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:
            pass
        print(f"  HTTP {exc.code}: {raw[:300]}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    items = body.get("data") or []
    if not items:
        print("  the server returned no embedding", file=sys.stderr)
        return None
    return [float(value) for value in items[0]["embedding"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder-file",
                        default=str(Path.home() / ".cache/huggingface"
                                    / "embed-models"
                                    / "nomic-embed-text-v1.5.Q4_K_M.gguf"))
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--store", default=str(ROOT / "results" / "store.json"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    exe = find_llama_server()
    if not exe:
        print("llama-server not found", file=sys.stderr)
        return 2
    model = Path(args.embedder_file)
    if not model.is_file():
        print(f"embedder file not found: {model}", file=sys.stderr)
        return 2

    store_path = Path(args.store)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        store_path.unlink()

    problems: list[str] = []

    # ---- phase 1: embed and store -------------------------------------
    print("\n--- phase 1: embed and store ---", flush=True)
    vectors = embed(exe, model, args.port, FACTS, timeout=args.timeout)
    if not vectors:
        print("\nphase 1 failed: no vectors", file=sys.stderr)
        return 1

    store = VectorStore(store_path)
    store.add_many(zip(FACTS, vectors))
    store.save()
    print(f"\n  stored {store.count} facts in {store_path}", flush=True)
    print(f"  dimension: {store.dimension}", flush=True)

    # The vectors have to be usable as they are. llama.cpp returns
    # normalised vectors, and a query vector has to live in the same
    # space, so the similarity between a fact and its own embedding is
    # 1.0.
    for text, vector in zip(FACTS, vectors):
        score = cosine_similarity(vector, vector)
        if abs(score - 1.0) > 0.001:
            problems.append(f"the vector of {text[:32]!r} is not unit "
                            f"length: cosine to itself is {score:.4f}")

    # ---- phase 2: a new process reads it back ------------------------
    print("\n--- phase 2: restart and recall ---", flush=True)
    reloaded = VectorStore(store_path).load()
    print(f"  reloaded {reloaded.count} facts, dimension "
          f"{reloaded.dimension}", flush=True)

    if reloaded.count != len(FACTS):
        problems.append(
            f"the store has {reloaded.count} facts after the reload, "
            f"{len(FACTS)} were stored")
    if reloaded.dimension != store.dimension:
        problems.append(
            f"the dimension changed on reload: {reloaded.dimension} != "
            f"{store.dimension}")

    # The queries are embedded by a second, freshly started server. The
    # store was written by the first one.
    queries = [query for query, _ in QUERIES]
    expected = [expected for _, expected in QUERIES]
    query_vectors = embed(exe, model, args.port, queries,
                           timeout=args.timeout)
    if not query_vectors:
        print("\nphase 2 failed: no query vectors", file=sys.stderr)
        return 1

    print()
    for question, query_vector, wanted in zip(queries, query_vectors,
                                              expected):
        results = reloaded.search(query_vector, limit=len(FACTS))
        if not results:
            problems.append(f"the query {question!r} found nothing")
            continue
        best_text, best_score = results[0]
        print(f"  {question}")
        print(f"    best: {best_text[:60]!r}  (score {best_score:.4f})")
        for text, score in results[1:3]:
            print(f"    also: {text[:60]!r}  (score {score:.4f})")
        if best_text != wanted:
            problems.append(
                f"the query {question!r} returned {best_text[:40]!r}, "
                f"expected {wanted[:40]!r}")
        # The right answer has to be clearly ahead of the others.
        #
        # Measured with nomic-embed-text-v1.5 on the four facts above:
        # the smallest gap between the best and the second best was
        # 0.1341. The threshold sits below that, but it is not trivial.
        if len(results) > 1 and best_score - results[1][1] < 0.08:
            problems.append(
                f"the query {question!r} is not decisive: "
                f"{best_score:.4f} against {results[1][1]:.4f}")

    print()
    if problems:
        print("MISMATCH:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("The store survives a restart and the recall finds the right "
          "fact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The memory engine.

The engine wraps mem0 so that the daemon does not have to know how
the pieces fit together: it gets an LLM, an embedder and a store, and
it gets ``remember`` and ``recall``.

**The providers are OpenAI-compatible endpoints.** llama.cpp speaks the
OpenAI protocol, so mem0's `lmstudio` provider points at it - the name
is a leftover from mem0, the protocol is the same.

**The dimensions have to match the model.** Nomic produces 768
dimensions. If the store was created with a different count, the
search fails with a shape error deep inside qdrant. The dimension
count is therefore checked before the store is opened, not after.

**The store is opened exactly once.** Qdrant's embedded mode takes an
exclusive lock on its directory. A second client on the same path
fails. The daemon is the only client; the API serves everybody.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: How many memories a list call returns.
#:
#: mem0's own default is 20. That is a silent truncation: a store
#: with more memories would return only the first 20, and the rest
#: would be unfindable. There is no upper bound in mem0, only a
#: lower one.
LIST_LIMIT = 100_000


class EngineError(RuntimeError):
    """The engine cannot do its job.

    The message names the cause, because the cause is usually a
    misconfiguration that the user can fix.
    """


@dataclass
class Endpoint:
    """An OpenAI-compatible endpoint."""

    base_url: str
    model: str
    api_key: str = "cpt-mem0"

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"


def llama_endpoint(port: int, model: str) -> Endpoint:
    """The endpoint of a llama-server."""
    return Endpoint(f"http://127.0.0.1:{port}/v1", model)


def probe(endpoint: Endpoint, timeout: float = 2.0) -> bool:
    """Whether the endpoint answers.

    A server that answers /v1/models proves that it is up. It does not
    prove that the model is on the GPU - only the VRAM does that.
    """
    try:
        request = urllib.request.Request(
            endpoint.url("/models"),
            headers={"Authorization": f"Bearer {endpoint.api_key}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(1)
            return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def embed(endpoint: Endpoint, text: str,
          timeout: float = 30.0) -> list[float]:
    """The embedding of a text.

    This is the path the embedder has to serve. The vector is not
    normalised: llama.cpp returns the raw pooled vector, and the store
    has to see the same geometry on every write and every search.
    """
    payload = json.dumps({"input": [text], "model": endpoint.model}).encode()
    request = urllib.request.Request(
        endpoint.url("/embeddings"), data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {endpoint.api_key}"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return list(body["data"][0]["embedding"])


def memory_config(llm: Endpoint, embedder: Endpoint, store_path: Path,
                  collection: str = "mem0", dimensions: int = 768,
                  history_db_path: Path | None = None) -> dict[str, Any]:
    """The mem0 configuration.

    It is a plain dictionary because mem0 takes one. The parts that
    matter:

    - the LLM and the embedder point at the local llama-servers
    - the vector store is Qdrant in embedded mode, with the dimension
      count of the embedder model
    - the history database lives next to the store, so a backup of the
      data directory is complete
    """
    if store_path is None:
        raise EngineError("the store path is not set")

    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": dimensions,
                "path": str(store_path),
                "on_disk": True,
            },
        },
        "llm": {
            "provider": "lmstudio",
            "config": {
                "model": llm.model,
                "lmstudio_base_url": llm.base_url,
                "api_key": llm.api_key,
                "temperature": 0.0,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "lmstudio",
            "config": {
                "model": embedder.model,
                "lmstudio_base_url": embedder.base_url,
                "api_key": embedder.api_key,
                "embedding_dims": dimensions,
            },
        },
        "history_db_path": str(
            history_db_path
            if history_db_path is not None
            else store_path.parent / "history.db"),
    }


class Engine:
    """The memory engine.

    It owns the mem0 instance and the store lock. Everything that
    touches the store goes through it.
    """

    def __init__(self, llm: Endpoint, embedder: Endpoint,
                 store_path: Path, collection: str = "mem0",
                 dimensions: int = 768,
                 user_id: str = "local") -> None:
        self.llm = llm
        self.embedder = embedder
        self.store_path = store_path
        self.collection = collection
        self.dimensions = dimensions
        self.user_id = user_id
        self._memory: Any = None

    def open(self) -> None:
        """Open the store and build the mem0 instance.

        The store takes an exclusive lock. A second instance on the
        same path fails, and the error names the lock - that message
        is the only clue the user gets.
        """
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise EngineError(
                "mem0 is not installed. The engine needs the 'mem0ai' "
                "package - see the installation instructions.") from exc

        if not self.store_path.parent.exists():
            self.store_path.parent.mkdir(parents=True, exist_ok=True)

        config = memory_config(self.llm, self.embedder, self.store_path,
                               self.collection, self.dimensions)
        try:
            self._memory = Memory.from_config(config)
        except Exception as exc:
            message = str(exc)
            if "already accessed by another instance" in message:
                raise EngineError(
                    f"the store at {self.store_path} is locked by another "
                    f"process. Qdrant's embedded mode allows one client "
                    f"per directory - stop the other instance or give "
                    f"this one a different store path.") from exc
            raise EngineError(
                f"the store at {self.store_path} could not be opened: "
                f"{message}") from exc

    def close(self) -> None:
        """Release the store."""
        memory = self._memory
        self._memory = None
        if memory is None:
            return
        # mem0 does not expose a close on the memory object; the
        # vector store client does.
        client = getattr(memory, "vector_store", None)
        closer = getattr(client, "client", None)
        if closer is not None:
            try:
                closer.close()
            except Exception:
                pass

    @property
    def is_open(self) -> bool:
        return self._memory is not None

    def _require_open(self) -> Any:
        if self._memory is None:
            raise EngineError("the engine is not open")
        return self._memory

    def remember(self, text: str, user_id: str | None = None,
                 metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store a text and return what mem0 extracted.

        The LLM does the extraction: it turns a sentence into facts.
        That is why a write costs an LLM call and a search does not.
        """
        memory = self._require_open()
        options: dict[str, Any] = {
            "user_id": user_id or self.user_id,
            "infer": True,
        }
        if metadata:
            options["metadata"] = metadata
        try:
            result = memory.add(text, **options)
        except TypeError:
            # Older mem0 versions do not take the keyword arguments the
            # same way. Try the positional form.
            result = memory.add(text, user_id or self.user_id)
        return _as_dict(result)

    def recall(self, query: str, user_id: str | None = None,
               limit: int = 5) -> list[dict[str, Any]]:
        """Search the store.

        The search uses the filter form: ``filters={"user_id": ...}``.
        Passing ``user_id`` directly raises a TypeError in mem0 2.2.1.

        The parameter is ``top_k``, not ``limit`` - measured on the
        mem0 2.2.1 signature. Passing ``limit=`` raises a TypeError,
        and the fallback would put the user id where ``top_k``
        belongs.
        """
        memory = self._require_open()
        try:
            result = memory.search(
                query,
                filters={"user_id": user_id or self.user_id},
                top_k=limit)
        except TypeError:
            result = memory.search(
                query,
                filters={"user_id": user_id or self.user_id},
                limit=limit)
        return _as_list(result)

    def all(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Everything in the store for this user.

        **The count is not capped.** mem0's ``get_all`` has a default
        of ``top_k=20`` - a store with 25 memories would silently
        return 20 of them, and the 21st would be unfindable. The
        limit is passed explicitly and set high.

        ``top_k`` has no upper bound in mem0, only a lower one.
        """
        memory = self._require_open()
        try:
            result = memory.get_all(
                filters={"user_id": user_id or self.user_id},
                top_k=LIST_LIMIT)
        except TypeError:
            result = memory.get_all(user_id=user_id or self.user_id,
                                    top_k=LIST_LIMIT)
        return _as_list(result)

    def update(self, memory_id: str, text: str) -> dict[str, Any]:
        """Change the content of one memory.

        A memory that is wrong has to be fixable. Without this the
        only way to correct one is to delete it and store it again,
        and the new one gets a new id - every client that holds the
        old id loses it.
        """
        if not memory_id:
            raise EngineError("the memory id is empty")
        if not text or not text.strip():
            raise EngineError("the new text is empty")
        memory = self._require_open()
        try:
            result = memory.update(memory_id=memory_id, text=text)
        except TypeError:
            result = memory.update(memory_id, text)
        return _as_dict(result)

    def delete_all(self, user_id: str | None = None) -> dict[str, Any]:
        """Delete every memory of one user.

        mem0's own implementation keeps listing after each batch,
        because the vector store caps list() at 100 results by
        default and a single call would silently leave the rest
        behind.
        """
        memory = self._require_open()
        try:
            result = memory.delete_all(user_id=user_id or self.user_id)
        except TypeError:
            result = memory.delete_all(user_id=user_id or self.user_id)
        return _as_dict(result)

    def reset(self) -> dict[str, Any]:
        """Delete absolutely everything.

        This is the whole store, not one user. mem0 deletes the
        collection and recreates it, and resets the history database
        next to it.

        The daemon only calls it when the caller says so explicitly.
        """
        memory = self._require_open()
        try:
            memory.reset()
        except Exception as exc:
            raise EngineError(
                f"the store could not be reset: {exc}") from exc
        return {"reset": True}

    def delete(self, memory_id: str,
               user_id: str | None = None) -> dict[str, Any]:
        """Delete one memory."""
        memory = self._require_open()
        try:
            result = memory.delete(memory_id)
        except TypeError:
            result = memory.delete(memory_id)
        return _as_dict(result)

    def count(self) -> int:
        """How many memories the store holds."""
        return len(self.all())

    def vectors(self, limit: int = 200) -> list[tuple[str, list[float]]]:
        """The stored memories with their vectors.

        The UI clusters them into the topic buttons. The vectors come
        from the vector store itself - they are already there, and
        asking the embedder again would cost a round trip per memory
        and might give a different geometry than the one the vectors
        were written with.
        """
        memory = self._require_open()
        client = getattr(
            getattr(memory, "vector_store", None), "client", None)
        if client is None:
            return []

        collection = self.collection
        try:
            points, _ = client.scroll(
                collection_name=collection,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
        except Exception:
            return []

        result: list[tuple[str, list[float]]] = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("data") or payload.get("text") or ""
            vector = point.vector
            if not text or vector is None:
                continue
            if isinstance(vector, dict):
                # A named vector: take the first one.
                vector = next(iter(vector.values()), None)
            if vector is None:
                continue
            result.append((str(text), list(vector)))
        return result


def _as_dict(value: Any) -> dict[str, Any]:
    """A mem0 result as a dictionary."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {"value": str(value)}


def _as_list(value: Any) -> list[dict[str, Any]]:
    """A mem0 result as a list of dictionaries."""
    if isinstance(value, list):
        return [_as_dict(item) for item in value]
    if isinstance(value, dict):
        # The shape of a search result: {"results": [...]}.
        for key in ("results", "memories", "items"):
            if key in value and isinstance(value[key], list):
                return [_as_dict(item) for item in value[key]]
        return [value]
    if value is None:
        return []
    return [_as_dict(value)]

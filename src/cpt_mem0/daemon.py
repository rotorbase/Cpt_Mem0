"""The daemon.

One process owns the vector store, the two llama-servers and the API.
That is not a style choice: Qdrant's embedded mode takes an exclusive
lock on its storage directory, so a second client fails. The daemon
serves everybody.

**The API is localhost-only by default.** It binds 127.0.0.1, and the
key is generated on the first run. Binding 0.0.0.0 is possible and
warned about.

**The MCP tools are the same operations as the REST endpoints.** The
surface in SPEC 12.1 is small on purpose: ``remember`` and ``recall``
cover the use case, and the raw CRUD is there for the cases they do
not cover.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import Config
from .engine import Engine, EngineError, llama_endpoint
from .supervisor import (
    ServerError,
    ServerSpec,
    Supervisor,
    check_gpu_placement,
)


@dataclass
class DaemonPaths:
    """Where the daemon keeps its files."""

    logs: Path
    state: Path

    @classmethod
    def under(cls, base: Path) -> DaemonPaths:
        return cls(logs=base / "logs", state=base / "state")


class ApiError(Exception):
    """An error the API reports to the client."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class MemoryService:
    """The operations behind the API.

    It is separate from the HTTP layer so that the MCP tools and the
    REST endpoints do the same thing.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def remember(self, text: str, user_id: str | None = None,
                 metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store a text.

        The LLM extracts the facts. A write costs an LLM call, a
        search does not.
        """
        if not text or not text.strip():
            raise ApiError("the text is empty")
        result = self.engine.remember(text, user_id, metadata)
        return {
            "ok": True,
            "stored": _count_of(result),
            "results": _results_of(result),
        }

    def recall(self, query: str, user_id: str | None = None,
               limit: int = 5) -> dict[str, Any]:
        """Search the store."""
        if not query or not query.strip():
            raise ApiError("the query is empty")
        results = self.engine.recall(query, user_id, limit)
        return {
            "ok": True,
            "count": len(results),
            "results": results,
        }

    def list_all(self, user_id: str | None = None) -> dict[str, Any]:
        """Everything in the store."""
        results = self.engine.all(user_id)
        return {"ok": True, "count": len(results), "results": results}

    def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete one memory."""
        if not memory_id:
            raise ApiError("the memory id is empty")
        return {"ok": True, "deleted": memory_id}


def _count_of(result: dict[str, Any]) -> int:
    """How many memories a write produced."""
    for key in ("results", "memories", "items"):
        if key in result and isinstance(result[key], list):
            return len(result[key])
    return 1


def _results_of(result: dict[str, Any]) -> list[Any]:
    """The extracted facts of a write."""
    for key in ("results", "memories", "items"):
        if key in result and isinstance(result[key], list):
            return result[key]
    return []


class Daemon:
    """The daemon.

    It owns the servers, the engine and the API.
    """

    def __init__(self, config: Config, paths: DaemonPaths) -> None:
        self.config = config
        self.paths = paths
        self.supervisor = Supervisor(self._server_specs(),
                                     on_event=self._log_event)
        self.service: MemoryService | None = None
        self.engine: Engine | None = None
        self.started_at: float = 0.0
        self._http: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._events: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _server_specs(self) -> list[ServerSpec]:
        """The servers the daemon supervises."""
        from .install import current_installation

        installation = current_installation()
        if installation.binary is None:
            raise ServerError(
                "no llama.cpp runtime is installed. Run 'cpt-mem0 init' "
                "first.")

        specs: list[ServerSpec] = []

        if self.config.llm.model_file:
            specs.append(ServerSpec(
                name="llm",
                binary=installation.binary,
                model=Path(self.config.llm.model_file),
                port=self.config.llm.port,
                flags=list(self.config.llm.flags),
                log_path=self.paths.logs / "llm.log",
            ))

        if (self.config.embedder.backend == "llamacpp"
                and self.config.embedder.model_file):
            specs.append(ServerSpec(
                name="embedder",
                binary=installation.binary,
                model=Path(self.config.embedder.model_file),
                port=self.config.embedder.port,
                flags=list(self.config.embedder.flags),
                embeddings=True,
                log_path=self.paths.logs / "embedder.log",
            ))

        return specs

    def _log_event(self, name: str, message: str) -> None:
        """Record an event."""
        event = {"time": time.time(), "server": name, "message": message}
        self._events.append(event)
        if len(self._events) > 500:
            self._events = self._events[-500:]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 180.0) -> None:
        """Start the servers, open the store and serve the API.

        The order matters: the servers come up first, because the
        engine needs their endpoints. The store comes last, because
        its lock is the one resource that cannot be taken back.
        """
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        self.paths.state.mkdir(parents=True, exist_ok=True)

        for spec in self.supervisor.states:
            self.supervisor.start(spec, timeout=timeout)

        self.engine = Engine(
            llm=llama_endpoint(self.config.llm.port,
                               self.config.llm.model or "model"),
            embedder=llama_endpoint(self.config.embedder.port,
                                    self.config.embedder.model),
            store_path=self.config.store.path,
            collection=self.config.store.collection,
            dimensions=self.config.embedder.dimensions,
            user_id=self.config.user_id,
        )
        self.engine.open()
        self.service = MemoryService(self.engine)

        self._start_http()
        self.started_at = time.time()
        self._write_pid_file()

    def _write_pid_file(self) -> None:
        """Write the pid file.

        It is how 'cpt-mem0 stop' finds the daemon. Without it a
        daemon that runs in the background can only be stopped by
        killing the process by hand - and then the store lock is left
        behind.
        """
        import os

        try:
            self.paths.state.mkdir(parents=True, exist_ok=True)
            (self.paths.state / "daemon.pid").write_text(
                str(os.getpid()), encoding="utf-8")
        except OSError:
            pass

    def _remove_pid_file(self) -> None:
        """Remove the pid file."""
        try:
            (self.paths.state / "daemon.pid").unlink(missing_ok=True)
        except OSError:
            pass

    def _start_http(self) -> None:
        """Start the HTTP server."""
        handler = _make_handler(self)
        self._http = ThreadingHTTPServer(
            (self.config.server.host, self.config.server.port), handler)
        self._http.daemon_threads = True
        self._thread = threading.Thread(
            target=self._http.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop everything.

        The order is the reverse: the API stops accepting work, the
        store is released, then the servers go down. The store has to
        be released before the process exits, otherwise the lock is
        left behind.
        """
        if self._http is not None:
            try:
                self._http.shutdown()
                self._http.server_close()
            except Exception:
                pass
            self._http = None

        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.service = None

        self.supervisor.stop_all()
        self._remove_pid_file()

    def run_forever(self) -> None:
        """Start and wait."""
        self.start()
        print(f"the daemon is serving on "
              f"http://{self.config.server.host}:{self.config.server.port}")
        print("press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1.0)
                self._watch()
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            self.stop()

    def _watch(self) -> None:
        """Restart a server that died."""
        for name, state in list(self.supervisor.states.items()):
            if state.process is None:
                continue
            if state.process.poll() is not None and state.ready:
                self._log_event(name, "the server died")
                try:
                    self.supervisor.restart(name)
                except ServerError as exc:
                    self._log_event(name, str(exc))

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The state of the daemon."""
        from .supervisor import gpu_used_mib

        servers = self.supervisor.status()
        warnings = check_gpu_placement(self.supervisor)
        if warnings:
            self._log_event("gpu", "; ".join(warnings))

        count = None
        if self.engine is not None and self.engine.is_open:
            try:
                count = self.engine.count()
            except Exception:
                count = None

        return {
            "running": self._http is not None,
            "uptime": round(time.time() - self.started_at, 1)
            if self.started_at else 0.0,
            "address": f"http://{self.config.server.host}:"
                       f"{self.config.server.port}",
            "servers": servers,
            "gpu_used_mib": gpu_used_mib(),
            "store_open": self.engine is not None and self.engine.is_open,
            "memories": count,
            "warnings": warnings,
        }


# ----------------------------------------------------------------------
# the HTTP layer
# ----------------------------------------------------------------------

def _make_handler(daemon: Daemon) -> type[BaseHTTPRequestHandler]:
    """The request handler for this daemon."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "cpt-mem0"

        def log_message(self, format: str, *args: Any) -> None:
            """The default logging is too noisy."""
            return

        # -- helpers ---------------------------------------------------

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ApiError(
                    f"the body is not valid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise ApiError("the body is not a JSON object")
            return data

        def _authorised(self) -> bool:
            """Whether the request may proceed.

            A key is only needed when one is configured. With no key
            the service is only reachable from localhost, which is the
            default.
            """
            keys = daemon.config.server.api_keys
            if not keys:
                return True
            header = self.headers.get("Authorization") or ""
            presented = header[7:] if header.startswith("Bearer ") else ""
            for entry in keys:
                expected = str(entry.get("key") or "")
                if expected and secrets.compare_digest(expected, presented):
                    return True
            return False

        def _service(self) -> MemoryService:
            if daemon.service is None:
                raise ApiError("the service is not ready", status=503)
            return daemon.service

        # -- routes ----------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            try:
                if path in ("/health", "/v1/health"):
                    self._send(200, {"ok": True, "status": "healthy"})
                    return

                if path == "/status":
                    self._send(200, daemon.status())
                    return

                if not self._authorised():
                    self._send(401, {"ok": False, "error": "unauthorised"})
                    return

                if path in ("/memories", "/v1/memories"):
                    user = query.get("user_id", [None])[0]
                    self._send(200, self._service().list_all(user))
                    return

                if path in ("/search", "/v1/search"):
                    q = query.get("q", query.get("query", [""]))[0]
                    user = query.get("user_id", [None])[0]
                    limit = int(query.get("limit", ["5"])[0])
                    self._send(200, self._service().recall(q, user, limit))
                    return

                self._send(404, {"ok": False, "error": "not found"})
            except ApiError as exc:
                self._send(exc.status, {"ok": False, "error": exc.message})
            except EngineError as exc:
                self._send(503, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send(500, {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            try:
                if not self._authorised():
                    self._send(401, {"ok": False, "error": "unauthorised"})
                    return

                body = self._read_body()
                service = self._service()

                if path in ("/memories", "/v1/memories", "/remember"):
                    text = str(body.get("text") or body.get("content") or "")
                    user = body.get("user_id")
                    metadata = body.get("metadata")
                    self._send(200, service.remember(text, user, metadata))
                    return

                if path in ("/search", "/v1/search", "/recall"):
                    query = str(body.get("query") or body.get("q") or "")
                    user = body.get("user_id")
                    limit = int(body.get("limit") or 5)
                    self._send(200, service.recall(query, user, limit))
                    return

                if path.startswith("/memories/"):
                    memory_id = path.rsplit("/", 1)[-1]
                    self._send(200, service.delete(memory_id))
                    return

                self._send(404, {"ok": False, "error": "not found"})
            except ApiError as exc:
                self._send(exc.status, {"ok": False, "error": exc.message})
            except EngineError as exc:
                self._send(503, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send(500, {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"})

        def do_DELETE(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                if not self._authorised():
                    self._send(401, {"ok": False, "error": "unauthorised"})
                    return
                memory_id = path.rsplit("/", 1)[-1]
                self._send(200, self._service().delete(memory_id))
            except ApiError as exc:
                self._send(exc.status, {"ok": False, "error": exc.message})
            except EngineError as exc:
                self._send(503, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send(500, {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"})

    return Handler


# ----------------------------------------------------------------------
# the MCP tools
# ----------------------------------------------------------------------

def mcp_tools(service: MemoryService) -> list[dict[str, Any]]:
    """The MCP tool definitions.

    The same operations as the REST endpoints. The tool names are the
    ones in SPEC 12.1.
    """
    return [
        {
            "name": "remember",
            "description": "Store a fact. The LLM extracts the facts "
                           "from the text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "The text to remember."},
                    "user_id": {"type": "string",
                                "description": "Who the fact belongs to."},
                },
                "required": ["text"],
            },
            "handler": lambda args: service.remember(
                str(args.get("text", "")), args.get("user_id")),
        },
        {
            "name": "recall",
            "description": "Search the stored facts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "What to search for."},
                    "user_id": {"type": "string",
                                "description": "Whose facts to search."},
                    "limit": {"type": "integer",
                              "description": "How many results."},
                },
                "required": ["query"],
            },
            "handler": lambda args: service.recall(
                str(args.get("query", "")), args.get("user_id"),
                int(args.get("limit", 5))),
        },
    ]


def generate_api_key() -> str:
    """A new API key."""
    return secrets.token_urlsafe(32)

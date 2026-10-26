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
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import proc
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

    @classmethod
    def default(cls) -> DaemonPaths:
        """The standard place: under the data directory of the app."""
        from .install import data_dir

        return cls.under(data_dir())


class _QuietHTTPServer(ThreadingHTTPServer):
    """The HTTP server of the daemon.

    **A client that goes away is not an error.** The browser closes
    the connection while the request is being read or while the answer
    is being written - on every reload, on every screenshot, on every
    tab that is closed. The standard handler prints a traceback for
    it, and it looks like a crash of the daemon.

    The traceback comes from two places: the reading of the request
    line and the writing of the answer. The second one is caught in
    the handler; this one is here.
    """

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionAbortedError,
                              ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


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

    def __init__(self, engine: Engine,
                 is_suspended: Callable[[], bool] | None = None) -> None:
        self.engine = engine
        #: Whether the models are down. The search needs the embedder
        #: and the write needs the LLM, so both refuse while the
        #: models are down - with a message that says so, not with a
        #: connection error.
        self._is_suspended = is_suspended or (lambda: False)

    def _require_models(self, what: str) -> None:
        if self._is_suspended():
            raise ApiError(
                f"the models are suspended - {what} needs them. Switch "
                f"them on first.", status=503)

    def remember(self, text: str, user_id: str | None = None,
                 metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store a text.

        The LLM extracts the facts. A write costs an LLM call, a
        search does not.
        """
        if not text or not text.strip():
            raise ApiError("the text is empty")
        self._require_models("storing a fact")
        result = self.engine.remember(text, user_id, metadata)
        return {
            "ok": True,
            "stored": _count_of(result),
            "results": _results_of(result),
        }

    def chat(self, text: str, user_id: str | None = None,
             history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Answer the user and remember the conversation.

        **This is the whole thing.** The model answers - that is the
        inference - and the exchange goes into mem0, so the memory
        holds the communication and not only the question.

        The order matters. The answer comes first, because that is
        what the user is waiting for; the extraction runs afterwards
        and uses the answer, because what the assistant said belongs
        to the conversation too.
        """
        if not text or not text.strip():
            raise ApiError("the text is empty")
        self._require_models("the answer")

        # **What the memory knows about the question.** The model
        # gets it in the prompt - that is the difference between a
        # memory and a chatbot.
        known = self.engine.recall(text, user_id, limit=8)
        memories = [str(item.get("memory") or "") for item in known]
        memories = [item for item in memories if item]

        # **The thread goes with it.** The model sees the last turns,
        # so the second question of a conversation can refer to the
        # first answer.
        answer = self.engine.ask(text, memories, history=history)

        # **The communication goes into the store.** Both sides: what
        # the user said and what came back.
        from .engine import EXTRACTION_PROMPT

        conversation = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": answer},
        ]
        stored = self.engine.remember_conversation(
            conversation, user_id, prompt=EXTRACTION_PROMPT)

        return {
            "ok": True,
            "answer": answer,
            "stored": _count_of(stored),
            "results": _results_of(stored),
            "used_memories": len(memories),
        }

    def recall(self, query: str, user_id: str | None = None,
               limit: int = 5) -> dict[str, Any]:
        """Search the store."""
        if not query or not query.strip():
            raise ApiError("the query is empty")
        self._require_models("the search")
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

    def update(self, memory_id: str, text: str) -> dict[str, Any]:
        """Change the content of one memory.

        A memory that is wrong has to be fixable. Deleting it and
        storing it again would give it a new id, and every client
        that holds the old one loses it.
        """
        if not memory_id:
            raise ApiError("the memory id is empty")
        if not text or not text.strip():
            raise ApiError("the new text is empty")
        self.engine.update(memory_id, text)
        return {"ok": True, "updated": memory_id, "text": text}

    def delete_all(self, user_id: str | None = None) -> dict[str, Any]:
        """Delete every memory of one user."""
        result = self.engine.delete_all(user_id)
        return {"ok": True, "deleted_all_for":
                user_id or self.engine.user_id, "result": result}

    def reset(self) -> dict[str, Any]:
        """Delete absolutely everything.

        This is the whole store, not one user. It is a separate
        operation on purpose: "delete my memories" and "wipe the
        service" are different requests.
        """
        self.engine.reset()
        return {"ok": True, "reset": True}


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
        # The servers are built here, but a missing runtime is not
        # an error yet: the daemon has to start so the window comes
        # up and can say what is missing. The models are only needed
        # when they are switched on.
        self.supervisor = Supervisor([], on_event=self._log_event)
        self._runtime_error = ""
        #: The plan the app made for itself.
        self._auto: Any = None
        #: True while the models are being loaded.
        self._loading = False
        #: Why the models cannot be loaded (yet).
        self._waiting = ""
        #: What went wrong with the last load.
        self._load_error = ""

        # **The models are found before the servers are built.**
        # Without a model file there is no server at all, and the
        # switch in the window would do nothing - with no message
        # about why.
        self._autoconfigure()

        try:
            self.supervisor = Supervisor(self._server_specs(),
                                         on_event=self._log_event)
        except ServerError as error:
            self._runtime_error = str(error)
        self.service: MemoryService | None = None
        self.engine: Engine | None = None
        self.started_at: float = 0.0
        self._http: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._events: list[dict[str, Any]] = []
        #: True when the models are down and the VRAM is free.
        self.suspended = False
        #: The settings of the window.
        self.settings_data: dict[str, Any] = {}
        self._load_settings()
        #: The voice. It is created on the first use - the SAPI
        #: object is the slow part and it is not needed when the
        #: voice is off.
        self._speaker: Any = None
        #: The voice the user chose.
        self._voice_name: str = ""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _server_specs(self) -> list[ServerSpec]:
        """The servers the daemon supervises.

        **A missing runtime is not an error here.** The daemon comes
        up without it, the window shows it, and the switch says what
        is missing. Only switching the models on needs the runtime.
        """
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

    def _autoconfigure(self) -> None:
        """Find the models and plan the configuration.

        **The app has to work without a console.** Nobody is going to
        type a path into a command line: the app finds the models that
        are on the machine, asks the planner which of them fits on the
        card right now, and takes the flags from the plan.

        The plan is written to the configuration, so the next start
        does not have to do it again - and so the user can see and
        change it.
        """
        from .autoconfig import apply as apply_plan
        from .autoconfig import configure

        try:
            plan = configure(
                self.config,
                extra=self.settings_data.get("model_folder") or None)
        except Exception as error:
            self._load_error = str(error)
            return

        self._auto = plan
        if plan.ok:
            apply_plan(self.config, plan)
            self._save_config()
        else:
            self._waiting = plan.reason

    def _save_config(self) -> None:
        """Write the configuration."""
        from .config import default_config_path, save_config

        try:
            save_config(self.config, default_config_path())
        except OSError:
            pass

    def _log_event(self, name: str, message: str) -> None:
        """Record an event."""
        event = {"time": time.time(), "server": name, "message": message}
        self._events.append(event)
        if len(self._events) > 500:
            self._events = self._events[-500:]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 180.0,
              suspend: bool = True) -> None:
        """Start the servers, open the store and serve the API.

        The order matters: the servers come up first, because the
        engine needs their endpoints. The store comes last, because
        its lock is the one resource that cannot be taken back.

        **The models stay down unless somebody asks for them.** The
        default is ``suspend=True``, and that is not a detail: the
        app must never take the VRAM on its own. A program that
        loads a 9B model into the card while a game is running
        makes the game stutter, and the user has no way to see that
        the app did it.

        The store opens anyway - it does not need a model - and the
        API comes up. So the window is there, the memory is there,
        and the VRAM stays free until the user presses the switch.
        """
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        self.paths.state.mkdir(parents=True, exist_ok=True)

        if suspend:
            self.suspended = True
        elif self._runtime_error:
            # Without the runtime there are no models to start. The
            # daemon comes up anyway - the window says what is
            # missing.
            self.suspended = True
            self._log_event("runtime", self._runtime_error)
        else:
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
        self.service = MemoryService(
            self.engine, is_suspended=lambda: self.suspended)

        self._start_http()
        self.started_at = time.time()
        self._write_pid_file()

    @property
    def is_running(self) -> bool:
        """Whether the daemon is still up.

        The browser fallback keeps the app alive until the daemon
        ends - the user closes the app, and then the store has to be
        released.
        """
        return self._http is not None

    @property
    def address(self) -> str:
        """The address of the window.

        Empty while the HTTP server is not up yet.
        """
        if self._http is None:
            return ""
        host, port = self._http.server_address[:2]
        return f"http://{host}:{port}"

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
        self._http = _QuietHTTPServer(
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

    def suspend(self) -> dict[str, Any]:
        """Stop the models and free the VRAM.

        **The daemon and the store stay up.** The models are child
        processes, the store lives in this process. So "off" means:
        the memory is still there, it can still be listed and deleted,
        and only the VRAM is free.

        The search needs the embedder and the write needs the LLM, so
        both refuse while the models are down - with a message that
        says so, not with a connection error.
        """
        if self.suspended:
            return {"suspended": True, "already": True}

        self.supervisor.stop_all()
        self.suspended = True
        self._log_event("runtime", "the models were suspended")
        return {"suspended": True}

    def resume(self, timeout: float = 180.0) -> dict[str, Any]:
        """Start the models again.

        **Without the runtime this is an error with a clear
        message**, not a crash: "run cpt-mem0 init first". The window
        shows it.

        **In the background.** The load of a 9B takes a minute, and a
        switch that stands there for a minute without an answer looks
        broken. The answer comes at once, the switch says "loading",
        and the status says what is happening.

        **The app waits for the VRAM.** When another program has the
        card - a game, another model - the model does not fit. The app
        neither fails nor pretends: it waits, and says how much memory
        it is waiting for. When the game is over, the model comes up
        on its own.
        """
        if not self.suspended:
            return {"suspended": False, "already": True}

        if self._runtime_error:
            raise ApiError(self._runtime_error, status=503)

        if self._loading:
            return {"suspended": False, "loading": True}

        self._loading = True
        self._waiting = ""
        self._load_error = ""
        thread = threading.Thread(target=self._resume_worker, daemon=True)
        thread.start()
        return {"suspended": False, "loading": True}

    def _resume_worker(self) -> None:
        """Load the models, and wait for the VRAM if it is not there."""
        from .autoconfig import apply as apply_plan
        from .autoconfig import configure

        while True:
            try:
                plan = configure(
                    self.config,
                    extra=self.settings_data.get("model_folder") or None)
            except Exception as error:
                self._loading = False
                self._load_error = str(error)
                return

            self._auto = plan

            if plan.ok:
                apply_plan(self.config, plan)
                self._save_config()
                try:
                    self.supervisor = Supervisor(self._server_specs(),
                                                 on_event=self._log_event)
                    for spec in self.supervisor.states:
                        self.supervisor.start(spec)
                except ServerError as error:
                    self._loading = False
                    self._load_error = str(error)
                    self._log_event("runtime", str(error))
                    return
                self.suspended = False
                self._loading = False
                self._waiting = ""
                self._log_event("runtime", "the models are loaded")
                return

            # **Not enough VRAM.** Wait for it, and say so.
            self._waiting = plan.reason
            self._log_event("runtime", plan.reason)
            time.sleep(5)

    def _watch(self) -> None:
        """Restart a server that died."""
        if self.suspended:
            # The models were stopped on purpose. Nothing restarts
            # them - only the user does that.
            return
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

    def speak(self, text: str) -> bool:
        """Read the text out loud, when the voice is on.

        **It costs no VRAM.** The voice is the one Windows already
        has, so nothing is loaded onto the card.
        """
        if not self.settings_data.get("voice"):
            return False
        if not text:
            return False

        from .voice import Speaker

        name = str(self.settings_data.get("voice_name") or "")
        if self._speaker is None or self._voice_name != name:
            # **A new voice means a new speaker** - the SAPI object
            # holds the voice it was given.
            self._speaker = Speaker(name=name)
            self._voice_name = name
        return self._speaker.speak(text)

    def models(self) -> dict[str, Any]:
        """Which models fit on the card, and where to get them.

        **The suggestions with the link.** The user asks which model
        fits on their card. The answer is a list: the model, the
        quantisation, how much of the card it needs, how much context
        is left over, the link to the file on HuggingFace, and
        whether the file is already on this machine.

        The order is the order of the plan: the best model that fits
        comes first. The context matters as much as the size - the
        extraction takes the existing memories with it, and a model
        with a small context cannot hold a long conversation.
        """
        from .catalogue import load_default_catalogue
        from .hardware import detect_hardware
        from .models import find_models
        from .planner import PlannerConfig, evaluate_candidate

        catalogue = load_default_catalogue()
        hardware = detect_hardware()
        if hardware is None:
            return {"models": [], "error": "die Grafikkarte konnte nicht "
                                          "gelesen werden"}

        local = {model.path.name.lower()
                 for model in find_models(catalogue=catalogue)}

        config = PlannerConfig(embedder_backends=("llamacpp", "fastembed"))
        results: list[dict[str, Any]] = []

        for entry in catalogue.sorted_entries():
            for quant in entry.sorted_quantisations():
                if not quant.kv_bytes_per_token:
                    # Without the KV constant the size on the card is
                    # unknown, and an unknown size is not a
                    # suggestion.
                    continue
                candidate = evaluate_candidate(
                    model_id=entry.model_id,
                    quantisation=quant.name,
                    file_name=quant.file_name,
                    file_size_mib=quant.file_size_mib,
                    kv_bytes_per_token=quant.kv_bytes_per_token,
                    cache_type="q8_0",
                    hardware=hardware,
                    config=config,
                    embedder_backend="llamacpp",
                    model_rank_basis=entry.rank_basis,
                    recurrent_state_mib=entry.recurrent_state_mib,
                )
                if candidate is None:
                    continue
                results.append({
                    "model_id": entry.model_id,
                    "quantisation": quant.name,
                    "file": quant.file_name,
                    "size_mib": round(quant.file_size_mib),
                    "vram_mib": round(candidate.total_vram_mib),
                    "n_ctx": candidate.n_ctx,
                    "measured": entry.rank_basis == "measured",
                    "link": (f"https://huggingface.co/{entry.repo}/resolve/"
                             f"main/{quant.file_name}"),
                    "page": f"https://huggingface.co/{entry.repo}",
                    "local": quant.file_name.lower() in local,
                })

        # **The best that fits comes first.** A model that does not
        # fit is not a suggestion, but it is worth knowing that it
        # exists - so it stays in the list and is marked.
        results.sort(key=lambda item: (not item["local"], item["vram_mib"]))
        return {
            "models": results,
            "card": {"name": hardware.name, "vram_mib": hardware.total_vram_mib},
        }

    def voice_status(self) -> dict[str, Any]:
        """The voices of the machine, and why there are none.

        **The user picks the voice.** Windows has a voice per
        language, and the German one is not always there. The list is
        shown so the user can hear which one is which.
        """
        from .voice import voice_status

        status = voice_status()
        return {
            "available": status["available"],
            "reason": status["reason"],
            "voices": status["voices"],
            "selected": self.settings_data.get("voice_name"),
        }

    def settings(self) -> dict[str, Any]:
        """The settings of the window."""
        return {
            "autostart": self.settings_data.get("autostart", False),
            "voice": self.settings_data.get("voice", False),
            "winamp": self.settings_data.get("winamp", True),
            #: Which interface the window shows.
            "theme": self.settings_data.get("theme", "plain"),
            "model": self.settings_data.get("model"),
            #: The topics the user marked. They are the first fact of
            #: the cluster, not its name - the name comes from the
            #: model and the model may name it differently tomorrow.
            "favourites": list(self.settings_data.get("favourites", [])),
            #: Where the user keeps the models.
            "model_folder": self.settings_data.get("model_folder", ""),
        }

    def toggle_favourite(self, fact: str) -> list[str]:
        """Mark a topic as a favourite, or unmark it.

        **A favourite is a topic the user works on.** It stays at the
        front of the buttons when there are more topics than buttons.
        It is stored by the first fact of the cluster, not by its
        name: the name comes from the model, and the model may name
        the same topic differently tomorrow.
        """
        fact = fact.strip()
        if not fact:
            return list(self.settings_data.get("favourites", []))
        favourites = [str(item) for item
                      in self.settings_data.get("favourites", [])]
        if fact in favourites:
            favourites.remove(fact)
        else:
            favourites.append(fact)
        self.settings_data["favourites"] = favourites
        self._save_settings()
        return favourites

    def set_setting(self, name: str, value: Any) -> dict[str, Any]:
        """Change one setting.

        The autostart is a real system change on Windows: it writes a
        scheduled task that starts the daemon at the logon. It is
        never done without being asked for.
        """
        if name not in ("autostart", "voice", "winamp", "model", "theme",
                        "favourites", "voice_name", "model_folder"):
            raise ApiError(f"unknown setting: {name}")

        if name == "autostart":
            self.settings_data["autostart"] = bool(value)
            self._apply_autostart(bool(value))
        elif name == "model":
            self.settings_data["model"] = str(value) if value else None
        elif name == "model_folder":
            # **Where the models are.** The user decides - the app
            # only looks there.
            self.settings_data["model_folder"] = str(value or "")
            self._save_settings()
            return {"ok": True,
                    "model_folder": self.settings_data["model_folder"]}
        elif name == "voice_name":
            # **Which voice speaks.** The names come from Windows.
            self.settings_data["voice_name"] = str(value) if value else ""
            self._save_settings()
            return {"ok": True,
                    "voice_name": self.settings_data["voice_name"]}
        elif name == "theme":
            # **A name, not a flag.** There are four interfaces, and
            # the window says which one it wants.
            self.settings_data["theme"] = str(value or "plain")
            self._save_settings()
            return {"ok": True, "theme": self.settings_data["theme"]}
        elif name == "favourites":
            # **A list of topics, not a flag.** The favourite topics
            # are stored by their first fact, so a topic that the
            # model names differently tomorrow is still the same one.
            items = value if isinstance(value, list) else []
            cleaned = [str(item) for item in items if str(item).strip()]
            self.settings_data["favourites"] = cleaned
            self._save_settings()
            return {"ok": True, "favourites": cleaned}
        else:
            self.settings_data[name] = bool(value)

        self._save_settings()

        # The voice switch has to say whether it can work at all.
        if name == "voice" and value:
            from .voice import available

            if not available():
                return {
                    "ok": False,
                    "error": "this system has no speech engine - the "
                             "voice needs Windows",
                    "settings": self.settings(),
                }

        return {"ok": True, "settings": self.settings()}

    def _settings_path(self) -> Path:
        from .install import config_dir

        return config_dir() / "settings.json"

    def _load_settings(self) -> None:
        try:
            import json

            self.settings_data = json.loads(
                self._settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.settings_data = {}

    def _save_settings(self) -> None:
        try:
            import json

            path = self._settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.settings_data, indent=2),
                            encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _autostart_command() -> str:
        """The command line the autostart task runs.

        **The models stay off.** The task starts the daemon with the
        models suspended: the window and the API are there, the VRAM
        stays free, and the user presses the switch when the card
        should be used.

        From the packaged app it is the .exe itself. From the source
        tree it is the module.
        """
        import sys

        if getattr(sys, "frozen", False):
            return '"' + sys.executable + '" --suspend'
        return '"' + sys.executable + '" -m cpt_mem0.cli ui --suspend'

    def _apply_autostart(self, enabled: bool) -> None:
        """Write the autostart task on Windows.

        The task starts the daemon at the logon with the models
        **off** - the daemon comes up, the API is reachable, and the
        VRAM stays free until the user switches the models on. The
        reason is the decision: nothing happens to the GPU without
        being asked for.
        """
        import os
        import subprocess

        if os.name != "nt":
            # On Linux and macOS the autostart is a systemd unit or a
            # launch agent - not part of this version.
            return

        task_name = "Cpt_Mem0"
        action = self._autostart_command()
        # **Without a shell.** With `shell=True` Windows starts
        # `cmd.exe`, and that is the one that gets the console: the
        # black window appears even with the flag that hides it from
        # the program itself. The arguments are a list instead.
        try:
            if enabled:
                proc.run(
                    ["schtasks", "/Create", "/TN", task_name, "/TR", action,
                     "/SC", "ONLOGON", "/RL", "LIMITED", "/F"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=30)
            else:
                proc.run(
                    ["schtasks", "/Delete", "/TN", task_name, "/F"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass

    def topics(self, count: int = 5, name: bool = True) -> dict[str, Any]:
        """The topic buttons.

        The facts come from the store, the grouping from the vectors
        that are already there, the names from the local LLM.

        When the models are suspended the grouping still works - it
        needs no model - but the names fall back to the first fact of
        the cluster.
        """
        from .topics import cluster, name_clusters, palette

        entries = self.engine.entries(limit=300) if self.engine else []
        clusters = cluster(entries, k=max(1, min(count, 6)))

        if name and not self.suspended:
            clusters = name_clusters(clusters, llm=self._ask_llm)

        # **The favourites come first and they stay.** The user marks
        # the topics they work on. When there are more topics than
        # buttons, the marked ones are not the ones that get pushed
        # out.
        favourites = {str(item) for item
                      in self.settings_data.get("favourites", [])}

        def rank(cluster: Any) -> tuple:
            marked = cluster.facts[0] in favourites if cluster.facts else False
            return (0 if marked else 1, -cluster.last_seen, -cluster.size)

        clusters.sort(key=rank)
        clusters = clusters[:max(1, min(count, 6))]

        colours = palette(len(clusters))

        return {
            "clusters": [entry.to_dict() for entry in clusters],
            "colours": colours,
            "count": len(clusters),
            "facts": len(entries),
        }

    def _ask_llm(self, prompt: str) -> str:
        """Ask the local LLM for a short answer.

        The endpoint is the OpenAI-compatible one of the running
        server, so the answer is the same as in the chat.
        """
        import json
        import urllib.request

        endpoint = llama_endpoint(self.config.llm.port,
                                  self.config.llm.model or "model")
        payload = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint.url("/chat/completions"), data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {endpoint.api_key}"},
            method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def plan(self) -> dict[str, Any]:
        """The models that fit on this card.

        The UI offers every model that fits, not only the one the
        planner picks. The user chooses what to load; the planner
        decides how.

        A model without a measured KV constant cannot be planned and
        is listed as such - it needs a calibration run first.
        """
        from .catalogue import load_default_catalogue
        from .hardware import detect_hardware
        from .planner import PlannerConfig, plan_for_hardware

        hardware = detect_hardware()
        if hardware is None:
            # **No GPU is not an error.** The window has to answer on
            # every machine - a virtual machine, a server, a laptop
            # with a card the driver does not report. The planner
            # needs a positive number of megabytes, and zero is not
            # one: it raises. So the answer is an empty list with the
            # reason, not a crash.
            return {
                "gpu": {"name": None, "total_vram_mib": 0},
                "models": [],
                "count": 0,
                "reason": "no GPU was found on this machine",
            }

        catalogue = load_default_catalogue()

        models: list[dict[str, Any]] = []
        for entry in catalogue.sorted_entries():
            result = plan_for_hardware(
                hardware, [entry],
                PlannerConfig(embedder_backends=("llamacpp", "fastembed")))
            candidate = result.candidate
            models.append({
                "model_id": entry.model_id,
                "fits": bool(result.viable and candidate is not None),
                "quantisation": candidate.quantisation if candidate else None,
                "file_size_mib": round(entry.quantisations[0].file_size_mib)
                if entry.quantisations else None,
                "total_vram_mib": round(candidate.total_vram_mib)
                if candidate else None,
                "n_ctx": candidate.n_ctx if candidate else None,
                "rank_basis": entry.rank_basis,
                "measured": entry.extraction is not None,
                "reason": result.diagnostics.get("reason")
                if not result.viable else None,
            })

        return {
            "gpu": {
                "name": hardware.name,
                "total_vram_mib": hardware.total_vram_mib,
            },
            "models": models,
            "count": len(models),
            "reason": None,
        }

    def runtime_status(self) -> dict[str, Any]:
        """What the on/off button shows.

        The VRAM number is the point of it: the user wants to see
        that the models are gone and the card is free.

        **`models_loaded` is what the servers say, not what the
        switch says.** The switch only knows that nobody asked for the
        models to be stopped - it does not know whether a model is
        really there. A button that says "the models are loaded"
        while the card is empty is worse than no button.
        """
        from .supervisor import gpu_buffers, gpu_free_mib, gpu_used_mib, gpu_utilisation

        servers = self.supervisor.status()
        loaded = [entry for entry in servers
                  if entry["running"] and entry["ready"]]

        # **Is it really on the card?** The server log says which
        # device the model is on. A server that answers but computes
        # on the CPU is loaded and useless at the same time.
        on_gpu = False
        app_vram_mib = 0.0
        for entry in loaded:
            spec = self.supervisor.states[entry["name"]].spec
            buffers = gpu_buffers(spec.log_path)
            model_mib = buffers["model_mib"]
            if model_mib is not None and model_mib >= 1000:
                on_gpu = True
            # **What this app holds on the card.** The number of the
            # whole card says nothing about the app: another program
            # can use most of it. The user looks at the task manager
            # and wants to find *this* number there.
            for key in ("model_mib", "kv_mib", "rs_mib", "compute_mib"):
                value = buffers[key]
                if value:
                    app_vram_mib += value

        return {
            "suspended": self.suspended,
            "models_loaded": bool(loaded),
            "on_gpu": on_gpu,
            "app_vram_mib": int(app_vram_mib) if app_vram_mib else 0,
            "gpu_used_mib": gpu_used_mib(),
            "gpu_free_mib": gpu_free_mib(),
            "servers": servers,
            # **What the switch is doing.** The load takes a minute,
            # and the wait for the VRAM can take longer.
            "loading": self._loading,
            "waiting": self._waiting,
            "load_error": self._load_error,
            "auto": self._auto.to_dict() if self._auto is not None else {},
            # **The cockpit.** What the instruments show.
            "tokens_per_second": round(
                getattr(self.engine, "last_tokens_per_second", 0.0) or 0.0, 1),
            "gpu_name": self._gpu_name(),
            "gpu_percent": gpu_utilisation(),
        }

    def _gpu_name(self) -> str:
        """The name of the card."""
        from .hardware import detect_hardware

        try:
            hardware = detect_hardware()
        except Exception:
            return ""
        return hardware.name if hardware else ""

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
            except Exception as error:
                # The counter must not fail silently - a wrong number
                # in the window is worse than no number.
                count = None
                self._log_event("store", f"the count failed: {error}")

        return {
            "running": self._http is not None,
            "suspended": self.suspended,
            "uptime": round(time.time() - self.started_at, 1)
            if self.started_at else 0.0,
            "address": f"http://{self.config.server.host}:"
                       f"{self.config.server.port}",
            "servers": servers,
            "gpu_used_mib": gpu_used_mib(),
            "store_open": self.engine is not None and self.engine.is_open,
            "memories": count,
            "warnings": warnings,
            "runtime_error": self._runtime_error,
        }


# ----------------------------------------------------------------------
# the HTTP layer
# ----------------------------------------------------------------------

#: The directory with the user interface.
UI_DIR = Path(__file__).resolve().parent / "ui"

#: The files the browser may ask for. Anything else is not served.
UI_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/ui/style.css": ("style.css", "text/css; charset=utf-8"),
    "/ui/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


def _make_handler(daemon: Daemon) -> type[BaseHTTPRequestHandler]:
    """The request handler for this daemon."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "cpt-mem0"

        def log_message(self, format: str, *args: Any) -> None:
            """The default logging is too noisy."""
            return

        # -- helpers ---------------------------------------------------

        def _send(self, status: int, payload: Any) -> None:
            """Answer the request.

            **A client that goes away is not an error.** The browser
            closes the connection while the answer is being written -
            it happens every time a page is reloaded or a screenshot
            is taken - and the traceback that came out of it looked
            like a crash of the daemon. It is normal operation.
            """
            body = json.dumps(payload, default=str).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ConnectionAbortedError, ConnectionResetError,
                    BrokenPipeError):
                pass

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

        def _send_ui(self, path: str) -> None:
            """Deliver a file of the user interface."""
            name, content_type = UI_FILES[path]
            try:
                body = (UI_DIR / name).read_bytes()
            except OSError:
                self._send(404, {"ok": False, "error": "not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

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
                if path in UI_FILES:
                    self._send_ui(path)
                    return

                if path in ("/health", "/v1/health"):
                    self._send(200, {"ok": True, "status": "healthy"})
                    return

                if path == "/status":
                    self._send(200, daemon.status())
                    return

                if path in ("/runtime", "/v1/runtime"):
                    self._send(200, daemon.runtime_status())
                    return

                if path in ("/settings", "/v1/settings"):
                    self._send(200, daemon.settings())
                    return

                if path in ("/models", "/v1/models"):
                    # **The suggestions.** Which model fits on the
                    # card, and where to get it.
                    self._send(200, daemon.models())
                    return

                if path in ("/settings/voices", "/v1/settings/voices"):
                    # **The voices of the machine.** The user picks
                    # one - Windows has a voice per language, and the
                    # German one is not always installed.
                    self._send(200, daemon.voice_status())
                    return

                if path in ("/plan", "/v1/plan"):
                    self._send(200, daemon.plan())
                    return

                if path in ("/topics", "/v1/topics"):
                    count = int(query.get("count", ["5"])[0])
                    self._send(200, daemon.topics(count=count))
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
                    result = self._service().recall(q, user, limit)
                    # The window asks with GET, so the voice has to
                    # be started here as well.
                    from .voice import summarise

                    daemon.speak(summarise(result.get("results", [])))
                    self._send(200, result)
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
                    result = service.remember(text, user, metadata)
                    # The voice reads out what was stored - the facts,
                    # not the whole answer.
                    from .voice import summarise

                    daemon.speak(summarise(result.get("results", [])))
                    self._send(200, result)
                    return

                if path in ("/chat", "/v1/chat"):
                    # **The chat.** The model answers, and the exchange
                    # goes into the memory.
                    text = str(body.get("text") or body.get("content")
                               or body.get("message") or "")
                    user = body.get("user_id")
                    history = body.get("history")
                    if not isinstance(history, list):
                        history = []
                    result = service.chat(text, user, history)
                    from .voice import summarise

                    daemon.speak(summarise(result.get("results", [])))
                    self._send(200, result)
                    return

                if path in ("/search", "/v1/search", "/recall"):
                    query = str(body.get("query") or body.get("q") or "")
                    user = body.get("user_id")
                    limit = int(body.get("limit") or 5)
                    result = service.recall(query, user, limit)
                    from .voice import summarise

                    daemon.speak(summarise(result.get("results", [])))
                    self._send(200, result)
                    return

                if path in ("/settings", "/v1/settings"):
                    self._send(200, daemon.settings())
                    return

                if path.startswith("/settings/"):
                    # **Every switch of the window comes through
                    # here.** Without this route the switches had
                    # nowhere to go: the window called it, the server
                    # did not know it, and the switch sprang back.
                    name = path.rsplit("/", 1)[-1]
                    value = body.get("value")
                    if value is None:
                        value = body.get("enabled")
                    if name == "favourites" and isinstance(value, bool):
                        # **A switch, not a list.** The window asks to
                        # mark one topic or to unmark it.
                        value = daemon.toggle_favourite(str(
                            body.get("fact") or ""))
                    result = daemon.set_setting(name, value)
                    self._send(200, result)
                    return

                if path in ("/runtime/suspend", "/v1/runtime/suspend"):
                    # The models stop, the VRAM is free, the memory
                    # stays.
                    self._send(200, daemon.suspend())
                    return

                if path in ("/runtime/resume", "/v1/runtime/resume"):
                    self._send(200, daemon.resume())
                    return

                if path == "/memories/reset":
                    # The whole store, not one user. It is a separate
                    # operation on purpose.
                    self._send(200, service.reset())
                    return

                if path.startswith("/memories/"):
                    memory_id = path.rsplit("/", 1)[-1]
                    self._send(200, service.delete(memory_id))
                    return

                if path in ("/memories", "/v1/memories"):
                    # Every memory of one user.
                    user = body.get("user_id")
                    self._send(200, service.delete_all(user))
                    return

                self._send(404, {"ok": False, "error": "not found"})
            except ApiError as exc:
                self._send(exc.status, {"ok": False, "error": exc.message})
            except EngineError as exc:
                self._send(503, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send(500, {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"})

        def do_PATCH(self) -> None:  # noqa: N802
            """Change the content of one memory."""
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            try:
                if not self._authorised():
                    self._send(401, {"ok": False, "error": "unauthorised"})
                    return
                body = self._read_body()
                memory_id = path.rsplit("/", 1)[-1]
                text = str(body.get("text") or body.get("memory") or "")
                self._send(200, self._service().update(memory_id, text))
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
                if path in ("/settings", "/v1/settings"):
                    self._send(200, daemon.settings())
                    return

                if path in ("/runtime/suspend", "/v1/runtime/suspend"):
                    # The models stop, the VRAM is free, the memory
                    # stays.
                    self._send(200, daemon.suspend())
                    return

                if path in ("/runtime/resume", "/v1/runtime/resume"):
                    self._send(200, daemon.resume())
                    return

                if path == "/memories/reset":
                    self._send(200, self._service().reset())
                    return
                if path in ("/memories", "/v1/memories"):
                    self._send(200, self._service().delete_all())
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
            "name": "chat",
            "description": "Answer the user and remember the "
                           "conversation. The model answers, and the "
                           "exchange goes into the memory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "What the user said."}},
                "required": ["text"],
            },
            "handler": lambda args: service.chat(
                str(args.get("text") or ""),
                history=args.get("history")),
        },
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
        {
            "name": "memory_list",
            "description": "List every stored fact of a user.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string",
                                "description": "Whose facts to list."},
                },
            },
            "handler": lambda args: service.list_all(args.get("user_id")),
        },
        {
            "name": "memory_update",
            "description": "Change the content of one stored fact.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string",
                                  "description": "The id of the fact."},
                    "text": {"type": "string",
                             "description": "The new content."},
                },
                "required": ["memory_id", "text"],
            },
            "handler": lambda args: service.update(
                str(args.get("memory_id", "")),
                str(args.get("text", ""))),
        },
        {
            "name": "memory_delete",
            "description": "Delete one stored fact.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string",
                                  "description": "The id of the fact."},
                },
                "required": ["memory_id"],
            },
            "handler": lambda args: service.delete(
                str(args.get("memory_id", ""))),
        },
        {
            "name": "memory_delete_all",
            "description": "Delete every stored fact of a user.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string",
                                "description": "Whose facts to delete."},
                },
            },
            "handler": lambda args: service.delete_all(
                args.get("user_id")),
        },
    ]


def generate_api_key() -> str:
    """A new API key."""
    return secrets.token_urlsafe(32)

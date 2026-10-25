"""Command line interface.

``cpt-mem0 plan`` is the command that matters: it computes the
configuration for the local hardware and prints it without starting
anything. Every other command is a thin wrapper around the same data.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .catalogue import (
    Catalogue,
    default_catalogue_path,
    load_catalogue,
)
from .hardware import detect_hardware
from .planner import Hardware, PlannerConfig, PlanResult, plan_for_hardware

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NO_VIABLE_CONFIG = 3


def _load_catalogue(path: str | None) -> Catalogue:
    if path:
        return load_catalogue(path)
    default = default_catalogue_path()
    if default.is_file():
        return load_catalogue(default)
    return Catalogue(version=1, updated="", entries=[])


def _format_plan(result: PlanResult, hardware: Hardware) -> str:
    lines: list[str] = []
    lines.append(f"GPU     {hardware.name or 'unknown'}".ljust(42)
                 + f"{hardware.total_vram_mib} MiB total")
    if result.viable and result.candidate:
        c = result.candidate
        lines.append(f"LLM     {c.model_id} {c.quantisation}".ljust(42)
                     + f"{c.file_size_mib:.0f} MiB")
        lines.append(f"        weights {c.weights_vram_mib:.0f} MiB  "
                     f"KV {c.kv_vram_mib:.0f} MiB "
                     f"({c.kv_bytes_per_token} B/tok, {c.cache_type})")
        lines.append(f"        n_ctx {c.n_ctx}  ({c.n_ctx_slot} per slot, "
                     f"{c.parallel} slots)")
        # Print the flags that are actually used. A flag shown here that
        # does not exist in the target build would make the server abort
        # during argument parsing, before it loads anything.
        lines.append("        --fit off  --flash-attn on  "
                     "--reasoning off  --kv-unified")
        if c.recurrent_state_mib > 0:
            lines.append(f"        recurrent state     {c.recurrent_state_mib:.0f} MiB "
                         f"(hybrid model, fixed)")
        lines.append(f"        ranking basis: {c.model_rank_basis}")
        lines.append(f"EMBED   {c.embedder_backend}".ljust(42)
                     + f"{'0 MiB (in-process ONNX)' if c.embedder_backend == 'fastembed' else 'on GPU'}")
        if result.budget:
            b = result.budget
            lines.append(f"TOTAL   {b.used_mib:.0f} MiB used  +  "
                         f"{b.reserve_mib} reserve  +  {b.headroom_mib} headroom"
                         f"  =  {b.used_mib + b.reserve_mib + b.headroom_mib:.0f}"
                         f" / {b.total_mib:.0f}")
    else:
        diag = result.diagnostics
        lines.append(f"NO VIABLE CONFIGURATION: {diag.get('reason', 'unknown')}")
        if diag.get("shortfall_mib"):
            lines.append(f"        short by {diag['shortfall_mib']} MiB")
        for option in diag.get("options", []):
            lines.append(f"        - {option}")
    if result.diagnostics.get("degraded"):
        lines.append("")
        lines.append(f"WARNING: {result.diagnostics.get('warning', '')}")
    return "\n".join(lines)


def _plan_json(result: PlanResult, hardware: Hardware) -> dict:
    data: dict = {
        "gpu": {
            "name": hardware.name,
            "vendor": hardware.vendor,
            "total_mib": hardware.total_vram_mib,
        },
        "viable": result.viable,
    }
    if result.viable and result.candidate:
        c = result.candidate
        data["llm"] = {
            "model": c.model_id,
            "quantisation": c.quantisation,
            "file": c.file_name,
            "size_mib": round(c.file_size_mib),
            "weights_vram_mib": round(c.weights_vram_mib),
            "kv_vram_mib": round(c.kv_vram_mib),
            "kv_bytes_per_token": c.kv_bytes_per_token,
            "cache_type": c.cache_type,
            "n_ctx": c.n_ctx,
            "n_ctx_slot": c.n_ctx_slot,
            "parallel": c.parallel,
            "rank_basis": c.model_rank_basis,
            "flags": c.as_flags(),
        }
        data["embedder"] = {"backend": c.embedder_backend}
    if result.budget:
        b = result.budget
        data["budget"] = {
            "total_mib": round(b.total_mib),
            "used_mib": round(b.used_mib),
            "reserved_mib": b.reserve_mib,
            "headroom_mib": b.headroom_mib,
            "free_mib": round(b.free_mib),
        }
    if result.diagnostics:
        data["diagnostics"] = result.diagnostics
    return data


def cmd_plan(args: argparse.Namespace) -> int:
    hardware = None
    if args.vram:
        hardware = Hardware(total_vram_mib=int(args.vram), name=args.gpu_name or "")
    else:
        hardware = detect_hardware()
    if hardware is None:
        print("No GPU detected. Pass --vram to plan for a specific card.",
              file=sys.stderr)
        return EXIT_ERROR

    catalogue = _load_catalogue(args.catalogue)
    config = PlannerConfig(
        reserve_mib=args.reserve,
        headroom_mib=args.headroom,
        allow_degraded=args.allow_degraded,
    )
    result = plan_for_hardware(hardware, catalogue.sorted_entries(), config)

    if args.json:
        print(json.dumps(_plan_json(result, hardware), indent=2))
    else:
        print(_format_plan(result, hardware))
    return EXIT_OK if result.viable else EXIT_NO_VIABLE_CONFIG


def cmd_models(args: argparse.Namespace) -> int:
    catalogue = _load_catalogue(args.catalogue)
    if args.json:
        print(json.dumps(catalogue.to_dict(), indent=2))
        return EXIT_OK
    print(f"{'model':26} {'params':>8} {'quant':>10} {'MiB':>9}  basis")
    for entry in catalogue.sorted_entries():
        for quant in entry.sorted_quantisations():
            print(f"{entry.model_id:26} {entry.params/1e9:7.1f}B "
                  f"{quant.name:>10} {quant.file_size_mib:9.0f}  "
                  f"{quant.kv_source}")
    return EXIT_OK


def cmd_validate_catalogue(args: argparse.Namespace) -> int:
    catalogue = _load_catalogue(args.catalogue)
    problems: list[str] = []
    for entry in catalogue.entries:
        if not entry.quantisations:
            problems.append(f"{entry.model_id}: no quantisations")
        derived = entry.kv_bytes_from_geometry()
        for quant in entry.quantisations:
            if quant.kv_source == "calibrate" and quant.kv_bytes_per_token:
                problems.append(
                    f"{entry.model_id}/{quant.name}: marked calibrate but has a constant")
            if quant.kv_source in ("measured", "formula") and not quant.kv_bytes_per_token:
                problems.append(
                    f"{entry.model_id}/{quant.name}: {quant.kv_source} but no constant")
            if derived is not None and quant.kv_bytes_per_token != derived:
                problems.append(
                    f"{entry.model_id}/{quant.name}: constant "
                    f"{quant.kv_bytes_per_token} != geometry {derived}")
    for problem in problems:
        print(f"  {problem}")
    if problems:
        return EXIT_ERROR
    print(f"catalogue OK: {len(catalogue.entries)} models, "
          f"{sum(len(e.quantisations) for e in catalogue.entries)} quantisations")
    return EXIT_OK


def cmd_runtime(args) -> int:
    """Report the installed llama.cpp build."""
    from . import install as installer

    found = installer.current_installation()
    builds = installer.list_builds()
    print(f"runtime directory: {installer.runtime_dir()}")
    if builds:
        print(f"builds on disk   : {', '.join(builds)}")
    if not found.installed:
        print("no build is installed - run 'cpt-mem0 init'")
        return EXIT_NO_VIABLE_CONFIG
    print(f"build in use     : {found.build}")
    print(f"binary           : {found.binary}")
    print("CUDA runtime     : "
          + ("present" if found.has_cuda_runtime else "ABSENT"))
    if not found.has_cuda_runtime:
        print()
        print("Without the CUDA runtime the server falls back to the CPU "
              "silently: it starts, it answers, and it computes at a "
              "fraction of the speed. Install the cudart- variant.")
    return EXIT_OK


def cmd_init(args) -> int:
    """Install the pinned llama.cpp build and write the config.

    The runtime and the configuration are two separate things, but
    they belong to the same command: a fresh install needs both, and
    a user who runs 'init' expects a working setup afterwards.
    """
    from . import install as installer
    from .runtime import Platform, detect_cuda_version, detect_platform

    target = detect_platform()
    if args.cpu:
        target = Platform(target.system, target.machine, False, "")
    elif args.cuda_version:
        target = Platform(target.system, target.machine, True,
                          args.cuda_version)
    elif target.cuda:
        version = detect_cuda_version()
        if version:
            target = Platform(target.system, target.machine, True, version)

    print(f"platform: {target.system}-{target.machine}"
          + (f", CUDA {target.cuda_version}" if target.cuda
             else ", CPU only"))

    if not args.no_runtime:
        try:
            result = installer.install(
                getattr(args, "build", None) or None,
                target=target,
                archive=Path(args.archive) if args.archive else None)
        except LookupError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERROR

        print()
        print(f"the runtime is ready: {result.binary}")

    if not args.no_config:
        code = _write_config(args, target)
        if code != EXIT_OK:
            return code

    return EXIT_OK


def _write_config(args, target) -> int:
    """Plan the configuration and write it.

    The planner decides the model, the quantisation and the context.
    The flags are taken from the plan, never written by hand: a wrong
    flag means a wrong context, and the context is what decides whether
    the service can extract facts at all.
    """
    from .config import load_config, save_config
    from .hardware import detect_hardware

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    if args.model_file:
        config.llm.model_file = Path(args.model_file)
    if args.embedder_file:
        config.embedder.model_file = Path(args.embedder_file)
        config.embedder.backend = "llamacpp"

    if not config.llm.model_file:
        print()
        print("The runtime is installed. To write the configuration, the "
              "model file is needed:")
        print("  cpt-mem0 init --model-file <path to the GGUF>")
        print()
        print("Run 'cpt-mem0 plan' to see what fits on the card.")
        return EXIT_OK

    hardware = detect_hardware()
    if hardware is None:
        print("error: the GPU could not be read. The planner needs to "
              "know how much VRAM the card has.", file=sys.stderr)
        return EXIT_ERROR
    catalogue = _load_catalogue(args.catalogue)
    entries = catalogue.sorted_entries()

    if args.model:
        entries = [e for e in entries if e.model_id == args.model]
        if not entries:
            print(f"error: the model {args.model!r} is not in the "
                  f"catalogue", file=sys.stderr)
            return EXIT_ERROR

    planner_config = PlannerConfig(
        embedder_backends=(config.embedder.backend,)
        if config.embedder.backend != "auto" else ("llamacpp", "fastembed"),
        allow_degraded=config.planner.allow_degraded,
    )
    result = plan_for_hardware(hardware, entries, planner_config)
    if not result.viable or result.candidate is None:
        print("error: no viable configuration on this card:",
              result.diagnostics.get("reason"), file=sys.stderr)
        return EXIT_NO_VIABLE_CONFIG

    candidate = result.candidate
    config.llm.model = candidate.model_id
    config.llm.quantisation = candidate.quantisation
    config.llm.context = candidate.n_ctx
    config.llm.flags = candidate.as_flags()
    config.embedder.backend = candidate.embedder_backend

    if config.embedder.backend == "llamacpp" and not config.embedder.model_file:
        print()
        print("The embedder model is needed for the GPU backend:")
        print("  cpt-mem0 init --embedder-file <path to the GGUF>")
        return EXIT_OK

    if config.embedder.backend == "llamacpp":
        config.embedder.flags = [
            "--parallel", "1",
            "-c", str(config.embedder.context),
            "-ngl", "99",
            "--embeddings",
            "--pooling", "mean",
            "--kv-unified",
        ]

    path = save_config(config, config_path)
    print()
    print(f"the configuration is written to {path}")
    print(f"  LLM      {candidate.model_id} {candidate.quantisation}, "
          f"n_ctx {candidate.n_ctx}")
    print(f"  embedder {candidate.embedder_backend}")
    print(f"  store    {config.store.path}")
    return EXIT_OK


def cmd_update(args) -> int:
    """Install a newer build, or go back to the previous one."""
    from . import install as installer

    if args.rollback:
        result = installer.rollback()
        if result is None:
            return EXIT_NO_VIABLE_CONFIG
        print(f"the build in use is now {result.build}")
        return EXIT_OK

    if args.check:
        found = installer.current_installation()
        print(f"build in use: {found.build or 'nothing'}")
        print("builds on disk: "
              + (", ".join(installer.list_builds()) or "none"))
        print()
        print("A newer build is installed with 'cpt-mem0 update --build "
              "<tag>'.")
        print("The tag pattern is b<number>, for example b11838.")
        return EXIT_OK

    try:
        installed, previous = installer.update(
            getattr(args, "build", None) or None, dry_run=False)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print()
    print(f"the build in use is now {installed.build}")
    if previous is not None and previous.build:
        print(f"the previous build ({previous.build}) is kept - "
              f"'cpt-mem0 update --rollback' goes back to it")
    return EXIT_OK


def cmd_start(args) -> int:
    """Start the daemon."""
    from .config import load_config
    from .daemon import Daemon, DaemonPaths
    from .install import data_dir

    config = load_config(Path(args.config) if args.config else None)

    if not config.llm.model_file:
        print("error: no model is configured. Run 'cpt-mem0 init' "
              "to write the configuration.", file=sys.stderr)
        return EXIT_NO_VIABLE_CONFIG

    paths = DaemonPaths(logs=data_dir() / "logs", state=data_dir() / "state")
    daemon = Daemon(config, paths)

    try:
        daemon.start(timeout=args.timeout)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        daemon.stop()
        return EXIT_ERROR

    print(f"the daemon is serving on "
          f"http://{config.server.host}:{config.server.port}")
    for entry in daemon.status()["servers"]:
        state = "up" if entry["running"] else "down"
        print(f"  {entry['name']:10} {state:5} port {entry['port']}")

    if args.foreground:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            daemon.stop()
        return EXIT_OK

    # In the background the daemon keeps running. The process is
    # detached from the console.
    _detach()
    return EXIT_OK


def _detach() -> None:
    """Let the daemon run after the command returns."""
    try:
        if os.name == "nt":
            subprocess.Popen(
                [sys.executable, "-m", "cpt_mem0.cli", "start", "--foreground"],
                creationflags=subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(
                [sys.executable, "-m", "cpt_mem0.cli", "start", "--foreground"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
    except OSError:
        # The detached process could not be started. The daemon in this
        # process keeps running until it is stopped.
        pass


def cmd_stop(args) -> int:
    """Stop the daemon."""
    from .install import data_dir

    pid_file = data_dir() / "state" / "daemon.pid"
    if not pid_file.is_file():
        print("the daemon is not running - there is no pid file")
        return EXIT_NO_VIABLE_CONFIG

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        print("the pid file is unreadable")
        pid_file.unlink(missing_ok=True)
        return EXIT_ERROR

    if not _process_alive(pid):
        print(f"the daemon (pid {pid}) is not running - removing the "
              f"stale pid file")
        pid_file.unlink(missing_ok=True)
        return EXIT_NO_VIABLE_CONFIG

    print(f"stopping the daemon (pid {pid})")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # The daemon has to release the store lock before it exits.
    for _ in range(60):
        if not _process_alive(pid):
            break
        time.sleep(0.5)
    else:
        print("the daemon did not stop in 30s - killing it")
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=15)
            else:
                os.kill(pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass

    pid_file.unlink(missing_ok=True)
    return EXIT_OK


def _process_alive(pid: int) -> bool:
    """Whether a process is running.

    **tasklist writes the system encoding, not UTF-8.** With the
    default the reader thread raises a UnicodeDecodeError, the output
    comes back as None, and ``str(pid) in None`` fails with a
    TypeError about None not being iterable. The error names neither
    the command nor the encoding - it looks like a bug in the status
    command.
    """
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=10)
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
        output = result.stdout or ""
        return str(pid) in output
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def cmd_ui(args) -> int:
    """Start the daemon and open the window.

    The window is served by the daemon. It is a client: it never
    opens the store itself, the store has an exclusive lock and the
    daemon is the only one that may hold it.

    The native window is a WebView2 - the engine Edge already ships.
    Without pywebview the same three files open in the system
    browser; the interface is the same either way.

    With ``--suspend`` the daemon comes up with the models off. The
    API is reachable, the memory is there, and the VRAM stays free
    until the big switch is pressed.
    """
    from .config import Config
    from .daemon import Daemon, DaemonPaths
    from .window import (
        run_window,
        setup_logging,
        start_daemon_in_thread,
        wait_for_address,
    )

    # The packaged app has no console. Everything that goes wrong
    # around the window goes into the log file.
    setup_logging()

    config = Config()
    daemon = Daemon(config, DaemonPaths.default())

    start_daemon_in_thread(daemon, suspend=args.suspend)

    address = wait_for_address(daemon)
    if not address:
        print("the daemon did not come up", file=sys.stderr)
        return 1

    print(f"the window is at {address}")

    try:
        return run_window(daemon, address, open_browser=args.no_browser)
    except KeyboardInterrupt:
        print("stopping")
        daemon.stop()
        return 0


def cmd_status(args) -> int:
    """Report the state of the daemon."""
    from .install import data_dir
    from .supervisor import gpu_used_mib

    pid_file = data_dir() / "state" / "daemon.pid"
    running = False
    pid = None
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            running = _process_alive(pid)
        except (OSError, ValueError):
            running = False

    print(f"daemon:    {'running' if running else 'not running'}"
          + (f" (pid {pid})" if running and pid else ""))
    print(f"VRAM:      {gpu_used_mib()} MiB used")

    config_path = Path(args.config) if args.config else None
    try:
        from .config import load_config

        config = load_config(config_path)
    except Exception as exc:
        print(f"config:    unreadable: {exc}")
        return EXIT_ERROR

    print(f"config:    {config_path or 'default'}")
    print(f"server:    http://{config.server.host}:{config.server.port}")
    print(f"LLM:       {config.llm.model or 'not configured'} "
          f"{config.llm.quantisation or ''}")
    print(f"embedder:  {config.embedder.backend} "
          f"({config.embedder.dimensions} dimensions)")
    print(f"store:     {config.store.path}")
    return EXIT_OK if running else EXIT_NO_VIABLE_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpt-mem0",
        description="Self-contained local mem0 memory service for consumer GPUs.",
    )
    parser.add_argument("--catalogue", help="path to catalogue.json")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="compute the configuration, start nothing")
    plan.add_argument("--vram", type=int, help="plan for a card with this much VRAM (MiB)")
    plan.add_argument("--gpu-name", help="name to report in the output")
    plan.add_argument("--reserve", type=int, help="VRAM to reserve (MiB)")
    plan.add_argument("--headroom", type=int, help="VRAM headroom (MiB)")
    plan.add_argument("--allow-degraded", action="store_true",
                      help="permit configurations below the context floor")
    plan.add_argument("--json", action="store_true", help="machine-readable output")
    plan.set_defaults(func=cmd_plan)

    models = sub.add_parser("models", help="show the model catalogue")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=cmd_models)

    validate = sub.add_parser("validate-catalogue",
                              help="check the catalogue for internal consistency")
    validate.set_defaults(func=cmd_validate_catalogue)

    runtime = sub.add_parser("runtime",
                             help="show the installed llama.cpp build")
    runtime.set_defaults(func=cmd_runtime)

    init = sub.add_parser(
        "init", help="install the pinned llama.cpp build and write the config")
    init.add_argument("--build", help="build to install (default: the pinned one)")
    init.add_argument("--archive", help="install from a local archive")
    init.add_argument("--cpu", action="store_true",
                      help="install the CPU-only build")
    init.add_argument("--cuda-version", help="CUDA version to build for")
    init.add_argument("--config", help="path to the config file")
    init.add_argument("--catalogue", help="path to the model catalogue")
    init.add_argument("--model", help="model id from the catalogue")
    init.add_argument("--model-file", help="path to the GGUF file")
    init.add_argument("--embedder-file", help="path to the embedding model")
    init.add_argument("--no-runtime", action="store_true",
                      help="only write the config, do not touch the runtime")
    init.add_argument("--no-config", action="store_true",
                      help="only install the runtime, do not write the config")
    init.set_defaults(func=cmd_init)

    update = sub.add_parser("update", help="install a newer build and switch to it")
    update.add_argument("--build", help="build to install")
    update.add_argument("--check", action="store_true",
                        help="report only, change nothing")
    update.add_argument("--rollback", action="store_true",
                        help="go back to the previous build")
    update.set_defaults(func=cmd_update)

    start = sub.add_parser("start", help="start the daemon")
    start.add_argument("--config", help="path to the config file")
    start.add_argument("--foreground", action="store_true",
                       help="run in the foreground, do not detach")
    start.add_argument("--timeout", type=int, default=180,
                       help="how long to wait for the servers")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="stop the daemon")
    stop.set_defaults(func=cmd_stop)

    ui = sub.add_parser(
        "ui", help="open the window")
    ui.add_argument("--no-browser", action="store_true",
                    help="open the system browser instead of the native "
                         "window")
    ui.add_argument("--suspend", action="store_true",
                    help="start the daemon with the models off")
    ui.set_defaults(func=cmd_ui)

    status = sub.add_parser("status",
                             help="report the state of the daemon")
    status.add_argument("--config", help="path to the config file")
    status.set_defaults(func=cmd_status)

    return parser


def _log_problem(message: str) -> None:
    """Write a problem into the log file, if there is one.

    The packaged app has no console, so a message that goes to stderr
    is lost.
    """
    try:
        from .window import log_problem

        log_problem(message)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    # A bare flag belongs to the window: the app is a window, and
    # ``Cpt-Mem0.exe --suspend`` means ``ui --suspend``.
    #
    # **Only the flags of the window.** ``--help`` belongs to the
    # main command, and without this the exe would print the help of
    # the window instead of the list of the commands.
    window_flags = ("--suspend", "--no-browser")
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in window_flags:
        argv = ["ui"] + list(argv)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _log_problem(f"error: {exc}")
        return EXIT_ERROR
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        _log_problem(f"error: {type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

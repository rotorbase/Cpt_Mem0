"""The native window.

The window is a client of the daemon. It never opens the store
itself - Qdrant's embedded mode allows one client per directory, so
the store has an exclusive lock and the daemon is the only one that
may hold it.

**The window is a WebView2.** That is the engine Edge already ships
and every Windows 10 and 11 has it. So the app does not bring its own
Chromium: the package stays small and the rendering is the system's.

**Without pywebview the browser is used.** The daemon serves the same
three files either way. The fallback is not a degradation of the
feature, it is the same interface - only the frame around it is
different. That is what keeps the development loop working on a
machine without the package.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

#: Where the window problems are written down.
#:
#: **The packaged app has no console.** A traceback that goes to
#: stdout disappears, and the user sees a window that blinks and
#: nothing else. So every error around the window goes into a file.
LOG_FILENAME = "window.log"

#: What the module returns when it had to open the browser. The
#: command then keeps the daemon alive - the browser needs a moment,
#: and if the app ended in the meantime the daemon would go with it.
BROWSER_FALLBACK = 42

#: The title of the window.
WINDOW_TITLE = "Cpt Mem0"

#: The size of the window. The window is wider than it is high -
#: the chat line is a line, not a page.
WINDOW_SIZE = (1180, 760)

#: The smallest size. Below that the topic buttons and the chat
#: cannot both be seen.
WINDOW_MIN_SIZE = (820, 560)


def log_path() -> Any:
    """The file the window problems go into."""
    from pathlib import Path

    from .install import data_dir

    try:
        base = data_dir() / "logs"
        base.mkdir(parents=True, exist_ok=True)
        return base / LOG_FILENAME
    except OSError:
        return Path(LOG_FILENAME)


def setup_logging() -> None:
    """Write what pywebview says into a file.

    pywebview logs the reason a window does not come up - the WebView2
    runtime, the .NET host, the browser executable. Without this the
    packaged app only shows that something went wrong.
    """
    target = log_path()
    try:
        handler = logging.FileHandler(target, encoding="utf-8")
    except OSError:
        return
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger = logging.getLogger("pywebview")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # The .NET side of pywebview logs through the root logger.
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(handler)
        root.setLevel(logging.INFO)


def log_problem(message: str) -> None:
    """Write a problem into the log file."""
    try:
        with open(log_path(), "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def log_exception(message: str) -> None:
    """Write a problem with its traceback into the log file."""
    log_problem(message)
    try:
        with open(log_path(), "a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
    except OSError:
        pass


def webview_available() -> bool:
    """Whether the native window can be opened."""
    try:
        import webview  # noqa: F401
    except ImportError:
        return False
    return True


def _load_webview2_assemblies() -> list[str]:
    """Load the WebView2 assemblies.

    They are not part of .NET. pywebview brings them as .NET
    assemblies next to its own files, and the runtime does not find
    them by name - so they are loaded by their path.
    """
    import webview

    folder = Path(webview.__file__).resolve().parent / "lib"
    if not folder.is_dir():
        return []

    import clr

    loaded: list[str] = []
    for path in sorted(folder.glob("*.dll")):
        try:
            clr.AddReference(str(path))
            loaded.append(path.name)
        except Exception:
            pass
    return loaded


def _runtimeconfig_path() -> Path | None:
    """Where the .NET runtime configuration of the app is.

    It is next to the exe - in the source tree next to the package,
    in the packaged app next to the exe.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "Cpt_Mem0.runtimeconfig.json"
    return Path(__file__).resolve().parent.parent.parent / "packaging" / \
        "runtimeconfig.json"


def _load_windows_desktop() -> list[str]:
    """Load the assemblies of the WindowsDesktop framework.

    **WinForms is not in the .NET Core framework.** It lives in
    Microsoft.WindowsDesktop.App, and clr_loader writes a runtime
    configuration with exactly one framework - so the runtime comes up
    without WinForms, and the window cannot be created.

    The assemblies are loaded by their path instead. The path is
    found through the registry, so it works on every machine that has
    the runtime - no environment variable, no configuration file.

    An assembly that does not fit is skipped: some of them are
    built for another version, and WinForms does not need all of
    them.
    """
    loaded: list[str] = []
    folder = _windows_desktop_folder()
    if folder is None:
        return loaded

    import clr

    # Every assembly of the folder, not a hand-picked list: pywebview
    # asks for several of them by name, and the list changes with
    # the version. One that does not fit is skipped - it is not
    # needed by everything.
    for path in sorted(folder.glob("*.dll")):
        try:
            clr.AddReference(str(path))
            loaded.append(path.name)
        except Exception:
            pass
    return loaded


def _windows_desktop_folder() -> Path | None:
    """Where the WindowsDesktop framework is installed."""
    if sys.platform != "win32":
        return None

    roots: list[Path] = []
    dotnet_root = os.environ.get("DOTNET_ROOT")
    if dotnet_root:
        roots.append(Path(dotnet_root))
    try:
        import winreg

        for hive, key in (
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SOFTWARE\dotnet\Setup\InstalledVersions\x64\InstallLocation"),
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SOFTWARE\dotnet\Setup\InstalledVersions\x86\InstallLocation")):
            try:
                with winreg.OpenKey(hive, key) as handle:
                    value, _ = winreg.QueryValueEx(handle, "InstallLocation")
                if value:
                    roots.append(Path(value))
            except OSError:
                continue
    except ImportError:
        pass

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        roots.append(Path(program_files) / "dotnet")

    for root in roots:
        shared = root / "shared" / "Microsoft.WindowsDesktop.App"
        if not shared.is_dir():
            continue
        versions = sorted((d for d in shared.iterdir() if d.is_dir()),
                          key=lambda d: _version_key(d.name), reverse=True)
        for version in versions:
            if (version / "System.Windows.Forms.dll").is_file():
                return version
    return None


def _version_key(name: str) -> tuple:
    parts = []
    for piece in name.split("."):
        parts.append(int(piece) if piece.isdigit() else 0)
    return tuple(parts)


def _installed_dotnet_root() -> Path | None:
    """Where the .NET runtime is installed.

    The registry knows it, and that is the way that works on a
    machine where nobody set an environment variable. The
    development machine and the packaged app must look in the same
    place - the registry is the same on both.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None

    for hive, key in (
            (winreg.HKEY_LOCAL_MACHINE,
             r"SOFTWARE\dotnet\Setup\InstalledVersions\x64\InstallLocation"),
            (winreg.HKEY_LOCAL_MACHINE,
             r"SOFTWARE\dotnet\Setup\InstalledVersions\x86\InstallLocation")):
        try:
            with winreg.OpenKey(hive, key) as handle:
                value, _ = winreg.QueryValueEx(handle, "InstallLocation")
            if value:
                return Path(value)
        except OSError:
            continue
    return None


def open_window(address: str, title: str = WINDOW_TITLE,
                on_close=None) -> int:
    """Open the native window on the address.

    Returns the exit code of the app. The browser fallback returns 0
    as well - it does not keep the process alive.
    """
    if not webview_available():
        log_problem("pywebview is not installed")
        return _no_window(address)

    # **The .NET Core runtime, not the .NET Framework.**
    #
    # pythonnet picks the .NET Framework by default. The framework
    # loader resolves the framework through mscoree, and in the
    # frozen app it fails with "Failed to resolve
    # Python.Runtime.Loader.Initialize" - the bundled Python.Runtime.dll
    # is the .NET Core one, and that entry point does not exist in the
    # framework. With coreclr the .NET Core runtime is loaded, the
    # entry point exists, and the window opens.
    #
    # The WebView2 runtime is a different thing: it is the browser
    # engine, and it is in Windows already.
    os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")

    # **The .NET runtime that is installed, not the one that is
    # bundled.** clr_loader brings its own .NET Core runtime, and
    # that one has no WindowsDesktop - so WinForms cannot be found
    # and the window fails. The installed runtime is found through
    # the registry; DOTNET_ROOT is the way to say it explicitly.
    # **DOTNET_ROOT is always set.** Without it clr_loader runs
    # "dotnet --list-runtimes" - a command with a console window,
    # and the window flickers over the desktop every time the app
    # starts. With the variable the runtimes are found in the
    # directory, without a command.
    dotnet_root = _installed_dotnet_root()
    if dotnet_root:
        os.environ["DOTNET_ROOT"] = str(dotnet_root)
        log_problem(f"DOTNET_ROOT: {dotnet_root}")

    # **The runtime configuration of the .NET host.**
    #
    # pythonnet documents it: to use coreclr, the variable
    # PYTHONNET_CORECLR_RUNTIME_CONFIG has to point at a valid
    # runtimeconfig.json. Without it clr_loader writes its own - and
    # that one names exactly one framework, Microsoft.NETCore.App.
    # WinForms is in Microsoft.WindowsDesktop.App, so the window
    # cannot be created.
    #
    # The file is next to the exe, so the path works wherever the
    # user unpacked the app.
    config = _runtimeconfig_path()
    if config is not None and config.is_file():
        os.environ["PYTHONNET_CORECLR_RUNTIME_CONFIG"] = str(config)
        log_problem(f"die .NET-Konfiguration: {config}")
    else:
        log_problem("keine .NET-Konfiguration gefunden")

    # The runtime is loaded here and not inside pywebview: the error
    # then says what is wrong, instead of "pythonnet cannot be
    # loaded" - which says nothing.
    try:
        import clr  # noqa: F401

        log_problem("die .NET-Runtime ist geladen")

        # **The assemblies of the WindowsDesktop framework.**
        #
        # Some of them are not part of the framework itself but live
        # next to it: Microsoft.Win32.SystemEvents is one of them,
        # and pywebview asks for it by name. The runtime only finds
        # the ones the framework configuration names, so the others
        # are loaded by their path.
        loaded = _load_windows_desktop()
        log_problem(f"die WindowsDesktop-Assemblies: {len(loaded)}")
    except Exception as error:
        log_exception(f"die .NET-Runtime konnte nicht geladen werden: "
                      f"{error}")

    import webview

    # **The WebView2 assemblies.**
    #
    # They are not part of .NET - they are the browser engine of
    # Windows, and pywebview brings them as .NET assemblies next to
    # its own files. The runtime does not find them by name, so they
    # are loaded by their path.
    try:
        import clr

        loaded = _load_webview2_assemblies()
        log_problem(f"die WebView2-Assemblies: {loaded}")
    except Exception as error:
        log_exception(f"die WebView2-Assemblies: {error}")

    log_problem(f"the window is opened on {address}")
    window = webview.create_window(
        title,
        address,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN_SIZE,
        background_color="#0a0b0d",
        text_select=True,
    )

    def closed() -> None:
        # The window is gone, so the daemon has nothing to serve.
        if on_close is not None:
            on_close()

    window.events.closed += closed
    try:
        webview.start()
    except Exception as error:
        log_exception(f"the native window failed: {error}")
        raise
    log_problem("the native window was closed")
    return 0


def _no_window(address: str) -> int:
    """The window could not be opened.

    **No browser.** The app is a program, not a link: it opens a
    window of its own, and if that fails the user gets a message
    with the address and the log file - not a browser tab that cannot
    connect and flickers over the desktop.
    """
    log_problem(f"the window could not be opened - the address is "
                f"{address}")
    print(f"the window could not be opened")
    print(f"the interface is at {address}")
    print(f"the reason is in the log: {log_path()}")
    return BROWSER_FALLBACK


def run_window(daemon: Any, address: str,
               open_browser: bool = False) -> int:
    """Open the window and keep the daemon alive behind it.

    The daemon runs in a thread of this process. The window is the
    main thread, because the window toolkit wants the main thread.
    When the window closes, the daemon stops - the store lock has to
    be released.
    """
    if open_browser:
        return _open_browser(address)

    def on_close() -> None:
        try:
            daemon.stop()
        except Exception:
            pass

    try:
        return open_window(address, on_close=on_close)
    except Exception as error:
        # A window that does not open must not take the daemon with
        # it. The browser still works.
        log_exception(f"the native window could not be opened: {error}")
        print(f"the native window could not be opened: {error}")
        print(f"the interface is at {address}")
        return _no_window(address)


def wait_for_address(daemon: Any, timeout: float = 60.0) -> str:
    """Wait until the daemon answers.

    The window must not open before the server is up - it would show
    an error page.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        address = getattr(daemon, "address", "")
        if address:
            return address
        time.sleep(0.2)
    return getattr(daemon, "address", "")


def start_daemon_in_thread(daemon: Any, suspend: bool = False) -> threading.Thread:
    """Start the daemon in a background thread.

    **The exception of the thread goes into the log.** A daemon that
    does not come up throws inside the thread, the thread ends, and
    nothing is printed anywhere - the app then waits a minute and
    says "the daemon did not come up" without saying why. The
    packaged app has no console, so the log is the only place where
    the reason can be.
    """

    def run() -> None:
        try:
            daemon.start(suspend=suspend)
        except Exception as error:
            log_exception(f"the daemon did not come up: {error}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread

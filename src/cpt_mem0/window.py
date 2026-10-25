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
import threading
import time
import traceback
from typing import Any

#: Where the window problems are written down.
#:
#: **The packaged app has no console.** A traceback that goes to
#: stdout disappears, and the user sees a window that blinks and
#: nothing else. So every error around the window goes into a file.
LOG_FILENAME = "window.log"

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


def open_window(address: str, title: str = WINDOW_TITLE,
                on_close=None) -> int:
    """Open the native window on the address.

    Returns the exit code of the app. The browser fallback returns 0
    as well - it does not keep the process alive.
    """
    if not webview_available():
        log_problem("pywebview is not installed - the browser is used")
        return _open_browser(address)

    # **The edgechromium backend, not winforms.** The winforms
    # backend needs pythonnet, and pythonnet needs hostfxr.dll - which
    # the frozen app cannot find. The window then fell back to the
    # browser, and the user got a tab instead of a window. The
    # edgechromium backend talks to the WebView2 through ctypes and
    # needs no .NET at all.
    os.environ.setdefault("PYWEBVIEW_GUI", "edgechromium")

    import webview

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


def _open_browser(address: str) -> int:
    """The fallback: the system browser."""
    import webbrowser

    try:
        webbrowser.open(address)
    except Exception:
        print(f"the window is at {address}")
    return 0


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
        print(f"the window is at {address}")
        return _open_browser(address)


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
    """Start the daemon in a background thread."""
    thread = threading.Thread(
        target=daemon.start, kwargs={"suspend": suspend}, daemon=True)
    thread.start()
    return thread

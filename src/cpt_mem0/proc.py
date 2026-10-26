"""Starting child processes without a console window.

**Every child process of this program is a console program.**
`nvidia-smi`, `tasklist`, `schtasks`, the llama.cpp server - all of
them are console programs, and Windows opens a black window for each
of them that it starts.

The daemon calls `nvidia-smi` twice for every status, and the
interface asks for the status every few seconds. The result was a
black window that appeared and disappeared over and over while the
app was open.

The flag that stops it is `CREATE_NO_WINDOW`. It only works when the
process is started directly - **not with `shell=True`**, because then
`cmd.exe` is the one that gets the console, and `cmd.exe` starts the
real program in a new window of its own. So the shell is not used
here at all: the programs are called with their arguments as a list.
"""

from __future__ import annotations

import os
import subprocess

#: Whether this runs on Windows.
IS_WINDOWS = os.name == "nt"


def no_window() -> dict[str, int]:
    """The flags that keep a child process from opening a console.

    Empty on the other systems - there is no console window there.
    """
    if not IS_WINDOWS:
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    """Run a command without a console window.

    The same as `subprocess.run`, with the flags that Windows needs
    so that no black window flashes up.
    """
    flags = no_window()
    flags.update(kwargs)
    return subprocess.run(command, **flags)  # type: ignore[arg-type]

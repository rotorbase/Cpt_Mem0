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

def die_with_me(process: subprocess.Popen) -> None:
    """Bind a child to the life of this program.

    **A program that outlives its app is a program nobody asked
    for.** The speech recognition is started by the daemon and keeps
    a piece of the card - if the window is closed with the cross, the
    daemon is killed without a goodbye, and the recognition stays
    and holds 250 MB of VRAM until the next start.

    Windows has an answer for that: a job object with the flag
    "kill on close". The handle lives in this process; when the
    process dies, the handle is closed, and the job kills everything
    in it.

    Without pywin32 the program stays as it is - it is then stopped
    by the normal path, and only a hard kill leaves it behind.
    """
    if not IS_WINDOWS:
        return
    try:
        import win32api
        import win32con
        import win32job
    except ImportError:
        return
    try:
        job = win32job.CreateJobObject(None, "")
        limits = win32job.QueryInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        win32job.SetInformationJobObject(
            job, win32job.JobObjectExtendedLimitInformation, limits)
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True,
                                      process.pid)
        win32job.AssignProcessToJobObject(job, handle)
        # **The handle has to stay open.** When it is closed, the
        # job kills the process - so it is kept here until the
        # program ends.
        _JOBS.append(job)
    except Exception:  # noqa: BLE001
        # **Without the binding it still works.** The program is
        # stopped by the normal path.
        return


#: The job objects that are kept open.
_JOBS: list[object] = []

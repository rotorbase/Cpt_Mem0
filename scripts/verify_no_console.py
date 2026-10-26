"""Prove that the app does not open a console window.

**The flicker.** A console window on the desktop that appears and
disappears over and over makes an app feel broken - and it is
invisible in the code. `nvidia-smi`, `tasklist` and `schtasks` are
console programs, and the daemon calls them for every status. The
interface asks for the status every few seconds, so the black window
came and went the whole time the app was open.

**The proof has two parts.**

1. A fake `nvidia-smi` stands in front of the real one. It counts
   how often it is called and writes every call into a file. It is a
   `.cmd`, so it is the hardest case: a batch file starts `cmd.exe`,
   and that is where a console window would come from.

2. Every window on the desktop is watched while the app runs. A
   console window has the class `ConsoleWindowClass`.

Both parts have to hold: the calls happen, and no window appears.

    python scripts/verify_no_console.py --seconds 60
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The class of a console window.
CONSOLE_CLASS = "ConsoleWindowClass"

user32 = ctypes.windll.user32
EnumWindowsProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

#: The fake. It answers the three questions the app asks and counts
#: every call.
FAKE = """@echo off
echo %* >> "%CPT_MEM0_FAKE_LOG%"
echo %* | findstr /C:"memory.used" >nul && (echo 8000 & exit /b 0)
echo %* | findstr /C:"memory.free" >nul && (echo 4288 & exit /b 0)
echo %* | findstr /C:"driver_version" >nul && (echo 580.97 & exit /b 0)
exit /b 0
"""


def list_windows() -> list[dict]:
    """Every window on the desktop that has a title."""
    found: list[dict] = []

    def callback(hwnd, _lparam):
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if not title:
            return True
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        found.append({
            "hwnd": hwnd,
            "title": title,
            "class": cls.value,
            "pid": pid.value,
            "visible": bool(user32.IsWindowVisible(hwnd)),
        })
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return found


def process_name(pid: int) -> str:
    """The name of the process behind a window."""
    try:
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return ""
        buffer = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(512)
        ctypes.windll.kernel32.QueryFullProcessImageNameW(
            handle, 0, buffer, ctypes.byref(size))
        ctypes.windll.kernel32.CloseHandle(handle)
        return buffer.value
    except Exception:
        return ""


def app_candidates() -> list[Path]:
    """The programs that could be started."""
    return [
        ROOT / "desktop" / "dist-electron" / "win-unpacked" / "Cpt Mem0.exe",
        ROOT / "dist" / "Cpt_Mem0" / "Cpt_Mem0.exe",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--app", type=Path, default=None)
    arguments = parser.parse_args()

    if os.name != "nt":
        print("this check is about the windows of Windows")
        return 2

    # The fake in front of the real nvidia-smi.
    fake_dir = Path(tempfile.mkdtemp(prefix="cpt-fake-"))
    fake_log = fake_dir / "calls.log"
    (fake_dir / "nvidia-smi.cmd").write_text(FAKE, encoding="utf-8")

    environment = dict(os.environ)
    environment["PATH"] = str(fake_dir) + os.pathsep + environment.get("PATH", "")
    environment["CPT_MEM0_FAKE_LOG"] = str(fake_log)

    app = arguments.app
    if app is None:
        candidates = [path for path in app_candidates() if path.is_file()]
        app = candidates[0] if candidates else None
    if app is None:
        print("no app found")
        return 2

    print(f"starting {app}")
    process = subprocess.Popen(
        [str(app)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW, env=environment)

    before = {entry["hwnd"] for entry in list_windows()}
    seen_console: list[dict] = []
    status_calls = 0
    deadline = time.time() + arguments.seconds

    while time.time() < deadline:
        for entry in list_windows():
            if entry["hwnd"] in before:
                continue
            if entry["class"] == CONSOLE_CLASS:
                key = (entry["class"], entry["title"], entry["pid"])
                if not any(item["key"] == key for item in seen_console):
                    seen_console.append({
                        "key": list(key),
                        "process": process_name(entry["pid"]),
                    })
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:8765/status", timeout=3) as response:
                response.read(64)
                status_calls += 1
        except Exception:
            pass
        time.sleep(0.01)

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    subprocess.run(["taskkill", "/F", "/IM", "Cpt_Mem0.exe"],
                   capture_output=True, timeout=60)

    calls = 0
    if fake_log.is_file():
        calls = len(fake_log.read_text(encoding="utf-8",
                                       errors="replace").splitlines())

    report = {
        "seconds": arguments.seconds,
        "status_calls": status_calls,
        "nvidia_smi_calls": calls,
        "console_windows": seen_console,
    }
    print(json.dumps(report, indent=2))
    shutil.rmtree(fake_dir, ignore_errors=True)

    if seen_console:
        print(f"\nFAILED: {len(seen_console)} console window(s)")
        return 1
    if calls == 0:
        print("\nFAILED: the fake nvidia-smi was never called - "
              "the test proved nothing")
        return 1
    print(f"\nOK: {calls} calls to nvidia-smi, no console window")
    return 0


if __name__ == "__main__":
    sys.exit(main())

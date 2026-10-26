"""The ear of the assistant.

**Dictation runs on the machine.** The microphone of the headset
goes into the app, the app makes text out of it, and the text goes
to the model like a typed question. Nothing is sent anywhere - the
speech recognition is a program on the disk, and it runs on the
processor while the graphics card is busy with the model.

**The parts.** whisper.cpp does the work: a small program that turns
a wave file into words. The build puts it into the package, with a
model of German speech beside it - so the user installs the app and
can talk to it, without downloading anything and without an account.

**The program stays alive.** Loading the model takes seconds, and a
dictation that waits for that is a dictation nobody uses. So the
recognition runs as a server that keeps the model in the memory -
the first word is slow, every word after it is quick.

**The microphone belongs to the user.** The app asks the system for
it when the user presses the button, and it stops when the user
lets go. Nothing is recorded while the button is not pressed.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

#: How long a recognition may take.
TIMEOUT = 120

#: How long the server may need to come up.
START_TIMEOUT = 90

#: The name of the program of whisper.cpp.
SERVER = "whisper-server.exe"

#: The model of the speech. Small and quantized - it understands
#: German well enough for a question, and on the card it needs a
#: fifth of a second.
MODEL = "ggml-small-q5_1.bin"

#: The language of the speech. The user speaks German.
LANGUAGE = "de"

#: Where whisper.cpp is in the package.
BUNDLE_DIR = "whisper"

#: The server that is running, and where it listens.
_server: subprocess.Popen[bytes] | None = None
_address = ""
_lock = threading.Lock()


def whisper_dir() -> Path:
    """Where whisper.cpp lives.

    **In the app first.** The program and the model are part of the
    package - the user does not download an ear, he installs an app
    that has one.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", ""))
        bundled = base / BUNDLE_DIR
        if bundled.is_dir():
            return bundled

    from .install import data_dir

    folder = data_dir() / "whisper"
    if folder.is_dir():
        return folder

    return (Path(__file__).resolve().parent.parent.parent
            / "packaging" / "whisper")


def server_path() -> Path:
    """The program that hears."""
    return whisper_dir() / SERVER


def model_path() -> Path:
    """The model of the speech."""
    return whisper_dir() / MODEL


def available() -> bool:
    """Whether the machine can hear.

    **Both parts or nothing.** The program without the model does
    nothing, and the model without the program does nothing either.
    """
    return server_path().is_file() and model_path().is_file()


def status() -> dict[str, object]:
    """What the app knows about the ear."""
    return {
        "available": available(),
        "binary": str(server_path()),
        "model": str(model_path()),
        "language": LANGUAGE,
        "running": bool(_address and _server and _server.poll() is None),
        "reason": "" if available() else
                  "Die Spracheingabe ist in diesem Build nicht enthalten",
    }


def _free_port() -> int:
    """A port nobody uses."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def server() -> str:
    """The address of the recognition, started when it is needed.

    **The model stays in the memory.** The first word waits for the
    load, every word after it does not.
    """
    global _server, _address
    with _lock:
        if _server is not None and _server.poll() is None and _address:
            return _address

        port = _free_port()
        command = [
            str(server_path()),
            "-m", str(model_path()),
            "--host", "127.0.0.1",
            "--port", str(port),
            "-l", LANGUAGE,
            "-t", str(max(1, os.cpu_count() or 4)),
        ]
        # **No window.** A program that opens a console window for a
        # second is the flicker the user hates.
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        _server = subprocess.Popen(
            command, cwd=str(whisper_dir()),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=flags)

        address = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline:
            if _server.poll() is not None:
                raise RuntimeError("Die Spracheingabe ist nicht gestartet")
            try:
                with urllib.request.urlopen(f"{address}/health", timeout=2):
                    _address = address
                    return address
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        stop()
        raise RuntimeError("Die Spracheingabe ist nicht gestartet")


def stop() -> None:
    """Stop the recognition."""
    global _server, _address
    with _lock:
        if _server is not None:
            try:
                _server.terminate()
                _server.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    _server.kill()
                except Exception:  # noqa: BLE001
                    pass
        _server = None
        _address = ""


def _post(address: str, wave: bytes) -> str:
    """Give the wave file to the recognition."""
    boundary = "----cptmem0diktat"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n").encode("utf-8") + wave + \
        f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        f"{address}/inference", data=body,
        headers={"Content-Type":
                 f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("text") or "").strip()


def transcribe(audio: bytes) -> str:
    """Turn sound into words.

    The sound comes as a wave file - the window makes it 16 kHz
    mono, because that is what the program wants.
    """
    if not audio:
        return ""
    if not available():
        raise RuntimeError("Die Spracheingabe ist in diesem Build nicht "
                           "enthalten")
    return _post(server(), audio)


def transcribe_base64(payload: str) -> str:
    """Turn sound from the window into words."""
    data = payload.split(",", 1)[-1] if payload.startswith("data:") else payload
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"die Aufnahme war nicht lesbar: {error}") from error
    return transcribe(raw)

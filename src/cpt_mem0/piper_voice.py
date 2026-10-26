"""The voice of Piper - a neural voice that runs on the processor.

**Why a second voice.** The voice of Windows is old - a synthesizer
from the year 2000, thin and robotic. Piper is a neural voice that
runs on the processor of the machine, without a graphics card, and
it is as fast as the speech itself.

**The model is not in the package.** It is 109 MB, and it is
downloaded when the user asks for the voice - the same way the
llama.cpp build is downloaded. The voice of Windows is the fallback:
it is always there, and it needs nothing.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

#: Where the voices are in the package. The build puts them there,
#: and the app finds them without asking for anything.
BUNDLE_DIR = "piper_voices"


def voice_dir() -> Path:
    """Where the voices live.

    **In the app first.** The voice is part of the program - the
    user does not download a voice, he installs an app that has
    one. Only when the app runs from the sources is the directory of
    the machine the place.
    """
    import sys

    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", ""))
        bundled = base / BUNDLE_DIR
        if bundled.is_dir():
            return bundled

    from .install import data_dir

    folder = data_dir() / "voices"
    if folder.is_dir():
        return folder

    # **From the sources.** The build puts the model here, and the
    # development version reads it from here.
    return (Path(__file__).resolve().parent.parent.parent
            / "packaging" / "voices")

#: The voice of the user. The name comes from the list of the
#: catalogue, and the file has the same name.
VOICES: dict[str, str] = {
    "Piper Thorsten": "de_DE-thorsten-high",
}

#: Where the files come from.
BASE_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/de/"
            "de_DE/thorsten/high")


def available() -> bool:
    """Whether the engine of Piper is there.

    **The package, not the model.** The model is downloaded when it
    is wanted; the engine has to be in the package.
    """
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    return True


def model_path(name: str) -> Path:
    """Where the file of a voice is."""
    key = VOICES.get(name, name)
    return voice_dir() / f"{key}.onnx"


def installed() -> list[dict[str, Any]]:
    """The voices whose files are on the disk.

    A voice that is not downloaded is not offered - the window
    offers what it can use.
    """
    result = []
    for name, key in VOICES.items():
        path = voice_dir() / f"{key}.onnx"
        result.append({
            "name": name,
            "language": "German",
            "description": f"{name} - German",
            "downloaded": path.is_file(),
            "path": str(path),
        })
    return result


def key_of(name: str) -> str:
    """The name of the file behind a voice."""
    return VOICES.get(name, name)


def download(name: str, progress=None) -> bool:
    """Get the file of a voice onto the disk.

    **The model is 109 MB.** The download happens in a thread of
    its own, and the window shows what is happening.
    """
    import urllib.request

    key = VOICES.get(name, key_of(name))
    folder = voice_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{key}.onnx"
    if target.is_file():
        return True
    url = f"{BASE_URL}/{key}.onnx"
    request = urllib.request.Request(
        url, headers={"User-Agent": "cpt-mem0"})
    with urllib.request.urlopen(request, timeout=600) as response:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        with open(target, "wb") as file:
            while True:
                chunk = response.read(262144)
                if not chunk:
                    break
                file.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    # The configuration belongs to the model.
    config = folder / f"{key}.onnx.json"
    if not config.is_file():
        request = urllib.request.Request(
            f"{BASE_URL}/{key}.onnx.json",
            headers={"User-Agent": "cpt-mem0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            config.write_bytes(response.read())
    return True


class PiperSpeaker:
    """Speaks with a neural voice.

    **The model is loaded once.** Loading takes a second, and every
    sentence is then a run through the network - fast enough to
    speak while the answer is still on the screen.
    """

    def __init__(self, name: str = "Piper Thorsten") -> None:
        self.name = name
        self._voice: Any = None
        self._lock = threading.Lock()
        self.last_error = ""

    def _load(self) -> Any:
        if self._voice is None:
            from piper import PiperVoice

            path = model_path(self.name)
            if not path.is_file():
                raise FileNotFoundError(
                    f"die Stimme ist nicht da: {path}")
            self._voice = PiperVoice.load(str(path))
        return self._voice

    def stop(self) -> None:
        """Stop the current utterance.

        The model is played at once, so there is nothing to stop -
        but the app asks, and the answer has to be there.
        """
        return None

    def speak(self, text: str) -> bool:
        """Say the text in the background."""
        text = (text or "").strip()
        if not text:
            return False
        if not available():
            self.last_error = "die Stimme von Piper ist nicht im Paket"
            return False

        def run() -> None:
            try:
                import wave

                with self._lock:
                    voice = self._load()
                    with wave.open(str(_temp_wav()), "wb") as target:
                        voice.synthesize_wav(text, target)
                _play(_temp_wav())
            except Exception as error:  # noqa: BLE001
                self.last_error = str(error)
                import logging

                logging.getLogger("cpt_mem0.voice").warning(
                    "the voice failed: %s", error)

        threading.Thread(target=run, daemon=True).start()
        return True


def _temp_wav() -> Path:
    import tempfile

    return Path(tempfile.gettempdir()) / "cpt-mem0-voice.wav"


def _play(path: Path) -> None:
    """Play the file - with the voice of Windows, or with the sound
    of the machine."""
    try:
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    except Exception:  # noqa: BLE001
        pass
    # Without winsound: the sound of the machine.
    try:
        from .proc import run

        run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
             f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
            capture_output=True, timeout=300)
    except Exception:  # noqa: BLE001
        pass

"""The voice of the assistant.

**One voice: the neural one.** The voice of Windows is a synthesizer
from the year 2000 - thin and robotic. The user heard it, and asked
for a better one. So the app has one voice now: a neural voice that
runs on the processor of the machine, without a graphics card, and
that is as fast as the speech itself.

**The model is not in the package.** It is 109 MB, and it is
downloaded when the user asks for the voice - the same way the
llama.cpp build is downloaded. Until it is there, the app says so
instead of speaking in the old voice.
"""

from __future__ import annotations

import threading
from typing import Any

#: The voice of the app. The model of it is downloaded on demand.
DEFAULT_VOICE = "Piper Thorsten"


def available() -> bool:
    """Whether the machine has a voice.

    **The engine, not the model.** The engine is in the package, the
    model is on the disk or it is not - and the voice list says
    which of the two it is.
    """
    from . import piper_voice

    return piper_voice.available()


def voices() -> list[dict[str, Any]]:
    """The voices that are on the disk.

    **Only the neural one.** The voices of Windows are not offered
    any more.
    """
    from . import piper_voice

    result: list[dict[str, Any]] = []
    for entry in piper_voice.installed():
        if entry.get("downloaded"):
            result.append(entry)
    return result


def voice_status() -> dict[str, Any]:
    """Whether there is a voice, and why not when there is none.

    **The reason belongs to the user.** A switch that changes
    nothing is worse than no switch: the user cannot tell whether
    the voice is off, whether the model is not downloaded yet, or
    whether the engine is missing from the build.
    """
    from . import piper_voice

    if not piper_voice.available():
        return {"available": False,
                "reason": "Die Sprachausgabe ist in diesem Build nicht "
                          "enthalten",
                "voices": []}

    installed = voices()
    if not installed:
        return {"available": False,
                "reason": "Die Stimme wird beim ersten Mal "
                          "heruntergeladen (109 MB)",
                "voices": []}

    return {"available": True, "reason": "", "voices": installed}


class Speaker:
    """Speaks the answers.

    The instance keeps the model loaded - loading it takes a second,
    and every sentence is then a run through the network.
    """

    def __init__(self, name: str = DEFAULT_VOICE) -> None:
        #: The voice the user picked.
        self.name = name or DEFAULT_VOICE
        #: **The last error.** The window shows it - a switch that
        #: switches nothing is worse than no switch.
        self.last_error = ""
        self._voice: Any = None
        self._lock = threading.Lock()

    def _piper(self):
        """The neural voice, when its model is on the disk."""
        from . import piper_voice

        if not piper_voice.available():
            return None
        if not piper_voice.model_path(self.name).is_file():
            return None
        if self._voice is None:
            self._voice = piper_voice.PiperSpeaker(self.name)
        return self._voice

    def speak(self, text: str) -> bool:
        """Say the text in the background.

        Returns whether it was handed over. A machine without the
        voice says so instead of failing the request.
        """
        text = (text or "").strip()
        if not text:
            return False

        piper = self._piper()
        if piper is None:
            self.last_error = (
                "die Stimme ist noch nicht da - sie wird beim ersten "
                "Mal heruntergeladen")
            return False

        def run() -> None:
            try:
                if not piper.speak(text):
                    self.last_error = piper.last_error
            except Exception as error:  # noqa: BLE001
                # **The error must not disappear.** A voice that
                # fails silently is a switch that does nothing.
                import logging

                logging.getLogger("cpt_mem0.voice").warning(
                    "the voice failed: %s", error)
                self.last_error = str(error)

        threading.Thread(target=run, daemon=True).start()
        return True

    def stop(self) -> None:
        """Stop the current utterance."""
        with self._lock:
            if self._voice is not None:
                try:
                    self._voice.stop()
                except Exception:  # noqa: BLE001
                    pass


def summarise(results: list[Any], limit: int = 5) -> str:
    """What is said out loud.

    The first few facts, each on its own line. A long list would
    take a minute to read and the user knows what they wrote.
    """
    lines = []
    for item in (results or [])[:limit]:
        if isinstance(item, str):
            text = item
        else:
            text = str(item.get("memory") or item.get("data")
                       or item.get("text") or "")
        text = text.strip()
        if text:
            lines.append(text)
    return "\n".join(lines)

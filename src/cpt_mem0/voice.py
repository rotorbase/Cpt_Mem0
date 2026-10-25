"""The voice output.

**It runs on the CPU and costs no VRAM.** The voice is the one
Windows already has - SAPI, the speech API that has been in Windows
since XP. There is no model to load, nothing to download, and nothing
that takes a bite out of the card.

The alternative would be a TTS model in llama.cpp. That would be a
few hundred megabytes of VRAM on a card that is already full - and
the Windows voice is good enough to read a list of facts aloud.

**The German voice is preferred.** The facts in the store are in
whatever language the user writes in. SAPI can speak English with a
German voice, but it is easier to hear than an English voice reading
German. When there is no German voice, the first one is used.

**The voice speaks in the background.** The answer must appear at
the same moment the voice starts, not when it ends. So the speaking
runs in a thread.
"""

from __future__ import annotations

import threading
from typing import Any


def available() -> bool:
    """Whether the machine has a voice.

    Windows only. On Linux and macOS there is no SAPI - and the app
    is not built for them yet.
    """
    import sys

    if sys.platform != "win32":
        return False
    try:
        import win32com.client  # noqa: F401
    except ImportError:
        return False
    return True


def voices() -> list[dict[str, Any]]:
    """The installed voices.

    Each voice has a name and a language. The list is empty when
    there is no voice.
    """
    if not available():
        return []

    import win32com.client

    result: list[dict[str, Any]] = []
    try:
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        for index in range(voice.GetVoices().Count):
            item = voice.GetVoices().Item(index)
            description = item.GetDescription()
            # "Microsoft Hedda Desktop - German"
            name, _, language = description.partition(" - ")
            result.append({
                "name": name.strip(),
                "language": language.strip(),
                "description": description,
            })
    except Exception:
        return []
    return result


def _pick_voice(voice: Any, preferred: str = "German") -> Any:
    """The voice with the preferred language, or the first one."""
    installed = []
    for index in range(voice.GetVoices().Count):
        item = voice.GetVoices().Item(index)
        description = item.GetDescription()
        installed.append((item, description))

    for item, description in installed:
        if preferred.lower() in description.lower():
            voice.Voice = item
            return voice

    if installed:
        voice.Voice = installed[0][0]
    return voice


class Speaker:
    """Speaks the answers.

    The instance keeps the SAPI object alive - creating it is the
    slow part, and the voice is used again and again.
    """

    def __init__(self, language: str = "German") -> None:
        self.language = language
        self._voice: Any = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _ensure_voice(self) -> Any:
        if self._voice is None:
            import win32com.client

            voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._voice = _pick_voice(voice, self.language)
        return self._voice

    def speak(self, text: str) -> bool:
        """Speak the text in the background.

        Returns whether it was handed over. A machine without a voice
        says so instead of failing the request.
        """
        text = (text or "").strip()
        if not text or not available():
            return False

        def run() -> None:
            try:
                with self._lock:
                    voice = self._ensure_voice()
                    voice.Speak(text)
            except Exception:
                pass

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the current utterance."""
        with self._lock:
            if self._voice is not None:
                try:
                    self._voice.Speak("", 2)  # 2 = PURGE
                except Exception:
                    pass


def summarise(results: list[Any], limit: int = 5) -> str:
    """What is said out loud.

    The first few facts, each on its own line. A long list would
    take a minute to read and the user knows what they wrote.
    """
    lines = []
    for item in results[:limit]:
        if isinstance(item, dict):
            text = item.get("memory") or item.get("text") or ""
        else:
            text = str(item)
        text = str(text).strip()
        if text:
            lines.append(text)
    if not lines:
        return ""
    if len(results) > limit:
        lines.append(f"Und {len(results) - limit} weitere.")
    return "\n".join(lines)

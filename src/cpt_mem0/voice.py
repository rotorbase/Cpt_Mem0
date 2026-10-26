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

import os
import threading
from typing import Any


def available() -> bool:
    """Whether the machine has a voice.

    Windows only. On Linux and macOS there is no SAPI - and the app
    is not built for them yet.
    """
    # **The engine, not the voices.** A machine with the speech API
    # but without a voice installed can still speak - it just has
    # nothing to speak with. The voices are a separate question, and
    # voice_status() answers it.
    import sys

    if sys.platform != "win32":
        return False
    if _com_available():
        return True
    # **The other way to the speech API.** Windows has had it built
    # in since Vista, and .NET speaks to it without pywin32. The
    # package can be missing from the build - then this is the way
    # the voice still works.
    return _powershell_available()


def _com_available() -> bool:
    """Whether pywin32 is there and can reach the speech API."""
    try:
        import win32com.client  # noqa: F401
    except ImportError:
        return False
    return True


#: The script that asks .NET for the installed voices.
_LIST_SCRIPT = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.GetInstalledVoices() | ForEach-Object { "
    "$_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture }"
)


def _powershell_available() -> bool:
    """Whether .NET can speak on this machine.

    **The check costs a process.** It runs once, when the window
    asks for the voices - and the answer is remembered.
    """
    global _POWERSHELL_OK
    if _POWERSHELL_OK is not None:
        return _POWERSHELL_OK
    try:
        from .proc import run

        result = run(["powershell", "-NoProfile", "-NonInteractive",
                      "-Command", _LIST_SCRIPT],
                     capture_output=True, text=True, timeout=25)
        _POWERSHELL_OK = result.returncode == 0
    except Exception:
        _POWERSHELL_OK = False
    return _POWERSHELL_OK


#: Whether .NET can speak. Remembered after the first check.
_POWERSHELL_OK: bool | None = None


def voice_status() -> dict[str, Any]:
    """Whether there is a voice, and why not when there is none.

    **The reason belongs to the user.** A switch that changes
    nothing is worse than no switch: the user cannot tell whether the
    voice is off, whether there is no voice on the machine, or
    whether the package is missing from the build.

    The three reasons are: the system has no SAPI, the package that
    talks to it is missing, or the machine has no voice installed.
    """
    import sys

    if sys.platform != "win32":
        return {"available": False, "reason": "SAPI gibt es nur unter Windows",
                "voices": []}

    if not _com_available() and not _powershell_available():
        return {"available": False,
                "reason": "Die Sprachausgabe von Windows ist auf diesem "
                          "Rechner nicht erreichbar",
                "voices": []}

    try:
        installed = voices()
    except Exception as error:
        return {"available": False,
                "reason": f"Die Stimmen konnten nicht gelesen werden: {error}",
                "voices": []}

    if not installed:
        return {"available": False,
                "reason": "Auf diesem Rechner ist keine Stimme installiert",
                "voices": []}

    return {"available": True, "reason": "", "voices": installed}


def voices() -> list[dict[str, Any]]:
    """The installed voices.

    Each voice has a name and a language. The list is empty when
    there is no voice.
    """
    # **Through available().** That call only asks whether the speech
    # engine is there - it does not ask for the voices, so there is
    # no circle.
    if not available():
        return []

    if _com_available():
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
            return result
        except Exception:
            pass

    # **The other way.** .NET knows the same voices, and it works
    # without pywin32.
    return _powershell_voices()


def _powershell_voices() -> list[dict[str, Any]]:
    """The installed voices, asked from .NET.

    **The same voices as SAPI.** "Microsoft Hedda Desktop" is there
    with the language "de-DE" - and the name is what the user picks
    in the window, so the two ways have to give the same names.
    """
    try:
        from .proc import run

        result = run(["powershell", "-NoProfile", "-NonInteractive",
                      "-Command", _LIST_SCRIPT],
                     capture_output=True, text=True, timeout=25)
    except Exception:
        return []
    if result.returncode != 0:
        return []

    voices: list[dict[str, Any]] = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, _, culture = line.partition("|")
        name = name.strip()
        culture = culture.strip()
        if not name:
            continue
        # "de-DE" is what .NET says, "German" is what the rest of
        # the app says.
        language = culture.split("-")[0].lower()
        language = {"de": "German", "en": "English"}.get(language, culture)
        voices.append({
            "name": name,
            "language": language,
            "description": f"{name} - {language}",
        })
    return voices


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

    def __init__(self, language: str = "German",
                 name: str = "") -> None:
        self.language = language
        #: **The voice the user chose.** Windows has a voice per
        #: language, and the German one is not always installed - so
        #: the choice is the user's.
        self.name = name
        #: **The last error.** The window shows it - a switch that
        #: switches nothing is worse than no switch.
        self.last_error = ""
        self._voice: Any = None
        #: Whether COM was initialised in this thread.
        self._initialised = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _ensure_voice(self) -> Any:
        """The voice object - through pywin32, or through .NET.

        **Two ways to the same speech API.** pywin32 is the fast
        one: the object stays alive and every sentence is a call on
        it. Without pywin32 the app talks to .NET instead, and every
        sentence costs a process - slower, but it speaks.

        **The apartment belongs to the thread.** COM has to be
        initialised in every thread that uses it - and the speech
        runs in a thread of its own, because a sentence takes
        seconds and the window may not freeze. Without
        ``CoInitialize`` the call fails with "CoInitialize wurde
        nicht aufgerufen", the error goes into the log, and the
        window stays silent.
        """
        if self._voice is None:
            if not _com_available():
                return None
            import win32com.client

            # The apartment of this thread - and only of this
            # thread. The window may call this from the thread that
            # started the app, and the speech from the thread that
            # speaks.
            try:
                import pythoncom

                pythoncom.CoInitialize()
                self._initialised = True
            except Exception:
                # Already initialised - that is not an error.
                self._initialised = False

            voice = win32com.client.Dispatch("SAPI.SpVoice")
            if self.name:
                # The voice the user picked, by its name.
                for index in range(voice.GetVoices().Count):
                    item = voice.GetVoices().Item(index)
                    if str(item.GetDescription()) == self.name:
                        voice.Voice = item
                        self._voice = voice
                        return self._voice
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
                    if voice is not None:
                        voice.Speak(text)
                    else:
                        # **The other way to the speech API.** No
                        # pywin32 in this build - .NET speaks instead.
                        _powershell_speak(text, self.name)
            except Exception as error:
                # **The error must not disappear.** A voice that
                # fails silently is a switch that does nothing.
                import logging

                logging.getLogger("cpt_mem0.voice").warning(
                    "the voice failed: %s", error)
                # **And the window has to hear about it.** The user
                # cannot look into a log file of a frozen app.
                self.last_error = str(error)

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


def _powershell_speak(text: str, name: str = "") -> bool:
    """Say the text through .NET.

    **The way out when pywin32 is missing.** The text goes to
    PowerShell in a file - a long sentence with quotes and umlauts
    does not belong on a command line - and .NET speaks it. The
    process starts without a console window: the app is not allowed
    to make a black box flash up.
    """
    import tempfile

    from .proc import run

    # **The script is a file.** The arguments of
    # "powershell -Command" are appended to the command and the
    # parser breaks - a path with a colon in it is the end of the
    # line. With "-File" the arguments are what they say they are.
    script = (
        "param([string]$TextPath, [string]$VoiceName)\n"
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "if ($VoiceName) { $s.SelectVoice($VoiceName) }\n"
        "$t = [System.IO.File]::ReadAllText($TextPath, "
        "[System.Text.Encoding]::UTF8)\n"
        "$s.Speak($t)\n"
    )
    path = None
    script_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(text)
            path = handle.name
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False,
                                         encoding="utf-8") as script_file:
            script_file.write(script)
            script_path = script_file.name
        result = run(["powershell", "-NoProfile", "-NonInteractive",
                      "-ExecutionPolicy", "Bypass", "-File", script_path,
                      "-TextPath", path, "-VoiceName", name or ""],
                     capture_output=True, text=True, timeout=180)
        return result.returncode == 0
    except Exception as error:
        import logging

        logging.getLogger("cpt_mem0.voice").warning(
            "the voice failed: %s", error)
        return False
    finally:
        for temporary in (path, script_path):
            if not temporary:
                continue
            try:
                os.unlink(temporary)
            except OSError:
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

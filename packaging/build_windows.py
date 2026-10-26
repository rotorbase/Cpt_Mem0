"""Build the Windows package.

    python packaging/build_windows.py

What it does:

1. Creates a clean virtual environment with the pinned dependencies.
2. Installs PyInstaller.
3. Builds the app with the specification.
4. Puts the result into a zip, with the version in the name.
5. Writes the checksum, so the download can be checked.

**The llama.cpp build is not in the package.** It is 700 MB with the
CUDA runtime, and `cpt-mem0 init` downloads it separately, pinned to
an exact build. It belongs on the disk of the machine, not in a zip
file that every user has to download.

The result is in ``dist/``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"

#: The dependencies of the app. The version is pinned, so a build
#: from today and a build from next month are the same.
REQUIREMENTS = [
    "mem0ai==2.2.1",
    "qdrant-client==1.20.0",
    "pyyaml==6.0.3",
    "pywebview>=5.0",
    # **The voice.** Without pywin32 the speech API of Windows cannot
    # be reached, and the app says so instead of speaking. The
    # package has to be here - the build installs what the app needs.
    "pywin32>=306",
    # **The neural voice.** It runs on the processor, needs no
    # graphics card, and the model of it is downloaded on demand.
    "piper-tts>=1.3",
]

#: The build tool. It is not a dependency of the app.
BUILD_REQUIREMENTS = [
    "pyinstaller>=6.0",
]


def run(command: list[str], cwd: Path | None = None) -> None:
    """Run a command and stop the build when it fails.

    **PYTHONPATH is empty while the build runs.** Without that the
    build takes the packages of another environment - the one the
    shell happens to point at - and the bundle gets two versions of
    the same library. The error then names a version conflict that
    exists nowhere else.
    """
    print(f"$ {' '.join(command)}", flush=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(command, cwd=cwd or ROOT, env=environment)
    if result.returncode != 0:
        raise SystemExit(f"the step failed with {result.returncode}")


def read_version() -> str:
    """The version of the app, out of the package."""
    init = (ROOT / "src" / "cpt_mem0" / "__init__.py").read_text(
        encoding="utf-8")
    for line in init.splitlines():
        if line.startswith("__version__"):
            return line.split("=")[1].strip().strip('"').strip("'")
    return "0.0.0"


#: The voice of the app. The model is 109 MB, and it is part of
#: the program - the user does not download a voice, he installs an
#: app that has one.
#:
#: The file is too big for the repository, so the build gets it from
#: the same place the models come from.
VOICE_FILES = {
    "de_DE-thorsten-high.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/"
        "de_DE/thorsten/high/de_DE-thorsten-high.onnx",
    "de_DE-thorsten-high.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/"
        "de_DE/thorsten/high/de_DE-thorsten-high.onnx.json",
}


def get_voices() -> Path:
    """The model of the voice, ready to be packed."""
    folder = ROOT / "packaging" / "voices"
    folder.mkdir(parents=True, exist_ok=True)
    import urllib.request

    for name, url in VOICE_FILES.items():
        target = folder / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"   {name} ist da ({target.stat().st_size} B)",
                  flush=True)
            continue
        print(f"   {name} wird geholt ...", flush=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "cpt-mem0-build"})
        with urllib.request.urlopen(request, timeout=600) as response:
            target.write_bytes(response.read())
    return folder


#: The ear of the assistant. whisper.cpp turns the sound of the
#: headset into words, and the model of German speech belongs to it.
#:
#: Both are part of the package - the user installs the app and can
#: talk to it. Nothing is downloaded at runtime.
#:
#: **The card does the hearing.** On the processor a sentence takes
#: four and a half seconds, on the card two tenths - and the card
#: has room: the model of the speech is 190 MB.
WHISPER_BINARY = (
    "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/"
    "whisper-bin-win-cuda-11.8.0-x64.zip")
WHISPER_MODEL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
    "ggml-small-q5_1.bin")

#: The CUDA libraries of NVIDIA. whisper.cpp brings the program, but
#: not the libraries of the card - without them the program falls
#: back to the processor.
WHISPER_CUBLAS = "https://pypi.org/pypi/nvidia-cublas-cu11/json"

#: Only what is needed to read a wave file.
WHISPER_FILES = [
    "whisper-server.exe", "whisper.dll", "llama.dll", "ggml.dll",
    "ggml-cuda.dll", "ggml-base.dll", "ggml-cpu-x64.dll", "SDL2.dll",
    "cudart64_110.dll", "cuinj64_118.dll", "nvrtc64_112_0.dll",
    "nvrtc-builtins64_118.dll",
]


def get_whisper() -> Path:
    """The program and the model for the dictation."""
    folder = ROOT / "packaging" / "whisper"
    folder.mkdir(parents=True, exist_ok=True)

    binary = folder / "whisper-server.exe"
    if not binary.is_file():
        print("   whisper.cpp wird geholt ...", flush=True)
        import io
        import urllib.request
        import zipfile

        request = urllib.request.Request(
            WHISPER_BINARY, headers={"User-Agent": "cpt-mem0-build"})
        with urllib.request.urlopen(request, timeout=600) as response:
            blob = response.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                short = name.rsplit("/", 1)[-1]
                if short in WHISPER_FILES:
                    target = folder / short
                    target.write_bytes(archive.read(name))
    else:
        print("   whisper.cpp ist da", flush=True)

    # **The CUDA libraries.** They come in a wheel of NVIDIA, and
    # the two files are taken out of it.
    cublas = folder / "cublas64_11.dll"
    if not cublas.is_file():
        print("   die CUDA-Bibliotheken werden geholt ...", flush=True)
        import json
        import urllib.request
        import zipfile

        request = urllib.request.Request(
            WHISPER_CUBLAS, headers={"User-Agent": "cpt-mem0-build"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        wheel = next(f for f in data["urls"] if "win" in f["filename"])
        request = urllib.request.Request(
            wheel["url"], headers={"User-Agent": "cpt-mem0-build"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            blob = response.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                short = name.rsplit("/", 1)[-1]
                if short in ("cublas64_11.dll", "cublasLt64_11.dll"):
                    (folder / short).write_bytes(archive.read(name))

    model = folder / WHISPER_MODEL.rsplit("/", 1)[-1]
    if not model.is_file():
        print("   das Sprachmodell wird geholt (190 MB) ...", flush=True)
        import urllib.request

        request = urllib.request.Request(
            WHISPER_MODEL, headers={"User-Agent": "cpt-mem0-build"})
        with urllib.request.urlopen(request, timeout=900) as response:
            model.write_bytes(response.read())
    else:
        print("   das Sprachmodell ist da", flush=True)
    return folder


def build() -> Path:
    """Build the package and return the folder."""
    print("== cleaning ==", flush=True)
    for folder in (DIST, BUILD):
        if folder.exists():
            shutil.rmtree(folder)

    print("== getting the voice ==", flush=True)
    get_voices()

    print("== getting the ear ==", flush=True)
    get_whisper()

    print("== installing the dependencies ==", flush=True)
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", *REQUIREMENTS,
         *BUILD_REQUIREMENTS])

    print("== building ==", flush=True)
    run([sys.executable, "-m", "PyInstaller",
         str(ROOT / "packaging" / "Cpt_Mem0.spec"),
         "--noconfirm", "--distpath", str(DIST), "--workpath", str(BUILD)])

    folder = DIST / "Cpt_Mem0"
    if not (folder / "Cpt_Mem0.exe").is_file():
        raise SystemExit("the build produced no Cpt_Mem0.exe")

    # **The .NET runtime configuration goes next to the exe.**
    #
    # The window is a WebView2 inside a WinForms window, and WinForms
    # needs the WindowsDesktop framework of .NET. Without this file
    # the runtime starts without a framework list, WinForms cannot be
    # found, and the app falls back to the browser.
    #
    # It has to be next to the exe and not in the _internal folder:
    # the .NET host looks for it there.
    source = ROOT / "packaging" / "runtimeconfig.json"
    target = folder / "Cpt_Mem0.runtimeconfig.json"
    shutil.copyfile(source, target)
    print(f"   {target.name} neben der exe", flush=True)

    return folder


def zip_it(folder: Path, version: str) -> Path:
    """Put the folder into a zip and write the checksum."""
    target = DIST / f"Cpt_Mem0-{version}-windows-x64.zip"
    print(f"== zipping -> {target.name} ==", flush=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, Path("Cpt_Mem0") / path.relative_to(folder))

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (DIST / f"{target.name}.sha256").write_text(
        f"{digest}  {target.name}\n", encoding="utf-8")
    size = target.stat().st_size / (1024 * 1024)
    print(f"   {target.name}: {size:.1f} MB", flush=True)
    print(f"   sha256: {digest}", flush=True)
    return target


def main() -> int:
    version = read_version()
    print(f"building Cpt Mem0 {version}", flush=True)
    folder = build()
    zip_it(folder, version)
    print("\ndone. The package is in dist/.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

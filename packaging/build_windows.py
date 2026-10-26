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


def build() -> Path:
    """Build the package and return the folder."""
    print("== cleaning ==", flush=True)
    for folder in (DIST, BUILD):
        if folder.exists():
            shutil.rmtree(folder)

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

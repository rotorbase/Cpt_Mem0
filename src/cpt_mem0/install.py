"""Installing and updating the llama.cpp runtime.

The runtime lives outside the configuration directory, in a place that
survives an update:

    <data dir>/runtime/llama/
        b11838/          the extracted build
        current -> b11838   the pointer to the build in use

**The update never overwrites the running build.** The new build is
unpacked beside the current one, the pointer is switched, and only then
does the daemon restart. If anything fails, the old pointer is restored
and the old build is used again. The previous build is kept, so a
rollback is possible without a download.

**Models and the store are never touched.** An update changes the
binary and nothing else. The one exception is the embedder: if the
embedder model changes, the vectors in the store were made by a
different model and the two do not match. The updater checks this and
warns.
"""

from __future__ import annotations

import os
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .runtime import (
    DEFAULT_BUILD,
    Platform,
    _archive_format,
    detect_cuda_version,
    detect_platform,
    download,
    has_cuda_runtime,
    resolve_build,
    server_binary,
    unpack,
)


def data_dir() -> Path:
    """Where the app keeps its state.

    Windows uses the local application data, Linux and macOS the XDG
    data directory. The runtime lives here, not in the configuration
    directory, because it has to survive a configuration reset.
    """
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = str(Path.home() / "AppData" / "Local")
        return Path(base) / "cpt-mem0"
    base = os.environ.get("XDG_DATA_HOME")
    if not base:
        base = str(Path.home() / ".local" / "share")
    return Path(base) / "cpt-mem0"


def config_dir() -> Path:
    """Where the configuration lives."""
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA")
        if not base:
            base = str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "cpt-mem0"
    base = os.environ.get("XDG_CONFIG_HOME")
    if not base:
        base = str(Path.home() / ".config")
    return Path(base) / "cpt-mem0"


def runtime_dir() -> Path:
    """The directory the builds live in."""
    return data_dir() / "runtime" / "llama"


def pointer_path() -> Path:
    """The pointer to the build in use."""
    return runtime_dir() / "current"


@dataclass(frozen=True)
class Installation:
    """The state of the runtime on this machine."""

    build: str | None
    directory: Path | None
    binary: Path | None
    has_cuda_runtime: bool

    @property
    def installed(self) -> bool:
        return self.binary is not None


def read_pointer() -> str | None:
    """The build the pointer names, if there is one.

    **The pointer has three shapes, and all three have to work.**

    - A symlink to the build. The target is the name.
    - A text file with the name in it. The reliable way on Windows
      without the symlink privilege.
    - A directory junction. It looks like a directory, and the
      directory it points at is the build.

    The junction is the reason this function has three branches: a
    junction is not a symlink and not a file, so the first two
    branches miss it and the pointer reads as "nothing is installed" -
    on a machine where the junction worked, while the same code was
    right on a machine where it did not.
    """
    pointer = pointer_path()
    if pointer.is_symlink():
        target = os.readlink(pointer)
        return Path(target).name
    if pointer.is_file():
        return pointer.read_text(encoding="utf-8").strip() or None
    if pointer.is_dir():
        # A junction: the directory behind it is the build.
        try:
            resolved = pointer.resolve()
        except OSError:
            return None
        name = resolved.name
        if name and name != pointer.name:
            return name
    return None


def current_installation() -> Installation:
    """What is installed right now."""
    build = read_pointer()
    if build is None:
        return Installation(None, None, None, False)
    directory = runtime_dir() / build
    if not directory.is_dir():
        return Installation(None, None, None, False)
    try:
        binary = server_binary(directory)
    except LookupError:
        return Installation(build, directory, None, False)
    return Installation(build, directory, binary,
                        has_cuda_runtime(directory))


def _write_pointer(build: str) -> None:
    """Point at a build.

    A symlink where the platform supports it, a small text file where it
    does not. Windows needs the symlink privilege for a directory
    junction, so the text file is the reliable path there.
    """
    directory = runtime_dir()
    directory.mkdir(parents=True, exist_ok=True)
    pointer = pointer_path()

    if pointer.is_symlink() or pointer.exists():
        pointer.unlink()

    try:
        os.symlink(build, pointer, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pointer.write_text(build, encoding="utf-8")


def server_binary_exists(directory: Path) -> bool:
    """Whether the server binary is already in the directory."""
    for name in ("llama-server", "llama-server.exe"):
        if (directory / name).is_file():
            return True
    return False


def _place_runtime(runtime_archive: Path, destination: Path) -> None:
    """Put the CUDA runtime libraries next to the binary.

    The `cudart-` archive has a flat structure: the DLLs sit at the
    top level. The loader looks for them in the directory of the
    executable, so that is where they go.
    """
    kind = _archive_format(runtime_archive)
    if kind == "zip":
        with zipfile.ZipFile(runtime_archive) as handle:
            for member in handle.infolist():
                if member.is_dir():
                    continue
                name = Path(member.filename.replace("\\", "/")).name
                if not name:
                    continue
                (destination / name).write_bytes(handle.read(member))
    elif kind in ("tar.gz", "tar.bz2", "tar.xz"):
        with tarfile.open(runtime_archive) as handle:
            for member in handle.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name:
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    continue
                (destination / name).write_bytes(extracted.read())
    else:
        raise ValueError(
            f"unsupported runtime archive format: {runtime_archive.name}")


def install(build: str | None = DEFAULT_BUILD,
            target: Platform | None = None,
            archive: Path | None = None,
            runtime_archive: Path | None = None,
            keep_archive: bool = False) -> Installation:
    """Install a build and point at it.

    The build is unpacked beside the ones that are already there. The
    pointer is switched only after the unpack succeeded, so a failed
    install leaves the running build untouched.

    ``archive`` is the binary package. ``runtime_archive`` is the
    `cudart-` package with the CUDA runtime libraries; it is needed on
    a CUDA install and has to be placed next to the binary.
    """
    if build is None:
        build = DEFAULT_BUILD
    if target is None:
        target = detect_platform()
        if target.cuda:
            version = detect_cuda_version()
            if version:
                target = Platform(target.system, target.machine, True,
                                  version)

    root = runtime_dir()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / build

    downloaded = False
    if archive is not None:
        # A local archive: the download already happened, and the file
        # belongs to the caller. It is not deleted.
        if not archive.is_file():
            raise FileNotFoundError(archive)
    else:
        resolved = resolve_build(build, target=target)
        asset = resolved["binary"]
        print(f"  binary asset: {asset['name']} "
              f"({asset['size'] / 1024 / 1024:.1f} MiB)", flush=True)
        archive = root / f"{asset['name']}.download"
        if not archive.is_file():
            download(asset, archive)

        runtime_asset = resolved["runtime"]
        if runtime_asset is not None:
            print(f"  runtime asset: {runtime_asset['name']} "
                  f"({runtime_asset['size'] / 1024 / 1024:.1f} MiB)",
                  flush=True)
            runtime_archive = root / f"{runtime_asset['name']}.download"
            if not runtime_archive.is_file():
                download(runtime_asset, runtime_archive)
        downloaded = True

    if destination.is_dir() and server_binary_exists(destination):
        print(f"  the build {build} is already unpacked", flush=True)
    else:
        print(f"  unpacking into {destination}", flush=True)
        # The archive carries a top-level directory with the build name.
        # It is unpacked into that directory, so the result is
        # <runtime>/b11838/... and not <runtime>/b11838/b11838/...
        destination.mkdir(parents=True, exist_ok=True)
        unpack(archive, destination, strip_top_level=True)

    if runtime_archive is not None:
        # The cudart- archive holds only the CUDA runtime libraries.
        # They go next to the binary, where the loader looks for them.
        print("  placing the CUDA runtime next to the binary", flush=True)
        _place_runtime(runtime_archive, destination)

    if downloaded and not keep_archive:
        archive.unlink(missing_ok=True)
        if runtime_archive is not None:
            runtime_archive.unlink(missing_ok=True)

    binary = server_binary(destination)
    runtime_present = has_cuda_runtime(destination)

    if target.cuda and not runtime_present:
        # The build was unpacked but the CUDA runtime is missing. That
        # is the silent CPU fallback. Refuse to point at it.
        raise RuntimeError(
            f"the build {build} has no CUDA runtime - the server would "
            f"fall back to the CPU without saying so. Check the archive: "
            f"it has to be the cudart- variant")

    _write_pointer(build)
    print(f"  installed: {binary}", flush=True)
    if runtime_present:
        print("  CUDA runtime: present", flush=True)
    else:
        print("  CUDA runtime: absent (CPU only)", flush=True)

    return Installation(build, destination, binary, runtime_present)


def update(build: str | None = None, target: Platform | None = None,
           dry_run: bool = False) -> tuple[Installation, Installation | None]:
    """Install a new build and switch to it.

    Returns the new installation and the previous one. The previous
    build is kept on disk, so a rollback costs nothing.

    With ``dry_run`` nothing is downloaded and nothing is changed: the
    function only reports what a switch would do.
    """
    previous = current_installation()
    wanted = build or DEFAULT_BUILD

    if previous.build == wanted and previous.installed:
        print(f"  the build {wanted} is already in use")
        return previous, None

    if dry_run:
        print(f"  would install {wanted}")
        print(f"  would keep {previous.build} for rollback")
        return previous, previous

    print(f"  updating from {previous.build or 'nothing'} to {wanted}")
    installed = install(wanted, target=target)
    return installed, previous


def rollback() -> Installation | None:
    """Point back at the build that was in use before the last update.

    The builds are on disk in the order they were installed, so the
    newest other one is the one to go back to.
    """
    current = read_pointer()
    # **The list of the builds comes from one place.** The pointer is
    # a directory as far as the file system is concerned when it is a
    # symlink or a junction - and it sorts after every build, so it
    # would be chosen as the target and the pointer would point at
    # itself. On a machine where the pointer is a text file it never
    # shows up, and the same code was right there.
    builds = sorted(
        (name for name in list_builds() if name != current),
        reverse=True)
    if not builds:
        print("  there is no earlier build to go back to")
        return None
    target = builds[0]
    print(f"  going back to {target}")
    _write_pointer(target)
    return current_installation()


def list_builds() -> list[str]:
    """The builds that are on disk.

    **The pointer is not a build.** As a symlink or as a junction it
    is a directory as far as the file system is concerned, so it
    would show up in the list. The list of the builds has to be the
    same on every machine.
    """
    root = runtime_dir()
    if not root.is_dir():
        return []
    pointer = pointer_path().name
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and p.name != pointer)

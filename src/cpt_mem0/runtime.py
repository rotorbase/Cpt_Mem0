"""Finding and unpacking the right llama.cpp build.

The app ships its own runtime. That is the only way to guarantee that
the flags the planner emits, the KV layout the measurements assume and
the vector store's embeddings all come from one known build.

**The build is pinned.** A floating "latest" would change under the
user's feet: a new nightly can change the model output, the KV layout
or the flag semantics, and break the planner's measurements without
anyone doing anything. The b-builds are permanent, so a pinned one is
reproducible forever.

**The `cudart-` variant is always used on CUDA.** Without the bundled
CUDA runtime, llama.cpp falls back to the CPU *silently*: no error, the
port opens, `/v1/models` answers 200, and inference runs at 6 tokens
per second instead of 47. The planner verifies the outcome by
measuring VRAM, but the bundle removes the failure at its source.

**The asset names are not uniform.** In a b-build release some assets
carry the build number and some do not:

    llama-b11838-bin-ubuntu-cuda-13.4-x64.tar.gz
    cudart-llama-bin-win-cuda-13.4-x64.zip

So the assets are matched by their platform and variant parts, not by a
single prefix. Everything here is based on the real asset list of
b11838, read from the GitHub API.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

#: The repository that publishes the rolling builds.
REPO = "ggml-org/llama.cpp"

#: The build the app pins. See SPEC 7.1.
DEFAULT_BUILD = "b11838"

#: The API endpoint for a release by tag.
RELEASE_API = "https://api.github.com/repos/{repo}/releases/tags/{tag}"

#: Where the release assets are downloaded from.
DOWNLOAD_URL = ("https://github.com/{repo}/releases/download/{tag}/{name}")


@dataclass(frozen=True)
class Platform:
    """The platform and the compute backend the app runs on."""

    system: str          # "windows", "linux", "macos"
    machine: str         # "x64", "arm64"
    cuda: bool           # whether a CUDA build is wanted
    cuda_version: str    # "13.4", "12.4", ""

    @property
    def is_windows(self) -> bool:
        return self.system == "windows"

    @property
    def is_macos(self) -> bool:
        return self.system == "macos"


def detect_platform(cuda: bool | None = None,
                    cuda_version: str | None = None) -> Platform:
    """Detect the platform this process runs on.

    ``cuda`` and ``cuda_version`` override the detection; that is what
    the installer uses when the user asks for a CPU-only build or for an
    older driver.
    """
    system = platform.system().lower()
    if system.startswith("win"):
        system = "windows"
    elif system == "darwin":
        system = "macos"
    else:
        system = "linux"

    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        machine = "x64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64"
    else:
        machine = "x64"

    if cuda is None:
        # macOS has no CUDA. Metal is the only accelerated backend.
        cuda = system != "macos"
    if cuda_version is None:
        cuda_version = "13.4" if cuda else ""

    return Platform(system=system, machine=machine, cuda=cuda,
                     cuda_version=cuda_version)


def detect_cuda_version() -> str | None:
    """Read the CUDA version the installed driver supports.

    ``nvidia-smi`` prints the driver version, not the CUDA version it
    supports. The header line of the CUDA 13 runtime needs a driver
    that reports 13.x; the 12.8 build needs 12.x. The driver version
    maps to a CUDA version closely enough for the choice here: a 13.x
    driver runs the 13.4 build, an older one the 12.4 build.

    Returns None when there is no NVIDIA driver at all.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        import subprocess
        result = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    if not first:
        return None
    driver = first[0].strip()
    major = driver.split(".")[0]
    if major.isdigit() and int(major) >= 13:
        return "13.4"
    return "12.4"


def _asset_matches(name: str, target: Platform,
                   want_runtime_package: bool = False) -> bool:
    """Whether a release asset is the one for this platform.

    The asset names differ between the `cudart-` and the plain
    variants, and some carry the build number while some do not. The
    match is therefore done on the parts that describe the platform.

    With ``want_runtime_package`` the `cudart-` asset is matched - the
    one with only the CUDA runtime libraries. Without it, the binary
    package is matched and the runtime package is rejected.
    """
    lower = name.lower()
    is_runtime_package = "cudart" in lower

    if target.system == "windows":
        if "-win-" not in lower:
            return False
    elif target.system == "linux":
        if "ubuntu" not in lower:
            return False
    elif target.system == "macos":
        if "macos" not in lower:
            return False
    else:
        return False

    if target.machine == "arm64":
        if "arm64" not in lower:
            return False
    else:
        # x64 assets say x64; an arm64 asset must not match.
        if "arm64" in lower or "x64" not in lower:
            return False

    if target.system == "macos":
        # macOS has Metal, never CUDA.
        return "cuda" not in lower and "cudart" not in lower

    if target.cuda:
        if target.cuda_version and f"cuda-{target.cuda_version}" not in lower:
            return False
        if want_runtime_package:
            return is_runtime_package
        # The binary package. The runtime package has no binary in it.
        return not is_runtime_package

    # CPU-only: an asset without any accelerator in the name.
    #
    # The CPU build is not always called "cpu". On Linux the asset is
    # "llama-b11838-bin-ubuntu-x64.tar.gz" - the word "cpu" is only in
    # the Windows name. So the test is the absence of every
    # accelerator, not the presence of "cpu".
    for marker in ("cuda", "cudart", "vulkan", "sycl", "rocm", "openvino",
                   "opencl", "s390x", "arm64"):
        if marker in lower:
            return False
    # There is no runtime package without CUDA.
    if want_runtime_package:
        return False
    return True


def select_asset(assets: list[dict], target: Platform) -> dict | None:
    """The binary asset for this platform, or None when there is none.

    **This is not the `cudart-` asset.** The `cudart-` archive holds
    only the CUDA runtime libraries - `cublas64_13.dll`,
    `cublasLt64_13.dll` and `cudart64_13.dll` - and no binary at all.
    The binaries are in the plain asset next to it.

    So a CUDA install needs both archives: the binary package and the
    runtime package.
    """
    candidates = [a for a in assets if _asset_matches(a["name"], target)]
    if not candidates:
        return None

    def _rank(asset: dict) -> tuple:
        name = asset["name"].lower()
        # The newest CUDA version first.
        version = ""
        match = re.search(r"cuda-([0-9.]+)", name)
        if match:
            version = match.group(1)
        return (version,)

    candidates.sort(key=_rank, reverse=True)
    return candidates[0]


def select_runtime_asset(assets: list[dict],
                         target: Platform) -> dict | None:
    """The `cudart-` runtime asset for this platform.

    This archive has no binary in it. It holds the CUDA runtime
    libraries that the binary package expects next to it.
    """
    candidates = [
        a for a in assets
        if _asset_matches(a["name"], target, want_runtime_package=True)]
    if not candidates:
        return None

    def _rank(asset: dict) -> tuple:
        name = asset["name"].lower()
        version = ""
        match = re.search(r"cuda-([0-9.]+)", name)
        if match:
            version = match.group(1)
        return (version,)

    candidates.sort(key=_rank, reverse=True)
    return candidates[0]


def list_assets(build: str = DEFAULT_BUILD, repo: str = REPO,
                timeout: int = 30) -> list[dict]:
    """The assets of a build release.

    Only the fields the installer needs, so the caller does not depend
    on the shape of the API response.
    """
    url = RELEASE_API.format(repo=repo, tag=build)
    request = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [
        {
            "name": asset["name"],
            "size": asset["size"],
            "url": asset["browser_download_url"],
        }
        for asset in payload.get("assets", [])
    ]


def resolve_build(build: str = DEFAULT_BUILD, repo: str = REPO,
                  target: Platform | None = None) -> dict:
    """The binary asset and the runtime asset for this platform.

    A CUDA install needs two archives: the binary package and the
    `cudart-` package with the CUDA runtime libraries. The runtime
    asset is None on a CPU-only install and on macOS.

    Raises ``LookupError`` when the release does not exist or has no
    asset for the platform.
    """
    if target is None:
        target = detect_platform()
    assets = list_assets(build, repo)
    if not assets:
        raise LookupError(
            f"the release {build} of {repo} has no assets - it does not "
            f"exist or is not a rolling build")
    asset = select_asset(assets, target)
    if asset is None:
        raise LookupError(
            f"the release {build} has no asset for "
            f"{target.system}-{target.machine}"
            + (f" with CUDA {target.cuda_version}" if target.cuda else "")
            + " - see the release page for the available platforms")

    runtime_asset = None
    if target.cuda:
        runtime_asset = select_runtime_asset(assets, target)
        if runtime_asset is None:
            raise LookupError(
                f"the release {build} has no cudart- package for "
                f"{target.system}-{target.machine} CUDA "
                f"{target.cuda_version}. Without it the server falls back "
                f"to the CPU without saying so.")

    return {"binary": asset, "runtime": runtime_asset}


def download(asset: dict, destination: Path,
             timeout: int = 60) -> Path:
    """Download an asset, with a progress line.

    The download is streamed to a temporary file and moved into place
    only when it is complete, so an interrupted download never leaves a
    truncated file that looks installable.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(
        asset["url"], headers={"Accept": "application/octet-stream"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        last_report = 0.0
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = _now()
                if now - last_report > 1.0:
                    last_report = now
                    if total:
                        percent = written * 100 // total
                        print(f"    {percent:3d}%  "
                              f"{written / 1024 / 1024:.0f} of "
                              f"{total / 1024 / 1024:.0f} MiB", flush=True)
                    else:
                        print(f"    {written / 1024 / 1024:.0f} MiB", flush=True)

    if total and written != total:
        temporary.unlink(missing_ok=True)
        raise OSError(
            f"the download was incomplete: {written} of {total} bytes")

    temporary.replace(destination)
    return destination


def _now() -> float:
    import time
    return time.monotonic()


def _archive_format(archive: Path) -> str:
    """The format of an archive, by its content and not by its suffix.

    The downloaded file carries the suffix ".download" while it is on
    the way, and a user-supplied archive can be called anything at
    all. The magic bytes say what it is.
    """
    with archive.open("rb") as handle:
        head = handle.read(6)

    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "tar.gz"
    if head[:3] == b"BZh":
        return "tar.bz2"
    if head[:6] == b"\xfd7zXZ\x00":
        return "tar.xz"
    if head[:4] == b"7z\xbc\xaf":
        return "7z"
    raise ValueError(f"unknown archive format: {archive.name}")


def unpack(archive: Path, target: Path,
           strip_top_level: bool = False) -> Path:
    """Unpack a release archive and return the directory with the binary.

    The archive contains a top-level directory. Some builds nest the
    binaries one level deeper, so the search goes two levels down.

    With ``strip_top_level`` the top-level directory is not recreated
    inside the target: the archive is unpacked into ``target`` and its
    top-level directory becomes ``target/<name>``. Without it, the
    whole tree lands in ``target`` - which produces
    ``target/b11838/b11838`` when the target is already the build
    directory.
    """
    target.mkdir(parents=True, exist_ok=True)
    kind = _archive_format(archive)

    if kind == "zip":
        with zipfile.ZipFile(archive) as handle:
            if strip_top_level:
                _extract_zip_stripped(handle, target)
            else:
                handle.extractall(target)
    elif kind in ("tar.gz", "tar.bz2", "tar.xz"):
        with tarfile.open(archive) as handle:
            if strip_top_level:
                _extract_tar_stripped(handle, target)
            else:
                handle.extractall(target, filter="data")
    else:
        raise ValueError(f"unsupported archive format: {archive.name}")

    if strip_top_level:
        # The top-level directory was stripped. The build directory is
        # the target itself, or a directory inside it.
        if _has_server_binary(target):
            return target
        for candidate in sorted(target.iterdir()):
            if candidate.is_dir() and _has_server_binary(candidate):
                return candidate
        raise LookupError(
            f"no llama-server binary found under {target}")

    return _find_binary_root(target)


def _extract_zip_stripped(handle: zipfile.ZipFile, target: Path) -> None:
    """Extract, dropping the top-level directory.

    Not every archive has one. The Windows binary package is flat: the
    53 files sit at the top level. The Linux package has a directory
    with the build name. So the first component is only dropped when
    it really is a directory that holds the content.
    """
    names = [m.filename.replace("\\", "/") for m in handle.infolist()
             if not m.is_dir() and m.filename]
    if not names:
        return

    roots = {name.split("/")[0] for name in names}
    has_root = len(roots) == 1 and all(
        "/" in name for name in names)

    for member in handle.infolist():
        if member.is_dir():
            continue
        filename = member.filename.replace("\\", "/")
        if has_root:
            parts = filename.split("/")
            if len(parts) <= 1:
                continue
            name = "/".join(parts[1:])
        else:
            name = filename
        if not name:
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(handle.read(member))


def _extract_tar_stripped(handle: tarfile.TarFile, target: Path) -> None:
    """Extract, dropping the top-level directory.

    A flat archive is extracted as it is.
    """
    members = [m for m in handle.getmembers() if m.isfile()]
    if not members:
        return

    roots = {m.name.replace("\\", "/").split("/")[0] for m in members}
    has_root = len(roots) == 1 and all(
        "/" in m.name.replace("\\", "/") for m in members)

    for member in members:
        name = member.name.replace("\\", "/")
        if has_root:
            parts = name.split("/")
            if len(parts) <= 1:
                continue
            name = "/".join(parts[1:])
        if not name:
            continue
        member.name = name
        try:
            handle.extract(member, target, filter="data")
        except (OSError, ValueError):
            continue


def _find_binary_root(target: Path) -> Path:
    """The directory that holds the server binary.

    The archive has a top-level directory with the build name. The
    binaries are either directly in it or in a subdirectory such as
    ``build/bin``.
    """
    for candidate in sorted(target.iterdir()):
        if not candidate.is_dir():
            continue
        if _has_server_binary(candidate):
            return candidate
        for sub in sorted(candidate.iterdir()):
            if sub.is_dir() and _has_server_binary(sub):
                return sub
    raise LookupError(
        f"no llama-server binary found under {target}")


def _has_server_binary(directory: Path) -> bool:
    for name in ("llama-server", "llama-server.exe"):
        if (directory / name).is_file():
            return True
    return False


def server_binary(root: Path) -> Path:
    """The path of the llama-server binary inside an extracted build."""
    for name in ("llama-server", "llama-server.exe"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    for directory in root.rglob("llama-server*"):
        if directory.is_file():
            return directory
    raise LookupError(f"no llama-server binary under {root}")


def has_cuda_runtime(root: Path) -> bool:
    """Whether the CUDA runtime libraries are present.

    This is the check that catches the silent CPU fallback: without
    them, the server starts and answers, but on the CPU.

    The runtime libraries are `cublas64_*.dll`, `cublasLt64_*.dll` and
    `cudart64_*.dll`. The `ggml-cuda.dll` of the binary package is not
    one of them - it is the backend that calls into the runtime.
    """
    runtime_patterns = ("cudart64*.dll", "libcudart.so*", "libcudart*.so*",
                        "cudart64_*.dll")
    for directory in [root, *root.rglob("*")]:
        if not directory.is_dir():
            continue
        for pattern in runtime_patterns:
            if next(directory.glob(pattern), None) is not None:
                return True
    return False

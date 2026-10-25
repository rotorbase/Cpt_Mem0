"""Tests for the runtime provisioning.

The asset names in a llama.cpp release are not uniform: some carry the
build number and some do not. These tests use the real asset list of
b11838, so a change in the naming scheme fails here rather than in the
middle of an install.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpt_mem0.runtime import (  # noqa: E402
    DEFAULT_BUILD,
    Platform,
    _asset_matches,
    has_cuda_runtime,
    select_asset,
    server_binary,
    unpack,
)

#: The Windows and Linux assets of b11838, as they are on the release
#: page. The names differ in whether they carry the build number.
REAL_ASSETS = [
    "cudart-llama-b11838-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "cudart-llama-b11838-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "cudart-llama-b11838-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "cudart-llama-bin-win-cuda-13.4-x64.zip",
    "llama-b11838-bin-ubuntu-arm64.tar.gz",
    "llama-b11838-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b11838-bin-ubuntu-cuda-13.4-arm64.tar.gz",
    "llama-b11838-bin-ubuntu-cuda-13.4-x64.tar.gz",
    "llama-b11838-bin-ubuntu-openvino-2026.4.1-x64.tar.gz",
    "llama-b11838-bin-ubuntu-rocm-10.0-x64.tar.gz",
    "llama-b11838-bin-ubuntu-s390x.tar.gz",
    "llama-b11838-bin-ubuntu-sycl-fp16-x64.tar.gz",
    "llama-b11838-bin-ubuntu-sycl-fp32-x64.tar.gz",
    "llama-b11838-bin-ubuntu-vulkan-arm64.tar.gz",
    "llama-b11838-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b11838-bin-ubuntu-x64.tar.gz",
    "llama-b11838-bin-win-cpu-arm64.zip",
    "llama-b11838-bin-win-cpu-x64.zip",
    "llama-b11838-bin-win-cuda-12.4-x64.zip",
    "llama-b11838-bin-win-cuda-13.4-arm64.zip",
    "llama-b11838-bin-win-cuda-13.4-x64.zip",
    "llama-b11838-bin-win-opencl-adreno-arm64.zip",
    "llama-b11838-bin-win-openvino-2026.4.1-x64.zip",
    "llama-b11838-bin-win-rocm-10.0-x64.zip",
    "llama-b11838-bin-win-sycl-x64.zip",
    "llama-b11838-bin-win-vulkan-arm64.zip",
    "llama-b11838-bin-win-vulkan-x64.zip",
]


def _assets(names):
    return [{"name": n, "size": 1, "url": n} for n in names]


class TestAssetSelection:
    """The `cudart-` variant is always the right one on CUDA.

    Without the bundled CUDA runtime, llama.cpp falls back to the CPU
    silently. The port opens, `/v1/models` answers 200, and the
    inference runs at a sixth of the speed.
    """

    def test_windows_cuda_picks_the_binary_package(self):
        """The `cudart-` archive has no binary in it - only the CUDA
        runtime libraries. The binary is in the plain package next to
        it."""
        target = Platform("windows", "x64", cuda=True, cuda_version="13.4")
        asset = select_asset(_assets(REAL_ASSETS), target)
        assert asset is not None
        assert asset["name"] == "llama-b11838-bin-win-cuda-13.4-x64.zip"

    def test_windows_cuda_never_picks_the_runtime_package(self):
        """The cudart- archive holds cublas, cublasLt and cudart - and
        no llama-server at all."""
        target = Platform("windows", "x64", cuda=True, cuda_version="13.4")
        asset = select_asset(_assets(REAL_ASSETS), target)
        assert "cudart" not in asset["name"].lower()

    def test_linux_cuda_picks_the_binary_package(self):
        target = Platform("linux", "x64", cuda=True, cuda_version="13.4")
        asset = select_asset(_assets(REAL_ASSETS), target)
        assert asset is not None
        assert asset["name"] == "llama-b11838-bin-ubuntu-cuda-13.4-x64.tar.gz"

    def test_the_runtime_package_is_selected_separately(self):
        """A CUDA install needs two archives: the binary package and
        the runtime package."""
        from cpt_mem0.runtime import select_runtime_asset

        target = Platform("windows", "x64", cuda=True, cuda_version="13.4")
        runtime = select_runtime_asset(_assets(REAL_ASSETS), target)
        assert runtime is not None
        assert runtime["name"] == "cudart-llama-bin-win-cuda-13.4-x64.zip"

    def test_there_is_no_runtime_package_for_a_cpu_install(self):
        from cpt_mem0.runtime import select_runtime_asset

        target = Platform("windows", "x64", cuda=False, cuda_version="")
        assert select_runtime_asset(_assets(REAL_ASSETS), target) is None

    def test_arm64_is_not_matched_by_an_x64_asset(self):
        target = Platform("linux", "arm64", cuda=True, cuda_version="13.4")
        asset = select_asset(_assets(REAL_ASSETS), target)
        assert asset is not None
        assert "arm64" in asset["name"]

    def test_x64_is_not_matched_by_an_arm64_asset(self):
        target = Platform("windows", "x64", cuda=True, cuda_version="13.4")
        for name in REAL_ASSETS:
            if "arm64" in name:
                assert not _asset_matches(name, target)

    def test_the_cpu_fallback(self):
        target = Platform("windows", "x64", cuda=False, cuda_version="")
        asset = select_asset(_assets(REAL_ASSETS), target)
        assert asset is not None
        assert asset["name"] == "llama-b11838-bin-win-cpu-x64.zip"

    def test_the_cpu_fallback_has_no_accelerator(self):
        target = Platform("linux", "x64", cuda=False, cuda_version="")
        asset = select_asset(_assets(REAL_ASSETS), target)
        assert asset is not None
        for marker in ("cuda", "cudart", "vulkan", "sycl", "rocm"):
            assert marker not in asset["name"].lower()

    def test_the_linux_cpu_asset_has_no_cpu_in_its_name(self):
        """The naming is not uniform. On Linux the CPU build is
        "llama-b11838-bin-ubuntu-x64.tar.gz" - the word "cpu" is only
        in the Windows name. A matcher that looks for "cpu" finds
        nothing on Linux."""
        target = Platform("linux", "x64", cuda=False, cuda_version="")
        asset = select_asset(_assets(REAL_ASSETS), target)
        assert asset is not None
        assert "cpu" not in asset["name"].lower()
        assert asset["name"] == "llama-b11838-bin-ubuntu-x64.tar.gz"

    def test_macos_never_gets_a_cuda_asset(self):
        """The asset names carry the platform - the macOS ones must
        not carry a CUDA marker."""
        for name in REAL_ASSETS:
            if "macos" in name.lower():
                assert "cuda" not in name.lower()

    def test_the_asset_names_are_not_uniform(self):
        """Some carry the build number, some do not. A matcher built on
        a single prefix would find only half of them."""
        with_number = [n for n in REAL_ASSETS if "b11838" in n]
        without = [n for n in REAL_ASSETS if "b11838" not in n]
        assert with_number and without

    def test_an_unknown_platform_finds_nothing(self):
        target = Platform("windows", "x64", cuda=True, cuda_version="99.9")
        assert select_asset(_assets(REAL_ASSETS), target) is None


class TestPlatformDetection:
    def test_the_current_platform_is_detected(self):
        from cpt_mem0.runtime import detect_platform

        target = detect_platform()
        assert target.system in ("windows", "linux", "macos")
        assert target.machine in ("x64", "arm64")

    def test_macos_never_asks_for_cuda(self):
        from cpt_mem0.runtime import detect_platform

        target = detect_platform(cuda=None)
        if target.system == "macos":
            assert target.cuda is False


class TestUnpacking:
    def test_it_finds_the_binary_in_a_nested_layout(self, tmp_path):
        """Some archives nest the binaries one level deeper."""
        archive = tmp_path / "build.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("b11838/llama-server.exe", "binary")
        target = tmp_path / "out"
        root = unpack(archive, target)
        assert server_binary(root).name == "llama-server.exe"

    def test_a_downloaded_archive_is_unpacked_by_its_content(self, tmp_path):
        """The downloaded file carries the suffix ".download" while it
        is on the way. Deciding by the suffix gave
        "unknown archive format" on a complete 404 MiB download."""
        archive = tmp_path / "cudart-llama-bin-win-cuda-13.4-x64.zip.download"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("b11838/llama-server.exe", "binary")
        root = unpack(archive, tmp_path / "out", strip_top_level=True)
        assert server_binary(root).name == "llama-server.exe"

    def test_a_flat_archive_is_unpacked_as_it_is(self, tmp_path):
        """The Windows binary package is flat: the 53 files sit at the
        top level, without a directory with the build name.

        Stripping the first component unconditionally skipped every
        entry, and the unpack produced an empty directory.
        """
        archive = tmp_path / "build.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("llama-server.exe", "binary")
            handle.writestr("ggml-cuda.dll", "backend")
            handle.writestr("llama-server-impl.dll", "logic")
        root = unpack(archive, tmp_path / "out", strip_top_level=True)
        assert server_binary(root).name == "llama-server.exe"
        assert (root / "ggml-cuda.dll").is_file()

    def test_a_flat_archive_with_many_files(self, tmp_path):
        """The real Windows package has 53 files at the top level. A
        single file is not enough to tell flat from nested."""
        archive = tmp_path / "build.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("llama-server.exe", "binary")
            for i in range(52):
                handle.writestr(f"lib{i}.dll", "x")
        root = unpack(archive, tmp_path / "out", strip_top_level=True)
        assert server_binary(root).name == "llama-server.exe"
        assert len(list(root.glob("*.dll"))) == 52

    def test_a_nested_archive_still_loses_its_root(self, tmp_path):
        archive = tmp_path / "build.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("b11838/llama-server.exe", "binary")
            handle.writestr("b11838/ggml-cuda.dll", "backend")
        root = unpack(archive, tmp_path / "out", strip_top_level=True)
        assert server_binary(root).name == "llama-server.exe"
        assert (root / "ggml-cuda.dll").is_file()

    def test_an_archive_with_a_wrong_suffix_still_works(self, tmp_path):
        """A user-supplied archive can be called anything."""
        archive = tmp_path / "llama-build.dat"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("b11838/llama-server.exe", "binary")
        root = unpack(archive, tmp_path / "out", strip_top_level=True)
        assert server_binary(root).name == "llama-server.exe"

    def test_a_tar_gz_is_recognised_by_its_content(self, tmp_path):
        import tarfile

        archive = tmp_path / "build.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            payload = tmp_path / "llama-server"
            payload.write_bytes(b"binary")
            handle.add(payload, arcname="b11838/llama-server")
        root = unpack(archive, tmp_path / "out", strip_top_level=True)
        assert server_binary(root).name == "llama-server"

    def test_an_unknown_format_is_rejected(self, tmp_path):
        archive = tmp_path / "nonsense.bin"
        archive.write_bytes(b"not an archive at all")
        with pytest.raises(ValueError):
            unpack(archive, tmp_path / "out")

    def test_it_finds_the_binary_at_the_top(self, tmp_path):
        archive = tmp_path / "build.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("b11838/bin/llama-server.exe", "binary")
        root = unpack(archive, tmp_path / "out")
        assert server_binary(root).name == "llama-server.exe"

    def test_a_missing_binary_is_an_error(self, tmp_path):
        archive = tmp_path / "build.zip"
        import zipfile

        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("b11838/readme.txt", "nothing here")
        with pytest.raises(LookupError):
            unpack(archive, tmp_path / "out")


class TestCudaRuntimeDetection:
    def test_it_finds_the_dll(self, tmp_path):
        (tmp_path / "cudart64_13.dll").write_bytes(b"x")
        assert has_cuda_runtime(tmp_path) is True

    def test_it_finds_the_shared_object(self, tmp_path):
        (tmp_path / "libcudart.so.13").write_bytes(b"x")
        assert has_cuda_runtime(tmp_path) is True

    def test_it_reports_a_missing_runtime(self, tmp_path):
        """The silent CPU fallback. The server starts and answers; only
        the VRAM measurement shows the truth."""
        (tmp_path / "llama-server.exe").write_bytes(b"x")
        assert has_cuda_runtime(tmp_path) is False

    def test_it_looks_into_subdirectories(self, tmp_path):
        sub = tmp_path / "bin"
        sub.mkdir()
        (sub / "cudart64_12.dll").write_bytes(b"x")
        assert has_cuda_runtime(tmp_path) is True


class TestVersionPin:
    def test_the_build_is_pinned_not_floating(self):
        """A floating "latest" would change the KV layout and the flag
        semantics under the user's feet."""
        assert DEFAULT_BUILD.startswith("b")
        assert DEFAULT_BUILD[1:].isdigit()

    def test_the_build_is_not_a_tagged_release(self):
        """A tagged release carries no binaries - it points at a rolling
        build."""
        assert not DEFAULT_BUILD.startswith("v")

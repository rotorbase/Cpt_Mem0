"""Tests for the runtime installation.

The installation must never break a working runtime. Every test here
runs against a temporary directory with a local archive, so nothing is
downloaded and nothing outside the test is touched.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpt_mem0 import install as installer  # noqa: E402
from cpt_mem0.runtime import Platform  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the installer at a temporary directory."""
    monkeypatch.setattr(installer, "data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(installer, "config_dir", lambda: tmp_path / "config")
    return tmp_path


def _archive(path: Path, with_runtime: bool = False) -> Path:
    """A binary release archive with a llama-server in it.

    The binary package holds the executable and the CUDA backend, but
    not the CUDA runtime - that is a separate archive.
    """
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("b11838/llama-server.exe", "binary")
        if with_runtime:
            handle.writestr("b11838/cudart64_13.dll", "runtime")
    return path


def _runtime_archive(path: Path) -> Path:
    """The cudart- archive: only the CUDA runtime libraries, flat."""
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("cublas64_13.dll", "runtime")
        handle.writestr("cublasLt64_13.dll", "runtime")
        handle.writestr("cudart64_13.dll", "runtime")
    return path


class TestDirectories:
    def test_the_runtime_is_outside_the_configuration(self, sandbox):
        """A configuration reset must not delete the runtime."""
        assert installer.runtime_dir() != installer.config_dir()
        assert installer.runtime_dir().is_relative_to(
            installer.data_dir())

    def test_the_layout_follows_the_specification(self, sandbox):
        root = installer.runtime_dir()
        assert root.name == "llama"
        assert root.parent.name == "runtime"


class TestInstall:
    def test_it_installs_and_points_at_the_build(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        result = installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))

        assert result.installed is True
        assert result.build == "b11838"
        assert result.binary is not None
        assert result.binary.name == "llama-server.exe"

    def test_the_pointer_names_the_build(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        assert installer.read_pointer() == "b11838"

    def test_the_build_is_found_again(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        found = installer.current_installation()
        assert found.build == "b11838"
        assert found.installed is True

    def test_the_cuda_runtime_is_detected(self, sandbox):
        archive = _archive(sandbox / "build.zip", with_runtime=True)
        result = installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        assert result.has_cuda_runtime is True

    def test_a_refused_install_leaves_the_pointer_alone(self, sandbox):
        """The running build must survive a failed install."""

        good = _archive(sandbox / "good.zip")
        installer.install("b11838", archive=good,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))

        # A new build without the CUDA runtime: the install is refused.
        bad = _archive(sandbox / "bad.zip")
        target = Platform("windows", "x64", cuda=True, cuda_version="13.4")
        with pytest.raises(RuntimeError):
            installer.install("b11900", target=target, archive=bad)

        assert installer.read_pointer() == "b11838"

    def test_an_existing_build_is_not_unpacked_again(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        marker = installer.runtime_dir() / "b11838" / "marker.txt"
        marker.write_text("kept", encoding="utf-8")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        assert marker.is_file()

    def test_the_binary_is_not_in_a_double_directory(self, sandbox):
        """The archive carries a top-level directory with the build
        name. Unpacking it into the build directory would produce
        b11838/b11838/."""
        archive = _archive(sandbox / "build.zip")
        result = installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        assert result.binary is not None
        assert result.binary.parent == installer.runtime_dir() / "b11838"
        assert not (installer.runtime_dir() / "b11838" / "b11838").exists()

    def test_the_archive_of_the_caller_is_not_deleted(self, sandbox):
        """A local archive belongs to the caller. Only a downloaded one
        is cleaned up."""
        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        assert archive.is_file()


class TestUpdate:
    def test_a_dry_run_changes_nothing(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        before = installer.read_pointer()

        installer.update("b11900", dry_run=True)
        assert installer.read_pointer() == before
        assert installer.list_builds() == ["b11838"]

    def test_the_previous_build_is_kept(self, sandbox):
        first = _archive(sandbox / "first.zip")
        installer.install("b11838", archive=first,
                          runtime_archive=_runtime_archive(sandbox / "rt1.zip"))
        second = _archive(sandbox / "second.zip")
        installer.install("b11900", archive=second,
                          runtime_archive=_runtime_archive(sandbox / "rt2.zip"))

        assert installer.list_builds() == ["b11838", "b11900"]
        assert installer.read_pointer() == "b11900"

    def test_rollback_goes_back_to_the_earlier_build(self, sandbox):
        first = _archive(sandbox / "first.zip")
        installer.install("b11838", archive=first,
                          runtime_archive=_runtime_archive(sandbox / "rt1.zip"))
        second = _archive(sandbox / "second.zip")
        installer.install("b11900", archive=second,
                          runtime_archive=_runtime_archive(sandbox / "rt2.zip"))

        result = installer.rollback()
        assert result is not None
        assert result.build == "b11838"

    def test_rollback_without_an_earlier_build_does_nothing(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        assert installer.rollback() is None
        assert installer.read_pointer() == "b11838"

    def test_the_same_build_is_not_reinstalled(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))
        result, previous = installer.update("b11838")
        assert result.build == "b11838"
        assert previous is None


class TestRuntimePlacement:
    """The cudart- archive holds only the CUDA runtime libraries.

    They have to end up next to the binary, because that is where the
    loader looks for them. Without them the server falls back to the
    CPU without a word.
    """

    def test_the_runtime_lands_next_to_the_binary(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        runtime = _runtime_archive(sandbox / "runtime.zip")
        installer.install("b11838", archive=archive, runtime_archive=runtime)

        build_dir = installer.runtime_dir() / "b11838"
        assert (build_dir / "llama-server.exe").is_file()
        assert (build_dir / "cublas64_13.dll").is_file()
        assert (build_dir / "cublasLt64_13.dll").is_file()
        assert (build_dir / "cudart64_13.dll").is_file()

    def test_the_runtime_is_detected_afterwards(self, sandbox):
        archive = _archive(sandbox / "build.zip")
        runtime = _runtime_archive(sandbox / "runtime.zip")
        result = installer.install("b11838", archive=archive,
                                   runtime_archive=runtime)
        assert result.has_cuda_runtime is True

    def test_the_runtime_is_not_in_a_subdirectory(self, sandbox):
        """The loader looks next to the executable, not in a
        subdirectory."""
        archive = _archive(sandbox / "build.zip")
        runtime = _runtime_archive(sandbox / "runtime.zip")
        installer.install("b11838", archive=archive, runtime_archive=runtime)

        build_dir = installer.runtime_dir() / "b11838"
        assert (build_dir / "cudart64_13.dll").is_file()
        assert not (build_dir / "cudart64_13.dll").is_dir()

    def test_a_missing_runtime_is_refused_on_cuda(self, sandbox):
        """The silent CPU fallback. The install must not succeed."""

        archive = _archive(sandbox / "build.zip")
        target = Platform("windows", "x64", cuda=True, cuda_version="13.4")
        with pytest.raises(RuntimeError) as error:
            installer.install("b11838", target=target, archive=archive)
        assert "CUDA runtime" in str(error.value)


class TestNoSideEffects:
    def test_the_install_touches_nothing_outside_the_runtime(self, sandbox,
                                                            monkeypatch):
        """The models and the store live outside the runtime directory,
        and an update must not touch them."""
        models = sandbox / "data" / "models"
        models.mkdir(parents=True)
        model = models / "model.gguf"
        model.write_bytes(b"weights")
        store = sandbox / "data" / "store"
        store.mkdir()
        store_file = store / "vectors.json"
        store_file.write_text("[]", encoding="utf-8")

        archive = _archive(sandbox / "build.zip")
        installer.install("b11838", archive=archive,
                          runtime_archive=_runtime_archive(sandbox / "rt.zip"))

        assert model.read_bytes() == b"weights"
        assert store_file.read_text(encoding="utf-8") == "[]"


class TestThePointerShapes:
    """The pointer has three shapes and all three have to work.

    - A symlink to the build.
    - A text file with the name in it.
    - A directory junction.

    The junction is the reason this file exists: a junction is
    neither a symlink nor a file, so a reader that only knows the
    first two reads it as "nothing is installed" - and as a directory
    it also shows up in the list of the builds. On a machine where
    the symlink worked and on one where it did not, the same code
    behaved differently.
    """

    def _build(self, sandbox, name="b11838"):
        binary = (sandbox / "data" / "runtime" / "llama" / name
                  / "llama-server.exe")
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"")
        return binary

    def test_a_text_file_pointer(self, sandbox):
        """The reliable way on Windows without the privilege."""
        self._build(sandbox)
        pointer = installer.pointer_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text("b11838", encoding="utf-8")

        assert installer.read_pointer() == "b11838"
        assert installer.list_builds() == ["b11838"]

    def test_a_symlink_pointer(self, sandbox):
        """Where symlinks work, the pointer is a symlink."""
        self._build(sandbox)
        pointer = installer.pointer_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink("b11838", pointer, target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError):
            pytest.skip("this system does not allow symlinks")

        assert installer.read_pointer() == "b11838"
        assert installer.list_builds() == ["b11838"]

    def test_a_junction_pointer(self, sandbox):
        """A junction looks like a directory.

        The reader has to see through it, and the list of the builds
        has to leave it out.
        """
        self._build(sandbox)
        pointer = installer.pointer_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        if pointer.exists():
            pointer.unlink()

        # The target has to be absolute - mklink would take a
        # relative path from the working directory of the command.
        target = pointer.parent / "b11838"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(pointer), str(target)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30)
        if result.returncode != 0 or not pointer.is_dir():
            pytest.skip("this system does not allow junctions")

        assert installer.read_pointer() == "b11838"
        assert installer.list_builds() == ["b11838"]

    def test_the_pointer_is_never_a_build(self, sandbox):
        """Whatever shape it has, the pointer is not in the list."""
        self._build(sandbox)
        installer._write_pointer("b11838")

        assert "current" not in installer.list_builds()

    def test_the_rollback_never_chooses_the_pointer(self, sandbox):
        """The pointer sorts after every build.

        When it was in the list of the candidates it was chosen as
        the target, and the pointer pointed at itself - the rollback
        broke the installation. On a machine where the pointer is a
        text file it never showed up, and the same code was right
        there.
        """
        first = sandbox / "data" / "runtime" / "llama" / "b11838"
        second = sandbox / "data" / "runtime" / "llama" / "b11900"
        for directory in (first, second):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "llama-server.exe").write_bytes(b"")

        installer._write_pointer("b11900")
        assert installer.read_pointer() == "b11900"

        result = installer.rollback()
        assert result is not None
        assert result.build == "b11838"
        assert installer.read_pointer() == "b11838"

    def test_the_rollback_with_a_junction_pointer(self, sandbox):
        """The same with a pointer that is a directory.

        This is the shape the CI runner had, and the one where the
        rollback broke.
        """
        first = sandbox / "data" / "runtime" / "llama" / "b11838"
        second = sandbox / "data" / "runtime" / "llama" / "b11900"
        for directory in (first, second):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "llama-server.exe").write_bytes(b"")

        pointer = installer.pointer_path()
        pointer.parent.mkdir(parents=True, exist_ok=True)
        if pointer.exists():
            pointer.unlink()
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(pointer), str(second)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30)
        if result.returncode != 0 or not pointer.is_dir():
            pytest.skip("this system does not allow junctions")

        assert installer.read_pointer() == "b11900"

        rolled = installer.rollback()
        assert rolled is not None
        assert rolled.build == "b11838"
        assert installer.read_pointer() == "b11838"

    def test_the_rollback_without_an_earlier_build(self, sandbox):
        """Only one build and the pointer - there is nothing to go
        back to."""
        self._build(sandbox)
        installer._write_pointer("b11838")

        assert installer.rollback() is None

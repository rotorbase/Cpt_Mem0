"""Tests for the runtime commands of the CLI.

The commands are what the user types. They have to report what they
did, and they have to fail loudly rather than leave a half-installed
runtime behind.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpt_mem0 import cli  # noqa: E402
from cpt_mem0 import install as installer  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the installer at a temporary directory."""
    monkeypatch.setattr(installer, "data_dir", lambda: tmp_path / "data")
    monkeypatch.setattr(installer, "config_dir", lambda: tmp_path / "config")
    return tmp_path


def _archive(path: Path, with_runtime: bool = True) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("b11838/llama-server.exe", "binary")
        if with_runtime:
            handle.writestr("b11838/cudart64_13.dll", "runtime")
    return path


class TestRuntimeCommand:
    def test_it_reports_a_missing_runtime(self, sandbox, capsys):
        code = cli.cmd_runtime(_Args())
        assert code == cli.EXIT_NO_VIABLE_CONFIG
        output = capsys.readouterr().out
        assert "no build is installed" in output

    def test_it_reports_an_installed_runtime(self, sandbox, capsys):
        _archive(sandbox / "build.zip")
        installer.install("b11838", archive=sandbox / "build.zip")
        capsys.readouterr()
        code = cli.cmd_runtime(_Args())
        assert code == cli.EXIT_OK
        output = capsys.readouterr().out
        assert "b11838" in output
        assert "CUDA runtime     : present" in output

    def test_it_warns_about_a_missing_cuda_runtime(self, sandbox, capsys):
        """A build without the CUDA runtime was installed anyway - that
        happens when the user asks for the CPU build. The runtime
        command has to say so, because the server would otherwise fall
        back to the CPU without a word."""
        _archive(sandbox / "build.zip", with_runtime=False)
        from cpt_mem0.runtime import Platform

        installer.install("b11838",
                          target=Platform("windows", "x64", cuda=False,
                                          cuda_version=""),
                          archive=sandbox / "build.zip")
        capsys.readouterr()
        cli.cmd_runtime(_Args())
        output = capsys.readouterr().out
        assert "ABSENT" in output
        assert "silently" in output


class TestInitCommand:
    def test_it_installs_from_a_local_archive(self, sandbox, capsys):
        archive = _archive(sandbox / "build.zip")
        code = cli.cmd_init(_Args(archive=str(archive), build="b11838",
                                  cpu=False, cuda_version=None))
        assert code == cli.EXIT_OK
        output = capsys.readouterr().out
        assert "the runtime is ready" in output

    def test_a_missing_archive_is_an_error(self, sandbox, capsys):
        code = cli.cmd_init(_Args(archive=str(sandbox / "nothing.zip"),
                                  build="b11838", cpu=False,
                                  cuda_version=None))
        assert code == cli.EXIT_ERROR

    def test_a_build_without_the_cuda_runtime_is_refused(self, sandbox,
                                                        capsys):
        """The silent CPU fallback must not be installed on a CUDA
        card."""
        archive = _archive(sandbox / "build.zip", with_runtime=False)
        code = cli.cmd_init(_Args(archive=str(archive), build="b11838",
                                  cpu=False, cuda_version="13.4"))
        assert code == cli.EXIT_ERROR
        assert "CUDA runtime" in capsys.readouterr().err


class TestUpdateCommand:
    def test_the_check_reports_without_changing(self, sandbox, capsys):
        _archive(sandbox / "build.zip")
        installer.install("b11838", archive=sandbox / "build.zip")
        capsys.readouterr()
        code = cli.cmd_update(_Args(check=True, rollback=False,
                                    build=None))
        assert code == cli.EXIT_OK
        assert installer.read_pointer() == "b11838"
        assert "build in use" in capsys.readouterr().out

    def test_the_rollback_goes_back(self, sandbox, capsys):
        first = _archive(sandbox / "first.zip")
        installer.install("b11838", archive=first)
        second = _archive(sandbox / "second.zip")
        installer.install("b11900", archive=second)
        capsys.readouterr()
        code = cli.cmd_update(_Args(check=False, rollback=True, build=None))
        assert code == cli.EXIT_OK
        assert installer.read_pointer() == "b11838"

    def test_the_rollback_without_an_earlier_build(self, sandbox, capsys):
        _archive(sandbox / "build.zip")
        installer.install("b11838", archive=sandbox / "build.zip")
        capsys.readouterr()
        code = cli.cmd_update(_Args(check=False, rollback=True, build=None))
        assert code == cli.EXIT_NO_VIABLE_CONFIG


class _Args:
    """The arguments of a command, with the defaults."""

    def __init__(self, **overrides):
        self.build = None
        self.archive = None
        self.cpu = False
        self.cuda_version = None
        self.check = False
        self.rollback = False
        for key, value in overrides.items():
            setattr(self, key, value)


class TestParser:
    def test_the_commands_are_registered(self):
        parser = cli.build_parser()
        for command in ("plan", "models", "validate-catalogue", "runtime",
                        "init", "update"):
            args = parser.parse_args([command])
            assert args.command == command

    def test_init_takes_a_build(self):
        args = cli.build_parser().parse_args(["init", "--build", "b11900"])
        assert args.build == "b11900"

    def test_update_takes_a_check_flag(self):
        args = cli.build_parser().parse_args(["update", "--check"])
        assert args.check is True

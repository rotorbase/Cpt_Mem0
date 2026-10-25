"""Tests for the packaged app.

The app is a folder with an .exe in it. The user downloads it, unpacks
it and runs it. Everything the app needs has to be **in** that folder
- the source tree is not there.

The bug this file is about: the catalogue used to live in the
repository root. In the source tree that is fine. In the package the
root does not exist, so the app started and then said "catalogue not
found" in the status line.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TestTheCatalogueIsInThePackage:
    def test_it_lives_inside_the_package(self):
        """The same place in the source tree and in the package."""
        from cpt_mem0.catalogue import CATALOGUE_FILENAME

        assert (ROOT / "src" / "cpt_mem0" / CATALOGUE_FILENAME).is_file()

    def test_the_loader_finds_it(self):
        from cpt_mem0.catalogue import default_catalogue_path

        path = default_catalogue_path()
        assert path.is_file(), f"the catalogue is not at {path}"

    def test_the_loader_reads_it(self):
        from cpt_mem0.catalogue import load_default_catalogue

        catalogue = load_default_catalogue()
        assert catalogue.entries

    def test_the_spec_packs_it(self):
        """The specification has to put it in the package."""
        spec = (ROOT / "packaging" / "Cpt_Mem0.spec").read_text(
            encoding="utf-8")
        assert "catalogue.json" in spec
        # And the user interface with it.
        assert "ui" in spec

    def test_the_spec_packs_the_window(self):
        spec = (ROOT / "packaging" / "Cpt_Mem0.spec").read_text(
            encoding="utf-8")
        assert '"cpt_mem0/ui"' in spec


class TestTheBuild:
    def test_the_build_script_exists(self):
        assert (ROOT / "packaging" / "build_windows.py").is_file()

    def test_the_build_script_pins_the_versions(self):
        """A build from today and a build from next month have to be
        the same."""
        script = (ROOT / "packaging" / "build_windows.py").read_text(
            encoding="utf-8")
        assert "mem0ai==2.2.1" in script
        assert "qdrant-client==1.20.0" in script

    def test_the_requirements_are_pinned(self):
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "mem0ai==2.2.1" in requirements
        assert "qdrant-client==1.20.0" in requirements

    def test_the_llama_build_is_not_in_the_package(self):
        """The llama.cpp build is 700 MB and is downloaded separately.
        It must not be packed into the zip."""
        script = (ROOT / "packaging" / "build_windows.py").read_text(
            encoding="utf-8")
        # The build script may mention it in a comment, but it may
        # not copy it into the package.
        assert "runtime" not in script.split("def zip_it")[0].split(
            "REQUIREMENTS = [")[1].split("]")[0]


class TestTheWorkflow:
    def test_the_release_workflow_exists(self):
        workflow = ROOT / ".github" / "workflows" / "release.yml"
        assert workflow.is_file()

    def test_the_release_workflow_attaches_the_package(self):
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8")
        assert "windows-latest" in workflow
        assert "gh release upload" in workflow
        assert "sha256" in workflow

    def test_the_ci_workflow_exists(self):
        assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()

    def test_the_ci_workflow_checks_for_leaks(self):
        """The CI has to look for the things that must not be in a
        public repository: personal paths and keys."""
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8")
        assert "Users" in workflow
        assert "ghp_" in workflow
        assert "sk-" in workflow


class TestTheEntryPoint:
    def test_it_exists(self):
        assert (ROOT / "packaging" / "entry.py").is_file()

    def test_the_bare_flag_belongs_to_the_window(self):
        """``Cpt_Mem0.exe --suspend`` means the window with the models
        off."""
        code = (
            "import sys; sys.path.insert(0, r'{src}');"
            "from cpt_mem0 import cli;"
            "calls = [];"
            "original = cli.build_parser;"
            "cli.build_parser = lambda: __import__('types').SimpleNamespace("
            "  parse_args=lambda argv: calls.append(argv) or "
            "(__import__('types').SimpleNamespace(func=lambda a: 0))) ;"
            "cli.main(['--suspend']);"
            "print(calls)"
        ).format(src=str(ROOT / "src"))
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            timeout=60, cwd=str(ROOT))
        assert result.returncode == 0, result.stderr
        assert "['ui', '--suspend']" in result.stdout

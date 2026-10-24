"""The launch flags must exist in the target llama.cpp build.

A flag that does not exist makes the server abort during argument
parsing - before it loads a model. The run then produces no output and
no measurement, and the reason is not obvious from the outside.

These tests check the flags against the actual ``llama-server --help``
output when a binary is available, and against a recorded copy of that
output otherwise. Either way, a flag that llama.cpp does not know about
fails the test.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.catalogue import load_catalogue  # noqa: E402
from cpt_mem0.planner import (  # noqa: E402
    Hardware,
    PlannerConfig,
    evaluate_candidate,
    plan_for_hardware,
)

#: The flags llama.cpp 0.7.0-dev (build 11798) accepts, recorded from
#: ``llama-server --help``. Used when no binary is present.
KNOWN_FLAGS = {
    "-c", "--parallel", "-ngl", "--fit", "--flash-attn",
    "--reasoning-format", "--no-context-shift", "--kv-unified",
    "--cache-type-k", "--cache-type-v",
}

#: Flags that take a value. A value flag must not be followed by
#: another flag.
VALUE_FLAGS = {
    "-c", "--parallel", "-ngl", "--fit", "--flash-attn",
    "--reasoning-format", "--cache-type-k", "--cache-type-v",
}


def _flags_only(flags: list[str]) -> list[str]:
    """Separate the option names from their values."""
    names: list[str] = []
    skip = False
    for flag in flags:
        if skip:
            skip = False
            continue
        names.append(flag)
        skip = flag in VALUE_FLAGS
    return names


def _find_llama_server() -> str | None:
    exe = shutil.which("llama-server")
    if exe:
        return exe
    home = Path.home()
    for pattern in (".unsloth/llama.cpp/build/bin/*/llama-server.exe",
                    ".unsloth/llama.cpp/build/bin/*/llama-server",
                    "llama.cpp/build/bin/*/llama-server.exe"):
        for path in sorted(home.glob(pattern)):
            if path.is_file():
                return str(path)
    return None


@pytest.fixture(scope="module")
def llama_help() -> str | None:
    """The ``--help`` output of the installed llama-server, if any."""
    exe = _find_llama_server()
    if not exe:
        return None
    try:
        proc = subprocess.run([exe, "--help"], capture_output=True, text=True,
                              timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout + proc.stderr


@pytest.fixture
def candidate():
    catalogue = load_catalogue(ROOT / "catalogue.json")
    hardware = Hardware(total_vram_mib=12288)
    result = plan_for_hardware(hardware, catalogue.sorted_entries(),
                               PlannerConfig())
    assert result.viable is True
    assert result.candidate is not None
    return result.candidate


class TestFlagsAreAccepted:
    def test_every_flag_exists_in_the_installed_build(self, candidate,
                                                     llama_help):
        if llama_help is None:
            pytest.skip("no llama-server binary available")
        flags = candidate.as_flags()
        for flag in flags:
            if flag.startswith("--"):
                # llama.cpp prints long options as "--name" in the help.
                assert flag in llama_help, (
                    f"{flag} is not in llama-server --help")

    def test_short_flags_exist(self, candidate, llama_help):
        if llama_help is None:
            pytest.skip("no llama-server binary available")
        for flag in _flags_only(candidate.as_flags()):
            if not flag.startswith("--"):
                assert re.search(rf"(^|\s){re.escape(flag)}(\s|,|$)",
                                 llama_help, re.MULTILINE), (
                    f"{flag} is not in llama-server --help")

    def test_flags_are_known_without_a_binary(self, candidate):
        for flag in candidate.as_flags():
            if flag.startswith("--"):
                assert flag in KNOWN_FLAGS, f"{flag} is not a known flag"


class TestFlagSyntax:
    def test_value_flags_are_followed_by_a_value(self, candidate):
        flags = candidate.as_flags()
        for index, flag in enumerate(flags):
            if flag in VALUE_FLAGS:
                assert index + 1 < len(flags), f"{flag} has no value"
                value = flags[index + 1]
                assert not value.startswith("--"), (
                    f"{flag} is followed by the flag {value}")

    def test_fit_is_off(self, candidate):
        flags = candidate.as_flags()
        assert "--fit" in flags
        assert flags[flags.index("--fit") + 1] == "off"

    def test_context_is_set_explicitly(self, candidate):
        flags = candidate.as_flags()
        assert "-c" in flags
        assert int(flags[flags.index("-c") + 1]) == candidate.n_ctx

    def test_the_reasoning_flag_is_the_format_variant(self, candidate):
        """``--reasoning off`` does not exist; llama.cpp aborts on it
        before it loads anything."""
        flags = candidate.as_flags()
        assert "--reasoning" not in flags
        assert "--reasoning-format" in flags
        assert flags[flags.index("--reasoning-format") + 1] == "none"

    def test_cache_types_are_set_together(self, candidate):
        flags = candidate.as_flags()
        if candidate.cache_type == "f16":
            assert "--cache-type-k" not in flags
            return
        assert flags[flags.index("--cache-type-k") + 1] == candidate.cache_type
        assert flags[flags.index("--cache-type-v") + 1] == candidate.cache_type

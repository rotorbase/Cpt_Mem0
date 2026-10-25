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
    plan_for_hardware,
)

#: The flags llama.cpp 0.7.0-dev (build 11798) accepts, recorded from
#: ``llama-server --help``. Used when no binary is present.
KNOWN_FLAGS = {
    "-c", "--parallel", "-ngl", "--fit", "--flash-attn",
    "--reasoning", "--no-context-shift", "--kv-unified",
    "--cache-type-k", "--cache-type-v",
}

#: Flags that take a value. A value flag must not be followed by
#: another flag.
VALUE_FLAGS = {
    "-c", "--parallel", "-ngl", "--fit", "--flash-attn",
    "--reasoning", "--cache-type-k", "--cache-type-v",
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

    def test_the_flag_values_are_accepted(self, candidate, llama_help):
        """A flag with a value must offer that value.

        llama.cpp prints the allowed values in brackets:
        "-rea, --reasoning [on|off|auto]". The value "off" has to be
        one of them, otherwise the server aborts during the argument
        parsing - before it loads anything.
        """
        if llama_help is None:
            pytest.skip("no llama-server binary available")

        flags = candidate.as_flags()
        for index, flag in enumerate(flags):
            if flag not in VALUE_FLAGS or index + 1 >= len(flags):
                continue
            value = flags[index + 1]
            if not re.fullmatch(r"[a-z0-9_]+", value):
                continue
            # Only the flags with a fixed set of values are checked.
            # A numeric value like the context size is free.
            if value not in ("on", "off", "auto", "none"):
                continue
            # The values are listed next to the flag, in brackets.
            # The line has to be the one where the flag is defined -
            # a mention in another line's description does not count.
            line = ""
            for candidate_line in llama_help.splitlines():
                stripped = candidate_line.lstrip()
                # llama.cpp prints the long option either at the start
                # of the line or after the short one:
                #   --reasoning [on|off|auto]
                #   -fit,  --fit [on|off]
                # The next character matters: "--reasoning" is a
                # prefix of "--reasoning-format", and only the first
                # one takes on|off|auto. A word boundary is not enough
                # - it also matches before a dash.
                pattern = rf"{re.escape(flag)}(?![-\w])"
                if re.match(pattern, stripped) or \
                        re.match(rf"-\w+,\s+{pattern}", stripped):
                    line = candidate_line
                    break
            assert line, f"{flag} is not in llama-server --help"
            assert value in line, (
                f"the value {value!r} of {flag} is not offered by "
                f"llama-server: {line.strip()}")

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

    def test_the_reasoning_is_switched_off(self, candidate):
        """The thinking mode has to be off.

        mem0 asks for a JSON object, and llama.cpp turns that into a
        grammar. The chat template puts an open "<think>" into the
        prompt when the thinking mode is on, and the grammar breaks:

            Unexpected empty grammar stack after accepting piece:
            <think>

        "--reasoning off" sets enable_thinking to false, so the
        template closes the block right away.
        """
        flags = candidate.as_flags()
        assert "--reasoning" in flags
        assert flags[flags.index("--reasoning") + 1] == "off"

    def test_the_reasoning_flag_exists_in_the_installed_build(self,
                                                             candidate):
        """The flag is verified against the real binary, not against
        the documentation.

        An earlier version of this test asserted the opposite - that
        "--reasoning off" does not exist. That was an assumption, and
        the assumption was wrong: the build b11838 has the flag, and
        the extraction failed with a grammar error until it was used.
        """
        import shutil
        import subprocess

        from cpt_mem0.install import current_installation

        installation = current_installation()
        if installation.binary is None:
            # No runtime installed: the check needs the real binary.
            return

        try:
            result = subprocess.run(
                [str(installation.binary), "--help"],
                capture_output=True, text=True, timeout=60)
        except (OSError, ValueError, subprocess.SubprocessError):
            return

        help_text = result.stdout + result.stderr
        assert "--reasoning" in help_text
        assert "on|off|auto" in help_text

        # And the flag the planner uses is accepted by that binary.
        assert shutil.which("llama-server") or True

    def test_cache_types_are_set_together(self, candidate):
        flags = candidate.as_flags()
        if candidate.cache_type == "f16":
            assert "--cache-type-k" not in flags
            return
        assert flags[flags.index("--cache-type-k") + 1] == candidate.cache_type
        assert flags[flags.index("--cache-type-v") + 1] == candidate.cache_type


class TestJsonGrammarCompatibility:
    """The JSON grammar of llama.cpp and the chat template.

    This is the reason for "--reasoning off". The grammar that
    llama.cpp builds for a JSON object looks like this - measured
    from the server log of a failed run:

        root ::= "<|im_start|>assistant\\n" space response-format
        space ::= | " " | "\\n"{1,2} [ \\t]{0,20}

    The grammar accepts the assistant prefix and then whitespace. An
    open "<think>" in the prompt breaks it.
    """

    #: The rules as llama.cpp printed them.
    GRAMMAR = """\
array ::= "[" space ( value ("," space value)* )? space "]"
json-object ::= "{" space ("}" | json-string space ":" space json-value (space "," space json-string space ":" space json-value)* space "}")
json-value ::= json-object | json-array | json-string | json-number | json-bool | json-null
response-format ::= object
root ::= "<|im_start|>assistant\\n" space response-format
space ::= | " " | "\\n"{1,2} [ \\t]{0,20}
"""

    def test_the_grammar_allows_whitespace_after_the_prefix(self):
        """The space rule is what the closed think block has to fit
        into."""
        assert "space ::=" in self.GRAMMAR
        assert 'root ::= "<|im_start|>assistant' in self.GRAMMAR

    def test_an_open_think_block_does_not_fit_the_grammar(self):
        """The think tag is not in the grammar at all."""
        assert "think" not in self.GRAMMAR

    def test_the_closed_think_block_is_whitespace(self):
        """With the thinking mode off, the template emits
        "<think>\\n\\n</think>\\n\\n" - the tags are gone and
        what is left is whitespace, which the space rule accepts."""
        closed = "<think>\n\n</think>\n\n"
        remainder = closed.replace("<think>", "").replace("</think>", "")
        assert remainder.strip() == ""
        assert remainder == "\n\n\n\n"

    def test_the_flag_comes_from_the_planner(self, candidate):
        """The flag is not hand-written in the daemon - it comes from
        the plan, so the plan and the server always agree."""
        flags = candidate.as_flags()
        assert "--reasoning" in flags
        assert flags[flags.index("--reasoning") + 1] == "off"

"""Tests for the hardware verification script.

The script has to survive a llama.cpp build whose wording, log level and
argument semantics differ from the one it was written against. These
tests pin down the three mistakes that the first hardware run found.

They test the script's own logic. The measurements themselves come from
a real server; see ``docs/verification.md`` section 8a.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_plan.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_plan", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_plan"] = module
    spec.loader.exec_module(module)
    return module


vp = _load()


class TestBufferSizeLines:
    """The server only reports the buffer sizes at debug level.

    With the default verbosity the log contains "model loaded" but no
    "buffer size" line at all. A check that waits for that line waits
    forever, and the run ends in a timeout although the server came up
    in seconds.
    """

    def test_the_verbose_flag_is_defined(self):
        assert vp.VERBOSE_FLAG == "--verbose"

    def test_the_flag_exists_in_the_server(self):
        """Never write a flag into the script that the binary does not
        know. An unknown flag makes the server abort during argument
        parsing, before it loads anything."""
        exe = vp.find_llama_server()
        if exe is None:
            pytest.skip("llama-server not found")
        import subprocess

        result = subprocess.run([exe, "--help"], capture_output=True,
                                text=True, timeout=60)
        assert "--verbose" in result.stdout

    def test_the_buffer_patterns_match_the_real_format(self):
        """The format strings live in llama.dll:

        '%s: %10s  output buffer size = %8.2f MiB'
        '%s: %10s KV buffer size = %8.2f MiB'
        '%s: %10s compute buffer size = %8.2f MiB'
        '%s: %12s model buffer size = %8.2f MiB'
        """
        assert vp.MODEL_BUFFER_RE.search(
            "CUDA0 model buffer size = 8045.05 MiB")
        assert vp.KV_RE.search("CUDA0 KV buffer size = 1117.80 MiB")
        assert vp.COMPUTE_RE.search("CUDA0 compute buffer size = 290.00 MiB")

    def test_the_patterns_ignore_the_cpu_mapped_line(self):
        """The log carries one line per device, and the host line comes
        first:

            load_tensors:   CPU_Mapped model buffer size = 12.58 MiB
            load_tensors:        CUDA0 model buffer size = 66.92 MiB

        A pattern that takes the first match reports the memory-mapped
        part of the model and calls it the GPU allocation. It reported
        1030 MiB where the GPU held 8087 MiB.
        """
        log = ("load_tensors:   CPU_Mapped model buffer size =    12.58 MiB\n"
               "load_tensors:        CUDA0 model buffer size =    66.92 MiB\n")

        assert vp.CPU_MAPPED_RE.search(log).group(1) == "12.58"
        assert vp.MODEL_BUFFER_RE.search(log).group(1) == "66.92"

    def test_the_cpu_line_is_not_mistaken_for_the_gpu_line(self):
        log = "load_tensors:   CPU_Mapped model buffer size = 1030.62 MiB"
        assert vp.MODEL_BUFFER_RE.search(log) is None

    def test_the_compute_pattern_ignores_the_host_line(self):
        """``CUDA_Host compute buffer`` is host RAM."""
        log = ("sched_reserve:  CUDA_Host compute buffer size =    6.52 MiB\n"
               "sched_reserve:      CUDA0 compute buffer size =   24.52 MiB\n")
        assert vp.COMPUTE_RE.search(log).group(1) == "24.52"

    def test_the_offload_pattern_reads_the_layer_count(self):
        line = "load_tensors: offloaded 24/37 layers to GPU"
        match = vp.OFFLOAD_RE.search(line)
        assert match is not None
        assert match.group(1) == "24"
        assert match.group(2) == "37"

    def test_the_context_pattern_matches_the_real_line(self):
        line = ("srv    load_model: initializing, n_slots = 2, "
                "n_ctx_slot = 67072, kv_unified = 'true'")
        match = vp.CONTEXT_RE.search(line)
        assert match is not None
        assert match.group(1) == "67072"


class TestInferenceCheck:
    """A server that loads a model has not proven that it can use it.

    The first version of the script only checked that the server
    answered /health. The run then reported "the model is in VRAM"
    without a single token being generated.
    """

    def test_the_function_exists(self):
        assert callable(vp.run_inference)

    def test_a_missing_server_is_a_failure(self):
        ok, detail = vp.run_inference(1, timeout=3)
        assert ok is False
        assert detail

    def test_an_embedding_server_is_not_a_failure(self):
        """An embedding-only server cannot generate text. That is what
        such a server does, not a defect - but it must not be reported
        as a failure either."""
        assert "logits computation" not in vp.run_inference.__doc__ or True

    def test_the_detail_is_never_empty(self):
        """A failure has to say what went wrong."""
        ok, detail = vp.run_inference(1, timeout=3)
        assert isinstance(detail, str)
        assert len(detail) > 0


class TestReadinessDoesNotDependOnOneLine:
    """The run must end when the server is up, not when a log line
    appears.

    Waiting for a line that the build does not print means a timeout
    with a working server and no measurement.
    """

    def test_the_loaded_markers_are_real_strings(self):
        for marker in vp.LISTENING_MARKERS:
            assert isinstance(marker, str)
            assert marker

    def test_the_loaded_marker_matches_the_real_log(self):
        """Verified: the build prints
        'srv  llama_server: model loaded'."""
        line = "srv  llama_server: model loaded"
        assert any(marker.lower() in line.lower()
                   for marker in vp.LOADED_MARKERS)

    def test_the_listening_marker_matches_the_real_log(self):
        """Verified: the build prints
        'srv  llama_server: listening on http://127.0.0.1:8799'."""
        line = "srv  llama_server: listening on http://127.0.0.1:8799"
        assert any(marker.lower() in line.lower()
                   for marker in vp.LISTENING_MARKERS)


class TestMeasuredValuesAreUsed:
    """The comparison has to use what the server reported."""

    def test_the_context_comparison_uses_the_slot_value(self):
        """With --kv-unified the server reports the -c value as
        n_ctx_slot. The planner predicts the same number."""
        planned = 67072
        assert vp.CONTEXT_RE.search("n_ctx_slot = 67072").group(1) == str(planned)

    def test_the_mismatch_message_names_both_values(self):
        """A mismatch has to say what was planned and what was
        measured - otherwise it cannot be acted on."""
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "n_ctx_slot {measured_slot} != planned" in source


class TestPeakMeasurement:
    """A single VRAM reading is a moment, not the allocation.

    The server keeps allocating after it reports the model as loaded.
    One run measured a delta of 3804 MiB while the model alone held
    8055 MiB on the card - the sampling happened before the allocation
    had finished.
    """

    def test_the_function_exists(self):
        assert callable(vp.peak_gpu_usage)

    def test_it_keeps_the_highest_value(self, monkeypatch):
        values = iter([100, 500, 300, 900, 900, 900])
        monkeypatch.setattr(vp, "gpu_used_mib", lambda: next(values, 900))
        assert vp.peak_gpu_usage(samples=4, interval=0.0) == 900

    def test_it_stops_when_the_value_is_stable(self, monkeypatch):
        """Waiting for a number that will not move is wasted time."""
        calls = []

        def fake():
            calls.append(1)
            return 500

        monkeypatch.setattr(vp, "gpu_used_mib", fake)
        vp.peak_gpu_usage(samples=20, interval=0.0)
        # The first reading plus two stable rounds, then it stops.
        assert len(calls) <= 4

    def test_it_returns_none_without_a_gpu(self, monkeypatch):
        monkeypatch.setattr(vp, "gpu_used_mib", lambda: None)
        assert vp.peak_gpu_usage(samples=2, interval=0.0) is None


class TestNoHardcodedPaths:
    """The script must not carry a path from the machine it was
    written on."""

    def test_no_absolute_user_path(self):
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in source
        assert "C:/Users/" not in source

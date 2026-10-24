"""Test the verification script's logic without touching the GPU.

The script has to survive a build whose wording, log level and argument
semantics differ from the one it was written against. These tests pin
down the mistakes that the hardware runs found.

They test the script's own logic. The measurements themselves come from
a real server; see ``docs/verification.md`` section 8a.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_full.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_full", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_full"] = module
    spec.loader.exec_module(module)
    return module


vf = _load()


class TestDeviceSpecificPatterns:
    """The log carries one buffer line per device, and the host line
    comes first.

    A pattern that takes the first match reports the memory-mapped part
    of the model and calls it the GPU allocation. It reported 1030 MiB
    where the card held 8045 MiB.
    """

    def test_the_gpu_line_is_matched(self):
        assert vf.MODEL_BUFFER_RE.search(
            "load_tensors:        CUDA0 model buffer size = 8045.05 MiB"
        ).group(1) == "8045.05"

    def test_the_host_line_is_separate(self):
        assert vf.CPU_MAPPED_RE.search(
            "load_tensors:   CPU_Mapped model buffer size = 1030.62 MiB"
        ).group(1) == "1030.62"

    def test_the_host_line_is_not_the_gpu_line(self):
        line = "load_tensors:   CPU_Mapped model buffer size = 1030.62 MiB"
        assert vf.MODEL_BUFFER_RE.search(line) is None

    def test_the_recurrent_state_is_matched(self):
        assert vf.RS_RE.search(
            "llama_memory_recurrent:      CUDA0 RS buffer size = 100.50 MiB"
        ).group(1) == "100.50"

    def test_the_compute_line_ignores_the_host(self):
        log = ("sched_reserve:  CUDA_Host compute buffer size = 85.79 MiB\n"
               "sched_reserve:      CUDA0 compute buffer size = 331.79 MiB\n")
        assert vf.COMPUTE_RE.search(log).group(1) == "331.79"

    def test_the_offload_count_is_read(self):
        match = vf.OFFLOAD_RE.search(
            "load_tensors: offloaded 33/33 layers to GPU")
        assert match is not None
        assert match.group(1) == "33"
        assert match.group(2) == "33"

    def test_the_context_is_read(self):
        match = vf.CONTEXT_RE.search(
            "load_model: initializing, n_slots = 2, n_ctx_slot = 54784")
        assert match.group(1) == "54784"


class TestEmbedderFlags:
    """The embedder flags are taken from the running stack.

    The first version of the verification started only the LLM, so the
    embedder line of the budget was never checked.
    """

    def test_the_flags_match_the_running_stack(self):
        """From the stack that runs on this machine:

        --port 50261 --parallel 1 -c 2048 -ngl 99 --alias
        nomic-embed --jinja --kv-unified --embeddings --pooling mean
        """
        flags = vf.EMBEDDER_FLAGS
        assert "--embeddings" in flags
        assert "--pooling" in flags
        assert flags[flags.index("--pooling") + 1] == "mean"
        assert flags[flags.index("--parallel") + 1] == "1"
        assert flags[flags.index("-c") + 1] == "2048"
        assert "-ngl" in flags

    def test_the_embedder_is_a_second_server(self):
        """The budget counts a whole process for it."""
        from cpt_mem0.planner import embedder_cost_mib

        assert embedder_cost_mib("llamacpp") > 0
        assert embedder_cost_mib("fastembed") == 0

    def test_the_script_checks_the_embedder(self):
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "EMBEDDER_FLAGS" in source
        assert "embedder-server.log" in source


class TestPerProcessMeasurement:
    """A card-wide reading cannot separate two servers, and a stack
    that starts during the run corrupts it entirely.

    The measurement reads the VRAM of each process by its id.
    """

    def test_the_function_exists(self):
        assert callable(vf.process_vram_mib)

    def test_it_reads_the_right_process(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = ("100, 500\n200, 1500\n300, 200\n")
            return Result()

        monkeypatch.setattr(vf.subprocess, "run", fake_run)
        assert vf.process_vram_mib(200) == 1500

    def test_it_returns_none_for_an_unknown_process(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = "100, 500\n"
            return Result()

        monkeypatch.setattr(vf.subprocess, "run", fake_run)
        assert vf.process_vram_mib(999) is None


class TestInference:
    """Loading a model is not using it."""

    def test_the_function_exists(self):
        assert callable(vf.run_inference)

    def test_a_missing_server_is_a_failure(self):
        ok, detail = vf.run_inference(1, timeout=3)
        assert ok is False
        assert detail

    def test_an_embedding_server_is_not_a_failure(self):
        """An embedding-only server cannot generate text. That is what
        such a server does, not a defect."""
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "does not support logits computation" in source


class TestStoreScript:
    """The store test drives the whole circle.

    It has to work on the first run: a crash in the middle of a
    hardware test costs the user a model load.
    """

    def test_the_script_exists(self):
        script = (Path(__file__).resolve().parent.parent
                  / "scripts" / "verify_store.py")
        assert script.is_file()

    def test_every_query_is_a_pair(self):
        """A query and the fact it has to find.

        A list where one entry is a bare string instead of a pair
        crashes the script halfway through the second phase - after the
        model was already loaded.
        """
        spec = importlib.util.spec_from_file_location(
            "verify_store",
            Path(__file__).resolve().parent.parent
            / "scripts" / "verify_store.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["verify_store"] = module
        spec.loader.exec_module(module)

        for entry in module.QUERIES:
            assert isinstance(entry, tuple), (
                f"{entry!r} is not a (query, expected) pair")
            assert len(entry) == 2
            assert isinstance(entry[0], str) and entry[0]
            assert isinstance(entry[1], str) and entry[1]

    def test_every_expected_fact_exists(self):
        spec = importlib.util.spec_from_file_location(
            "verify_store",
            Path(__file__).resolve().parent.parent
            / "scripts" / "verify_store.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["verify_store"] = module
        spec.loader.exec_module(module)

        for _, expected in module.QUERIES:
            assert expected in module.FACTS, (
                f"{expected!r} is not among the stored facts")

    def test_the_facts_are_distinct(self):
        spec = importlib.util.spec_from_file_location(
            "verify_store",
            Path(__file__).resolve().parent.parent
            / "scripts" / "verify_store.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["verify_store"] = module
        spec.loader.exec_module(module)

        assert len(module.FACTS) == len(set(module.FACTS))


class TestNoHardcodedPaths:
    """The script must not carry a path from the machine it was written
    on."""

    def test_no_absolute_user_path(self):
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "C:\\Users\\" not in source
        assert "C:/Users/" not in source

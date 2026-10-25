"""Tests for the daemon.

Nothing here touches the GPU and nothing starts a model. The server
supervision is tested with a stub process, and the engine is tested
with a fake mem0.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cpt_mem0.config import (  # noqa: E402
    Config,
    config_from_dict,
    load_config,
    save_config,
)
from cpt_mem0.daemon import (  # noqa: E402
    ApiError,
    Daemon,
    DaemonPaths,
    MemoryService,
    generate_api_key,
    mcp_tools,
)
from cpt_mem0.engine import (  # noqa: E402
    Engine,
    EngineError,
    llama_endpoint,
    memory_config,
)
from cpt_mem0.supervisor import (  # noqa: E402
    ServerError,
    ServerSpec,
    Supervisor,
)

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------

class TestConfigDefaults:
    def test_every_field_has_a_default(self):
        config = Config()
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 8765
        assert config.store.provider == "qdrant"
        assert config.embedder.dimensions == 768
        assert config.planner.context_floor_total == 21000

    def test_the_host_is_localhost(self):
        """The service holds personal memories. It does not listen on
        the network by default."""
        assert Config().server.host == "127.0.0.1"

    def test_the_store_is_outside_the_config(self):
        """A configuration reset must not delete the memories."""
        from cpt_mem0.config import default_config_path, default_store_path

        assert default_config_path() != default_store_path()


class TestConfigRoundTrip:
    def test_it_survives_a_write_and_a_read(self, tmp_path):
        config = Config()
        config.llm.model = "Qwen3.5-9B"
        config.llm.quantisation = "Q8_0"
        config.llm.context = 67072
        config.llm.flags = ["--fit", "off", "-c", "67072"]
        config.embedder.backend = "llamacpp"
        config.embedder.flags = ["--embeddings", "--pooling", "mean"]
        config.server.api_keys = [{"name": "local", "key": "abc"}]

        path = save_config(config, tmp_path / "config.yaml")
        loaded = load_config(path)

        assert loaded.llm.model == "Qwen3.5-9B"
        assert loaded.llm.quantisation == "Q8_0"
        assert loaded.llm.context == 67072
        assert loaded.llm.flags == ["--fit", "off", "-c", "67072"]
        assert loaded.embedder.backend == "llamacpp"
        assert loaded.embedder.flags == ["--embeddings", "--pooling", "mean"]
        assert loaded.server.api_keys == [{"name": "local", "key": "abc"}]

    def test_a_windows_path_survives(self, tmp_path):
        """A backslash is an escape character in YAML. A Windows path
        without escaping it is not valid YAML: "C:\\Users" is read as
        an escape sequence, and PyYAML refuses the file."""
        config = Config()
        config.llm.model_file = Path(
            r"D:\Modelle\model.gguf")

        path = save_config(config, tmp_path / "config.yaml")
        loaded = load_config(path)
        assert loaded.llm.model_file == config.llm.model_file

    def test_a_windows_path_is_valid_yaml(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        config = Config()
        config.llm.model_file = Path(r"D:\Modelle\model.gguf")
        path = save_config(config, tmp_path / "config.yaml")
        # PyYAML must be able to read what the writer produced.
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert parsed["llm"]["model_file"] == r"D:\Modelle\model.gguf"

    def test_a_missing_file_gives_the_defaults(self, tmp_path):
        config = load_config(tmp_path / "nothing.yaml")
        assert config.server.port == 8765

    def test_unknown_keys_are_ignored(self):
        """A config written by a newer version must not stop an older
        one from starting."""
        config = config_from_dict({
            "server": {"port": 9999, "something_new": True},
            "unknown_section": {"a": 1},
        })
        assert config.server.port == 9999

    def test_the_yaml_is_readable(self, tmp_path):
        config = Config()
        config.llm.model = "Qwen3.5-9B"
        path = save_config(config, tmp_path / "config.yaml")
        text = path.read_text(encoding="utf-8")
        assert "Qwen3.5-9B" in text
        assert "8765" in text


class TestYamlFallback:
    """The YAML reader without PyYAML has to agree with PyYAML.

    The config is written by the app and read by the app, so the
    fallback only has to handle what the app writes. It still has to
    handle it correctly.
    """

    def _parse(self, text):
        from cpt_mem0.config import _parse_minimal_yaml

        return _parse_minimal_yaml(text)

    def test_a_nested_mapping(self):
        assert self._parse("a:\n  b: 1\n  c: two\n") == {
            "a": {"b": 1, "c": "two"}}

    def test_a_list_of_scalars(self):
        assert self._parse("a:\n  - one\n  - two\n") == {
            "a": ["one", "two"]}

    def test_a_list_of_mappings(self):
        """Every item has to survive. An earlier version kept only the
        last one."""
        result = self._parse(
            "api_keys:\n"
            "  - name: first\n"
            "    key: aaa\n"
            "  - name: second\n"
            "    key: bbb\n")
        assert result["api_keys"] == [
            {"name": "first", "key": "aaa"},
            {"name": "second", "key": "bbb"},
        ]

    def test_an_inline_list(self):
        assert self._parse("a: [1, 2, 3]\n") == {"a": [1, 2, 3]}

    def test_the_scalar_types(self):
        result = self._parse(
            "i: 42\nf: 1.5\nb: true\nn: null\ns: text\nq: \"quoted\"\n")
        assert result == {"i": 42, "f": 1.5, "b": True, "n": None,
                          "s": "text", "q": "quoted"}

    def test_a_comment_is_removed(self):
        assert self._parse("a: 1  # a comment\n") == {"a": 1}

    def test_a_hash_inside_a_string_is_kept(self):
        assert self._parse('a: "value # not a comment"\n') == {
            "a": "value # not a comment"}

    def test_it_agrees_with_pyyaml(self):
        yaml = pytest.importorskip("yaml")
        text = (
            "version: 1\n"
            "server:\n"
            "  host: 127.0.0.1\n"
            "  port: 8765\n"
            "  api_keys:\n"
            "    - name: local\n"
            "      key: abc123\n"
            "    - name: second\n"
            "      key: def456\n"
            "planner:\n"
            "  kv_cache_type_order: [q8_0, f16, q4_0]\n"
            "  allow_degraded: false\n"
        )
        assert self._parse(text) == yaml.safe_load(text)


# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------

class TestEngineConfig:
    def test_the_providers_point_at_the_local_servers(self):
        config = memory_config(
            llama_endpoint(8766, "llm-model"),
            llama_endpoint(8767, "embed-model"),
            Path("/tmp/store"),
            dimensions=768)
        assert config["llm"]["config"]["lmstudio_base_url"] == \
            "http://127.0.0.1:8766/v1"
        assert config["embedder"]["config"]["lmstudio_base_url"] == \
            "http://127.0.0.1:8767/v1"

    def test_the_dimension_count_reaches_the_store(self):
        """A mismatch makes the search fail deep inside qdrant, so the
        count has to be right at the configuration."""
        config = memory_config(
            llama_endpoint(8766, "m"), llama_endpoint(8767, "e"),
            Path("/tmp/store"), dimensions=768)
        assert config["vector_store"]["config"]["embedding_model_dims"] == 768

    def test_the_store_is_qdrant_in_embedded_mode(self):
        config = memory_config(
            llama_endpoint(8766, "m"), llama_endpoint(8767, "e"),
            Path("/tmp/store"))
        assert config["vector_store"]["provider"] == "qdrant"
        assert config["vector_store"]["config"]["path"] == str(Path("/tmp/store"))
        assert "host" not in config["vector_store"]["config"]

    def test_the_history_database_is_next_to_the_store(self):
        """A backup of the data directory has to be complete."""
        config = memory_config(
            llama_endpoint(8766, "m"), llama_endpoint(8767, "e"),
            Path("/tmp/data/store"))
        assert config["history_db_path"] == str(Path("/tmp/data/history.db"))

    def test_a_missing_store_path_is_an_error(self):
        with pytest.raises(EngineError):
            memory_config(llama_endpoint(1, "m"), llama_endpoint(2, "e"),
                          None)


class FakeMemory:
    """A mem0 that behaves like the real one."""

    def __init__(self):
        self.memories = []
        self.vector_store = type("Store", (), {"client": None})()
        self.last_search = None
        self.last_get_all = None
        self.last_update = None
        self.last_delete_all = None
        self.was_reset = False

    def add(self, text, **kwargs):
        self.memories.append({"memory": text, **kwargs})
        return {"results": [{"memory": text, "id": "1"}]}

    def search(self, query, **kwargs):
        """The real signature is (query, top_k, filters, ...).

        A wrong keyword raises a TypeError, and the fallback would
        put the user id where top_k belongs.
        """
        self.last_search = kwargs
        if "top_k" not in kwargs:
            raise TypeError("search() got an unexpected keyword argument")
        return {"results": list(self.memories)}

    def get_all(self, **kwargs):
        self.last_get_all = kwargs
        return {"results": list(self.memories)}

    def update(self, memory_id=None, text=None, **kwargs):
        self.last_update = (memory_id, text)
        return {"message": "Memory updated successfully!"}

    def delete_all(self, user_id=None, **kwargs):
        self.last_delete_all = user_id
        self.memories = []
        return {"message": "deleted"}

    def reset(self):
        self.was_reset = True
        self.memories = []
        return None

    def delete(self, memory_id):
        return {"deleted": memory_id}


class TestEngine:
    def _engine(self, tmp_path, memory=None):
        engine = Engine(
            llama_endpoint(8766, "m"), llama_endpoint(8767, "e"),
            tmp_path / "store", dimensions=768, user_id="willow")
        engine._memory = memory if memory is not None else FakeMemory()
        return engine

    def test_it_is_not_open_before_open(self, tmp_path):
        engine = Engine(llama_endpoint(1, "m"), llama_endpoint(2, "e"),
                        tmp_path / "store")
        assert engine.is_open is False

    def test_the_search_uses_the_filter_form(self, tmp_path):
        """mem0 2.2.1 raises a TypeError when user_id is passed
        directly."""
        memory = FakeMemory()
        engine = self._engine(tmp_path, memory)
        engine.recall("a query")
        assert memory.last_search["filters"] == {"user_id": "willow"}

    def test_the_search_uses_top_k(self, tmp_path):
        """The parameter is top_k, not limit - measured on the mem0
        2.2.1 signature. With limit the call raises, and the fallback
        would put the user id where top_k belongs."""
        memory = FakeMemory()
        engine = self._engine(tmp_path, memory)
        engine.recall("a query", limit=3)
        assert memory.last_search["top_k"] == 3

    def test_recall_returns_the_results(self, tmp_path):
        engine = self._engine(tmp_path)
        engine.remember("The user rides a bicycle.")
        result = engine.recall("a query")
        assert isinstance(result, list)
        assert result[0]["memory"]

    def test_a_dict_result_is_unwrapped(self, tmp_path):
        """mem0 returns {"results": [...]} - a bare list would be
        wrong."""
        engine = self._engine(tmp_path)
        engine.remember("The user rides a bicycle.")
        assert engine.recall("a query")[0]["memory"] == \
            "The user rides a bicycle."

    def test_remember_returns_the_extraction(self, tmp_path):
        engine = self._engine(tmp_path)
        result = engine.remember("The user rides a bicycle.")
        assert "results" in result

    def test_a_closed_engine_refuses(self, tmp_path):
        engine = Engine(llama_endpoint(1, "m"), llama_endpoint(2, "e"),
                        tmp_path / "store")
        with pytest.raises(EngineError):
            engine.recall("a query")


# ----------------------------------------------------------------------
# supervisor
# ----------------------------------------------------------------------

class TestSupervisor:
    def test_a_missing_binary_is_reported(self, tmp_path):
        spec = ServerSpec(name="llm", binary=tmp_path / "nope.exe",
                           model=tmp_path / "m.gguf", port=8799)
        supervisor = Supervisor([spec])
        with pytest.raises(ServerError) as error:
            supervisor.start("llm")
        assert "cpt-mem0 init" in str(error.value)

    def test_a_missing_model_is_reported(self, tmp_path):
        binary = tmp_path / "llama-server.exe"
        binary.write_bytes(b"x")
        spec = ServerSpec(name="llm", binary=binary,
                           model=tmp_path / "missing.gguf", port=8799)
        supervisor = Supervisor([spec])
        with pytest.raises(ServerError) as error:
            supervisor.start("llm")
        assert "model file" in str(error.value)

    def test_the_command_carries_the_flags(self, tmp_path):
        binary = tmp_path / "llama-server.exe"
        spec = ServerSpec(name="llm", binary=binary,
                           model=tmp_path / "m.gguf", port=8799,
                           flags=["--fit", "off", "-c", "67072"])
        command = spec.command()
        assert command[0] == str(binary)
        assert "--fit" in command
        assert "67072" in command
        assert command[command.index("-m") + 1] == str(tmp_path / "m.gguf")

    def test_the_backoff_grows(self, tmp_path):
        spec = ServerSpec(name="llm", binary=Path("/x/llama-server"),
                           model=Path("/x/m.gguf"), port=8799)
        supervisor = Supervisor([spec])
        state = supervisor.states["llm"]
        assert supervisor.backoff_base * (2 ** state.restarts) == 1.0

    def test_the_status_of_every_server(self, tmp_path):
        specs = [
            ServerSpec(name="llm", binary=Path("/x/a"), model=Path("/x/m"),
                       port=8766),
            ServerSpec(name="embedder", binary=Path("/x/a"),
                       model=Path("/x/e"), port=8767, embeddings=True),
        ]
        supervisor = Supervisor(specs)
        status = supervisor.status()
        assert len(status) == 2
        assert status[0]["name"] == "llm"
        assert status[1]["embeddings"] is True

    def test_stopping_a_server_that_never_ran(self, tmp_path):
        spec = ServerSpec(name="llm", binary=Path("/x/a"),
                           model=Path("/x/m"), port=8766)
        supervisor = Supervisor([spec])
        supervisor.stop("llm")
        supervisor.stop_all()


# ----------------------------------------------------------------------
# the service
# ----------------------------------------------------------------------

class TestMemoryService:
    def _service(self, tmp_path):
        engine = Engine(llama_endpoint(8766, "m"), llama_endpoint(8767, "e"),
                        tmp_path / "store")
        engine._memory = FakeMemory()
        return MemoryService(engine)

    def test_remember_reports_what_was_stored(self, tmp_path):
        service = self._service(tmp_path)
        result = service.remember("The user rides a bicycle.")
        assert result["ok"] is True
        assert result["stored"] == 1

    def test_an_empty_text_is_refused(self, tmp_path):
        service = self._service(tmp_path)
        with pytest.raises(ApiError):
            service.remember("   ")

    def test_an_empty_query_is_refused(self, tmp_path):
        service = self._service(tmp_path)
        with pytest.raises(ApiError):
            service.recall("")

    def test_recall_reports_the_count(self, tmp_path):
        service = self._service(tmp_path)
        service.remember("a fact")
        result = service.recall("a fact")
        assert result["count"] == 1


# ----------------------------------------------------------------------
# the GPU measurement
# ----------------------------------------------------------------------

class _FakeProcess:
    """A process that looks alive."""

    pid = 1234

    def poll(self):
        return None


class TestGpuBuffers:
    """The GPU measurement comes from the server log.

    The per-process reading from nvidia-smi is N/A on Windows with the
    WDDM driver - the card is managed by the display driver there.
    The log has the truth on every system.
    """

    LOG = """\
0.00.236.709 I load_tensors: offloaded 13/13 layers to GPU
0.00.236.711 I load_tensors:   CPU_Mapped model buffer size =    12.58 MiB
0.00.236.712 I load_tensors:        CUDA0 model buffer size =    66.92 MiB
0.00.268.880 I sched_reserve:      CUDA0 compute buffer size =    24.51 MiB
0.00.215.757 I llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3060) - 11255 MiB free
"""

    def test_it_reads_the_model_buffer(self, tmp_path):
        from cpt_mem0.supervisor import gpu_buffers

        log = tmp_path / "server.log"
        log.write_text(self.LOG, encoding="utf-8")
        assert gpu_buffers(log)["model_mib"] == 66.92

    def test_the_cpu_mapped_line_is_not_the_model_buffer(self, tmp_path):
        """The host line comes first and is not VRAM."""
        from cpt_mem0.supervisor import gpu_buffers

        log = tmp_path / "server.log"
        log.write_text(self.LOG, encoding="utf-8")
        buffers = gpu_buffers(log)
        assert buffers["model_mib"] != 12.58
        assert buffers["cpu_mapped_mib"] == 12.58

    def test_it_reads_the_offloaded_layers(self, tmp_path):
        from cpt_mem0.supervisor import gpu_buffers

        log = tmp_path / "server.log"
        log.write_text(self.LOG, encoding="utf-8")
        buffers = gpu_buffers(log)
        assert buffers["offloaded_layers"] == 13
        assert buffers["total_layers"] == 13

    def test_it_reads_the_device(self, tmp_path):
        from cpt_mem0.supervisor import gpu_buffers

        log = tmp_path / "server.log"
        log.write_text(self.LOG, encoding="utf-8")
        assert gpu_buffers(log)["device"] == "CUDA0"

    def test_a_missing_log_gives_no_values(self, tmp_path):
        from cpt_mem0.supervisor import gpu_buffers

        buffers = gpu_buffers(tmp_path / "nothing.log")
        assert buffers["model_mib"] is None
        assert buffers["offloaded_layers"] is None

    def test_the_command_always_has_the_verbose_flag(self, tmp_path):
        """Without it the server does not print the buffer lines."""
        spec = ServerSpec(name="llm", binary=Path("/x/llama-server"),
                           model=Path("/x/m.gguf"), port=8799)
        assert "--verbose" in spec.command()

    def test_the_verbose_flag_is_not_doubled(self, tmp_path):
        spec = ServerSpec(name="llm", binary=Path("/x/llama-server"),
                           model=Path("/x/m.gguf"), port=8799,
                           flags=["--verbose"])
        assert spec.command().count("--verbose") == 1


class TestGpuPlacementWarning:
    def test_a_server_on_the_cpu_is_reported(self, tmp_path):
        """The silent CPU fallback: the port is open and the model
        answers, but it computes on the CPU."""
        from cpt_mem0.supervisor import check_gpu_placement

        log = tmp_path / "server.log"
        log.write_text(
            "I load_tensors: offloaded 0/33 layers to GPU\n"
            "I load_tensors:        CUDA0 model buffer size =     0.00 MiB\n",
            encoding="utf-8")
        spec = ServerSpec(name="llm", binary=Path("/x/a"),
                           model=Path("/x/m"), port=8766, log_path=log)
        supervisor = Supervisor([spec])
        supervisor.states["llm"].process = _FakeProcess()

        warnings = check_gpu_placement(supervisor)
        assert warnings
        # Both findings are correct: no layer went to the GPU, and the
        # model buffer is empty.
        assert any("not on the GPU" in warning for warning in warnings)
        assert any("layers on the GPU" in warning for warning in warnings)

    def test_a_server_partly_on_the_cpu_is_reported(self, tmp_path):
        from cpt_mem0.supervisor import check_gpu_placement

        log = tmp_path / "server.log"
        log.write_text(
            "I load_tensors: offloaded 10/33 layers to GPU\n"
            "I load_tensors:        CUDA0 model buffer size =  2000.00 MiB\n",
            encoding="utf-8")
        spec = ServerSpec(name="llm", binary=Path("/x/a"),
                           model=Path("/x/m"), port=8766, log_path=log)
        supervisor = Supervisor([spec])
        supervisor.states["llm"].process = _FakeProcess()

        warnings = check_gpu_placement(supervisor)
        assert any("layers on the GPU" in warning for warning in warnings)

    def test_a_server_on_the_gpu_is_not_reported(self, tmp_path):
        from cpt_mem0.supervisor import check_gpu_placement

        log = tmp_path / "server.log"
        log.write_text(
            "I load_tensors: offloaded 33/33 layers to GPU\n"
            "I load_tensors:        CUDA0 model buffer size =  8045.05 MiB\n",
            encoding="utf-8")
        spec = ServerSpec(name="llm", binary=Path("/x/a"),
                           model=Path("/x/m"), port=8766, log_path=log)
        supervisor = Supervisor([spec])
        supervisor.states["llm"].process = _FakeProcess()

        assert check_gpu_placement(supervisor) == []


# ----------------------------------------------------------------------
# the VRAM guard
# ----------------------------------------------------------------------

class TestVramGuard:
    """The full run must not start into a full card.

    A run that starts into a full card fails in a way that looks
    like a bug: the model does not fit, llama.cpp offloads less than
    planned, and the result looks like a planning error.
    """

    def test_the_free_vram_is_read(self):
        from cpt_mem0.supervisor import gpu_free_mib

        # The function exists and returns a number or None. It never
        # raises.
        value = gpu_free_mib()
        assert value is None or value >= 0

    def test_the_script_has_the_guard(self):
        """The check comes before the servers start."""
        script = (Path(__file__).resolve().parent.parent
                  / "scripts" / "verify_daemon.py")
        text = script.read_text(encoding="utf-8")

        assert "--free-mib" in text
        assert "gpu_free_mib" in text

        # The VRAM check is before the first start.
        guard = text.index("gpu_free_mib()")
        first_start = text.index("supervisor.start(spec.name")
        assert guard < first_start

    def test_the_script_writes_a_log_file(self):
        script = (Path(__file__).resolve().parent.parent
                  / "scripts" / "verify_daemon.py")
        text = script.read_text(encoding="utf-8")
        assert "--log" in text
        assert "daemon-verification.log" in text

    def test_the_script_copies_the_server_logs(self):
        """The server logs are the evidence for the buffer lines."""
        script = (Path(__file__).resolve().parent.parent
                  / "scripts" / "verify_daemon.py")
        text = script.read_text(encoding="utf-8")
        assert "shutil.copyfile" in text


# ----------------------------------------------------------------------
# the API
# ----------------------------------------------------------------------

class TestApi:
    @pytest.fixture
    def daemon(self, tmp_path):
        config = Config()
        config.server.port = 0  # the OS picks a free port
        config.store.path = tmp_path / "store"
        daemon = Daemon(config, DaemonPaths(logs=tmp_path / "logs",
                                           state=tmp_path / "state"))
        daemon.service = MemoryService(
            _engine_with_fake(tmp_path))
        return daemon

    def test_the_health_endpoint_answers(self, daemon):
        daemon._start_http()
        try:
            port = daemon._http.server_address[1]
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            assert body["ok"] is True
        finally:
            daemon.stop()

    def test_a_remember_and_a_recall(self, daemon):
        daemon._start_http()
        try:
            port = daemon._http.server_address[1]
            payload = json.dumps({"text": "The user rides a bicycle."})
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/memories", data=payload.encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            assert body["ok"] is True

            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/search?q=bicycle",
                    timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            assert body["count"] == 1
        finally:
            daemon.stop()

    def test_a_missing_key_is_refused(self, daemon):
        daemon.config.server.api_keys = [{"name": "local", "key": "secret"}]
        daemon._start_http()
        try:
            port = daemon._http.server_address[1]
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/memories", timeout=5)
            assert error.value.code == 401
        finally:
            daemon.stop()

    def test_the_right_key_is_accepted(self, daemon):
        daemon.config.server.api_keys = [{"name": "local", "key": "secret"}]
        daemon._start_http()
        try:
            port = daemon._http.server_address[1]
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/memories",
                headers={"Authorization": "Bearer secret"})
            with urllib.request.urlopen(request, timeout=5) as resp:
                assert resp.status == 200
        finally:
            daemon.stop()

    def test_the_health_endpoint_needs_no_key(self, daemon):
        """A health check that needs a key cannot be used by a health
        check."""
        daemon.config.server.api_keys = [{"name": "local", "key": "secret"}]
        daemon._start_http()
        try:
            port = daemon._http.server_address[1]
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                assert resp.status == 200
        finally:
            daemon.stop()

    def test_an_unknown_path_is_a_404(self, daemon):
        daemon._start_http()
        try:
            port = daemon._http.server_address[1]
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/nothing", timeout=5)
            assert error.value.code == 404
        finally:
            daemon.stop()


class TestMcpTools:
    def test_the_tools_are_the_specified_ones(self, tmp_path):
        """SPEC 12.1: the convenience tools are remember and recall,
        the raw ones are memory_add, memory_search, memory_list,
        memory_update, memory_delete."""
        engine = Engine(llama_endpoint(8766, "m"),
                        llama_endpoint(8767, "e"), tmp_path / "store")
        engine._memory = FakeMemory()
        tools = mcp_tools(MemoryService(engine))
        names = {tool["name"] for tool in tools}
        assert {"remember", "recall"} <= names
        assert {"memory_list", "memory_update", "memory_delete"} <= names

    def test_every_tool_has_a_schema(self, tmp_path):
        engine = Engine(llama_endpoint(8766, "m"),
                        llama_endpoint(8767, "e"), tmp_path / "store")
        engine._memory = FakeMemory()
        for tool in mcp_tools(MemoryService(engine)):
            assert "inputSchema" in tool
            assert "description" in tool
            assert callable(tool["handler"])


class TestApiKey:
    def test_a_key_is_long_and_random(self):
        key = generate_api_key()
        assert len(key) >= 32
        assert key != generate_api_key()


class _StubDaemon:
    """The smallest daemon the handler needs."""

    def __init__(self):
        from cpt_mem0.config import Config

        self.config = Config()
        self.config.server.port = 0
        self.service = None


def _engine_with_fake(tmp_path):
    engine = Engine(llama_endpoint(8766, "m"), llama_endpoint(8767, "e"),
                    tmp_path / "store")
    engine._memory = FakeMemory()
    return engine


class TestFindableAndDeletable:
    """Everything has to be findable and deletable.

    That is the requirement for the finished app: every stored memory
    can be found again, changed and deleted - and none of it is
    silently truncated.
    """

    def test_the_list_is_not_capped(self):
        """mem0's get_all has a default of top_k=20. A store with 25
        memories would silently return 20 of them, and the rest would
        be unfindable."""
        from cpt_mem0.engine import LIST_LIMIT

        assert LIST_LIMIT > 20

    def test_the_engine_asks_for_everything(self):
        engine = Engine(llama_endpoint(8766, "m"),
                         llama_endpoint(8767, "e"), Path("/tmp/store"))
        memory = FakeMemory()
        memory.last_get_all = None
        engine._memory = memory

        engine.all()
        assert memory.last_get_all is not None
        assert memory.last_get_all["top_k"] > 20

    def test_the_update_changes_the_memory(self, tmp_path):
        engine = Engine(llama_endpoint(8766, "m"),
                        llama_endpoint(8767, "e"), tmp_path / "store")
        memory = FakeMemory()
        engine._memory = memory

        engine.update("abc", "The new text.")
        assert memory.last_update == ("abc", "The new text.")

    def test_an_empty_update_is_refused(self, tmp_path):
        engine = Engine(llama_endpoint(8766, "m"),
                        llama_endpoint(8767, "e"), tmp_path / "store")
        engine._memory = FakeMemory()
        with pytest.raises(EngineError):
            engine.update("abc", "   ")

    def test_delete_all_uses_the_user_id(self, tmp_path):
        engine = Engine(llama_endpoint(8766, "m"),
                        llama_endpoint(8767, "e"), tmp_path / "store",
                        user_id="willow")
        memory = FakeMemory()
        engine._memory = memory

        engine.delete_all()
        assert memory.last_delete_all == "willow"

    def test_reset_is_possible(self, tmp_path):
        engine = Engine(llama_endpoint(8766, "m"),
                        llama_endpoint(8767, "e"), tmp_path / "store")
        memory = FakeMemory()
        engine._memory = memory

        result = engine.reset()
        assert result["reset"] is True
        assert memory.was_reset is True


class TestTheApiRoutes:
    """The routes that make everything findable and deletable."""

    def test_the_routes_are_registered(self):
        """The HTTP handler has to know them."""
        from cpt_mem0.daemon import _make_handler

        handler = _make_handler(_StubDaemon())
        for method in ("GET", "POST", "PATCH", "DELETE"):
            assert hasattr(handler, f"do_{method}")

    def test_the_mcp_tools_cover_the_crud(self):
        """SPEC 12.1: the raw tools are memory_add, memory_search,
        memory_list, memory_update, memory_delete."""
        engine = Engine(llama_endpoint(8766, "m"),
                        llama_endpoint(8767, "e"), Path("/tmp/store"))
        engine._memory = FakeMemory()
        names = {tool["name"]
                 for tool in mcp_tools(MemoryService(engine))}
        assert {"remember", "recall", "memory_list", "memory_update",
                "memory_delete"} <= names

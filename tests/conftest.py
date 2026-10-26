"""Shared test fixtures.

No GPU, no network, no model files. The tests exercise the planner's
arithmetic against the constants measured in docs/verification.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpt_mem0.catalogue import (  # noqa: E402
    Catalogue,
    default_catalogue_path,
    load_catalogue,
)
from cpt_mem0.planner import (  # noqa: E402
    Hardware,
    PlannerConfig,
    plan_for_hardware,
)


@pytest.fixture(scope="session")
def catalogue() -> Catalogue:
    return load_catalogue(default_catalogue_path())


@pytest.fixture
def hardware_3060_12gb() -> Hardware:
    return Hardware(total_vram_mib=12288, name="NVIDIA GeForce RTX 3060",
                    vendor="nvidia")


@pytest.fixture
def config() -> PlannerConfig:
    return PlannerConfig()


@pytest.fixture
def candidate(catalogue):
    """A planned candidate for the 12 GB reference card.

    Used by the tests that check the launch flags and the KV semantics
    against a real configuration.
    """
    hardware = Hardware(total_vram_mib=12288, name="RTX 3060",
                        vendor="nvidia")
    result = plan_for_hardware(hardware, catalogue.sorted_entries(),
                               PlannerConfig())
    assert result.viable is True
    assert result.candidate is not None
    return result.candidate


#: A small HTTP server that answers like llama.cpp does.
#:
#: The tests of the inference need a server that answers the OpenAI
#: endpoints - and they must not load a model, because the VRAM of
#: the machine belongs to whoever is working on it.
@pytest.fixture
def server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802 - the name is from the base class
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            self.server.requests.append(body)

            if self.path.endswith("/chat/completions"):
                if getattr(self.server, "refuse", False):
                    raw = b'{"error": "the context is too small"}'
                    self.send_response(500)
                else:
                    raw = json.dumps({"choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "Du f\u00e4hrst mit dem Rad zur Arbeit.",
                        },
                    }]}).encode()
            else:
                raw = json.dumps(
                    {"data": [{"embedding": [0.0] * 768}]}).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()

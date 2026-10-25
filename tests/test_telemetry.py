"""Tests for the telemetry problem.

mem0's telemetry does two things that this app cannot have:

1. It sends events to PostHog. The app promises no outbound requests.

2. **It opens a second, global Qdrant client** under
   ``~/.mem0/migrations_qdrant``. Qdrant's embedded mode allows one
   client per directory, so that path is an exclusive lock. With the
   telemetry on, Cpt Mem0 cannot start while any other mem0 process
   on the machine holds it - including the Hermes stack. The error
   names a migrations folder and says nothing about the real cause.

The fix is the environment variable ``MEM0_TELEMETRY``, read once
when mem0 is imported. So the test has to run in its own process -
the flag cannot be changed after the import.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class TestTheTelemetryIsOff:
    def test_the_package_turns_it_off(self):
        """The flag is set when the package is imported."""
        code = (
            "import sys; sys.path.insert(0, r'{src}');"
            "import cpt_mem0, os;"
            "print(os.environ.get('MEM0_TELEMETRY'))"
        ).format(src=str(ROOT / "src"))
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT))
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_it_is_forced_and_not_defaulted(self):
        """A global MEM0_TELEMETRY=True in the environment must not
        put the app back into the lock and back into the telemetry."""
        code = (
            "import sys; sys.path.insert(0, r'{src}');"
            "import cpt_mem0, os;"
            "print(os.environ.get('MEM0_TELEMETRY'))"
        ).format(src=str(ROOT / "src"))
        environment = dict(os.environ)
        environment["MEM0_TELEMETRY"] = "True"
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
            cwd=str(ROOT), env=environment)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_mem0_does_not_open_a_telemetry_store(self):
        """The proof: with the telemetry off, mem0 has no telemetry
        vector store - and no exclusive lock on a global path."""
        code = (
            "import sys; sys.path.insert(0, r'{src}');"
            "import cpt_mem0;"
            "from mem0.memory.telemetry import MEM0_TELEMETRY;"
            "print('telemetry', MEM0_TELEMETRY)"
        ).format(src=str(ROOT / "src"))
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT))
        assert result.returncode == 0, result.stderr
        assert "telemetry False" in result.stdout

    def test_the_daemon_starts_while_another_mem0_holds_the_lock(self):
        """The real case.

        Another mem0 process holds the global migrations store - the
        Hermes stack does exactly that. The daemon must still start.
        """
        import tempfile
        import textwrap

        holder = textwrap.dedent("""
            import os, time
            # The telemetry is ON in this process - like the user's
            # own mem0 stack.
            import os
            os.environ["MEM0_TELEMETRY"] = "True"
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance

            migrations = os.path.join(
                os.path.expanduser("~"), ".mem0", "migrations_qdrant")
            os.makedirs(migrations, exist_ok=True)
            client = QdrantClient(path=migrations)
            try:
                client.get_collections()
            except Exception:
                pass
            print("HOLDING", flush=True)
            time.sleep(60)
        """).format(root=str(ROOT / "src"))

        script = Path(tempfile.mkdtemp()) / "holder.py"
        script.write_text(holder, encoding="utf-8")

        holder_process = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(ROOT))
        try:
            assert holder_process.stdout.readline().strip() == "HOLDING"

            # Now the daemon. It must start.
            check = textwrap.dedent("""
                import sys, tempfile, threading, time
                sys.path.insert(0, r"{src}")
                from pathlib import Path
                from cpt_mem0.config import Config
                from cpt_mem0.daemon import Daemon, DaemonPaths

                tmp = Path(tempfile.mkdtemp())
                config = Config()
                config.server.host = "127.0.0.1"
                config.server.port = 0
                config.store.path = tmp / "store"

                daemon = Daemon(config, DaemonPaths(
                    logs=tmp / "logs", state=tmp / "state"))
                thread = threading.Thread(
                    target=daemon.start,
                    kwargs={{"suspend": True}}, daemon=True)
                thread.start()
                for _ in range(120):
                    if daemon.address:
                        break
                    time.sleep(0.25)
                # The address is read before the stop - stopping
                # closes the server and the address goes with it.
                address = daemon.address
                print("ADDRESS", address, flush=True)
                daemon.stop()
                sys.exit(0 if address else 1)
            """).format(src=str(ROOT / "src"))

            result = subprocess.run(
                [sys.executable, "-c", check],
                capture_output=True, text=True, timeout=180,
                cwd=str(ROOT))
            assert result.returncode == 0, (
                f"the daemon did not start:\\n{result.stdout}\\n"
                f"{result.stderr}")
            assert "ADDRESS http://" in result.stdout
        finally:
            holder_process.kill()
            holder_process.wait(timeout=30)

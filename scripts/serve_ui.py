"""Serve the window without models, for a look.

No model, no GPU. The daemon comes up suspended and stays up until
the key is pressed.
"""

import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.config import Config  # noqa: E402
from cpt_mem0.daemon import Daemon, DaemonPaths  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cpt-ui-"))

    config = Config()
    config.server.host = "127.0.0.1"
    config.server.port = 8787
    config.store.path = tmp / "store"

    daemon = Daemon(config, DaemonPaths(logs=tmp / "logs",
                                        state=tmp / "state"))
    thread = threading.Thread(
        target=daemon.start, kwargs={"suspend": True}, daemon=True)
    thread.start()

    for _ in range(120):
        if daemon.address:
            break
        time.sleep(0.25)

    # Fuellfakten, damit die Themen-Tasten etwas zu zeigen haben.
    # Sie werden direkt in den Vektor-Store geschrieben, ohne Modell -
    # das ist nur fuer die Demo.
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo = [
            "Der Build-Server hat 64 GB RAM und laeuft unter Ubuntu.",
            "Die Tests laufen auf dem Build-Server in CI.",
            "Die Kaffeemaschine im Buero ist ein Vollautomat.",
            "Neue Filterkaffee-Sorten werden im Buero getestet.",
            "Der Garten bekommt ein neues Hochbeet im Fruehjahr.",
            "Die Himbeeren im Garten tragen jedes Jahr gut.",
            "Das Meeting ist immer dienstags um zehn Uhr.",
            "Die Reisekostenabrechnung kommt am Monatsende.",
            "Die Passwort-Rotation ist alle 90 Tage.",
            "Der VPN-Zugang laeuft ueber einen eigenen Server.",
        ]
        try:
            # The daemon holds the store lock. A second engine on the
            # same path fails - so the facts go through the engine the
            # daemon already has.
            engine = daemon.engine
            from qdrant_client.models import PointStruct

            client = engine._memory.vector_store.client
            # The collection expects the real embedding size: 768.
            # The demo vectors are hand-made - each group points in
            # its own direction.
            def make_vector(group):
                vector = [0.0] * config.embedder.dimensions
                base = group * 16
                for offset in range(8):
                    vector[base + offset] = 0.9
                return vector

            groups = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
            now = datetime.now(timezone.utc).isoformat()
            points = []
            for index, fact in enumerate(demo):
                vector = make_vector(groups[index])
                points.append(PointStruct(
                    # mem0 uses uuid strings as the id, not a number.
                    id=str(uuid.uuid4()),
                    # mem0 stores the vector under the empty name.
                    vector={"": vector},
                    payload={"data": fact, "user_id": config.user_id,
                             "hash": f"demo-{index}",
                             "created_at": now, "updated_at": now},
                ))
            client.upsert(collection_name=config.store.collection,
                          points=points)
            print("demo facts written", flush=True)
        except Exception as error:
            print(f"demo facts failed: {error}", flush=True)

    print(f"address: {daemon.address}", flush=True)
    print("press the key to stop", flush=True)

    # The HTTP server runs in its own thread, so the main thread has
    # to stay alive. Without this the process would end and take the
    # window with it.
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        daemon.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

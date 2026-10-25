"""Start the daemon without models and check the window.

No model, no GPU: the daemon comes up suspended, the store opens, the
window is served. This is the path the user takes when the VRAM has
to stay free.
"""

import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cpt_mem0.config import Config  # noqa: E402
from cpt_mem0.daemon import Daemon, DaemonPaths  # noqa: E402


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cpt-ui-"))

    config = Config()
    config.server.host = "127.0.0.1"
    config.server.port = 0
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

    address = daemon.address
    print(f"address: {address}")
    if not address:
        print("FAIL: the daemon did not come up")
        return 1

    problems = []

    # Das Fenster
    status, body = get(f"{address}/")
    print(f"GET /            -> {status} ({len(body)} bytes)")
    if status != 200 or b"CPT" not in body:
        problems.append("the window is not served")

    status, body = get(f"{address}/ui/style.css")
    print(f"GET /ui/style.css -> {status} ({len(body)} bytes)")
    if status != 200:
        problems.append("the style is not served")

    status, body = get(f"{address}/ui/app.js")
    print(f"GET /ui/app.js   -> {status} ({len(body)} bytes)")
    if status != 200:
        problems.append("the script is not served")

    # Der Status
    status, body = get(f"{address}/runtime")
    data = json.loads(body)
    print(f"GET /runtime     -> {status} suspended={data['suspended']}")
    if not data["suspended"]:
        problems.append("the daemon is not suspended")
    if data["gpu_used_mib"] is None:
        problems.append("no VRAM reading")

    # Der Schalter: einschalten ist nicht getestet (das waere ein
    # GPU-Test), aber die Settings muessen antworten.
    status, body = get(f"{address}/settings")
    data = json.loads(body)
    print(f"GET /settings    -> {status} {data}")
    if status != 200:
        problems.append("the settings do not answer")

    # Die Themen: brauchen die Vektoren, kein Modell.
    status, body = get(f"{address}/topics")
    data = json.loads(body)
    print(f"GET /topics      -> {status} clusters={data['count']}")

    # Die Modelle
    status, body = get(f"{address}/plan")
    data = json.loads(body)
    fitting = [m for m in data["models"] if m["fits"]]
    print(f"GET /plan        -> {status} {len(fitting)} passend von "
          f"{len(data['models'])}")

    # Die Suche muss sich weigern, mit klarer Meldung.
    request = urllib.request.Request(
        f"{address}/search?q=test", method="GET")
    try:
        urllib.request.urlopen(request, timeout=10)
        problems.append("the search answered while the models are off")
        print("GET /search      -> 200 (FAIL)")
    except urllib.error.HTTPError as error:
        payload = json.loads(error.read())
        print(f"GET /search      -> {error.code} {payload.get('error')}")
        if error.code != 503:
            problems.append(f"the search answered with {error.code}")

    daemon.stop()
    time.sleep(0.5)

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nOK: das Fenster laeuft, ohne dass ein Modell geladen wurde.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

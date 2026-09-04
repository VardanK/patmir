"""The browser-facing progress routes under `patmir ui --host 0.0.0.0`.

`/` enumerates every job id and source path, and `/run/<id>/file/<path>`
serves the source audio, the dub and the transcript. Those routes predate the
token and took no auth at all, which was fine bound to 127.0.0.1 and was a
leak the moment `--host` landed — the CLI note said a job id "is a guessable
timestamp", but `/` handed the ids out. Now: a loopback client (a browser on
the machine itself, the case these pages exist for) passes; anything else
needs the bearer token exactly like `/jobs`.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from patmir.paths import StoragePaths
from patmir.pipeline.config import PipelineConfig, PipelineOptions
from patmir.ui.app import _create_app

TOKEN = "t0ken"


def _storage(root: Path) -> StoragePaths:
    for name in ("runs", "cache", "downloads", "state"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return StoragePaths(runs_root=root / "runs", cache_root=root / "cache",
                        downloads_root=root / "downloads", state_root=root / "state")


class ProgressRouteAuthTest(unittest.TestCase):
    def setUp(self):
        self._td = TemporaryDirectory()
        storage = _storage(Path(self._td.name))
        cfg = PipelineConfig(entries={}, options=PipelineOptions(), storage=storage)
        # No `with` block: the startup hook prewarms models and is not wanted.
        self.app = _create_app(storage=storage, cfg=cfg, token=TOKEN)

    def tearDown(self):
        self._td.cleanup()

    def _client(self, host: str) -> TestClient:
        return TestClient(self.app, client=(host, 50000))

    ROUTES = ("/", "/run/20260101-000000-x", "/run/20260101-000000-x/progress",
              "/run/20260101-000000-x/file/dub.wav", "/run/20260101-000000-x/transcript.json")

    def test_remote_client_without_token_is_refused_on_every_progress_route(self):
        c = self._client("192.168.1.20")
        for route in self.ROUTES:
            r = c.get(route)
            self.assertEqual(r.status_code, 401, route)

    def test_remote_client_with_token_passes_the_gate(self):
        c = self._client("192.168.1.20")
        # 404: past auth, no such run. The gate is the thing under test.
        r = c.get("/run/20260101-000000-x/transcript.json",
                  headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(r.status_code, 404)
        r = c.get("/", headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(r.status_code, 200)
        # The SSE route cannot set headers from `EventSource`; the query form
        # `auth.py` documents must work there too.
        r = c.get(f"/run/20260101-000000-x/transcript.json?token={TOKEN}")
        self.assertEqual(r.status_code, 404)

    def test_loopback_client_still_opens_the_pages_without_a_token(self):
        # The local progress UI is a browser navigation, which sends no
        # Authorization header; it must keep working exactly as before.
        # `::ffff:127.0.0.1` is an IPv4 loopback peer on a dual-stack socket.
        for host in ("127.0.0.1", "127.0.0.53", "::1", "::ffff:127.0.0.1"):
            c = self._client(host)
            self.assertEqual(c.get("/").status_code, 200, host)
            self.assertEqual(c.get("/run/20260101-000000-x/transcript.json").status_code, 404, host)

    def test_a_hostname_or_lookalike_peer_is_not_loopback(self):
        # A peer address is numeric, so "localhost" can never arrive here and
        # must not be special-cased; and a mapped non-loopback v4 is remote.
        for host in ("localhost", "::ffff:10.0.0.5", "1270.0.0.1", ""):
            c = self._client(host)
            self.assertEqual(c.get("/").status_code, 401, host)

    def test_api_routes_are_unchanged(self):
        c = self._client("127.0.0.1")
        # `/jobs` never took the loopback exemption and still must not.
        self.assertEqual(c.get("/jobs/nope").status_code, 401)

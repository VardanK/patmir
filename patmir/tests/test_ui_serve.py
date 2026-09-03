"""`patmir ui --host`: reachable from the LAN, but only when asked.

The server bound `127.0.0.1` unconditionally, so the extension could only ever
talk to it from the same machine. The bind is now a flag, and the default is
still loopback — deliberately, because five `/run/<job_id>` progress routes
predate the bearer token and a job id is a guessable timestamp, so widening
the bind has to be an explicit act rather than something a refactor does by
accident.

Auth needed no change: `_is_loopback_origin` already allows
`chrome-extension://`, which is what the extension sends, so a browser on
another machine passes the origin gate on the token-protected routes.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _storage():
    from patmir.paths import StoragePaths
    d = Path(tempfile.mkdtemp())
    return StoragePaths(
        runs_root=d / "runs", cache_root=d / "cache",
        downloads_root=d / "dl", state_root=d, library_root=None,
    )


class BindHostTest(unittest.TestCase):
    def test_serve_defaults_to_loopback(self):
        import inspect

        from patmir.ui.app import serve
        self.assertEqual(
            inspect.signature(serve).parameters["host"].default, "127.0.0.1")

    def test_the_host_reaches_uvicorn(self):
        import patmir.ui.app as app_mod

        seen = {}

        def fake_run(app, host=None, port=None):
            seen.update(host=host, port=port)

        with mock.patch.object(app_mod, "_create_app", return_value=object()), \
             mock.patch("uvicorn.run", fake_run):
            app_mod.serve(host="0.0.0.0", port=9999, storage=_storage(),
                          cfg=mock.Mock(), token="t")
        self.assertEqual(seen, {"host": "0.0.0.0", "port": 9999})

    def test_loopback_is_still_the_default_through_serve(self):
        import patmir.ui.app as app_mod

        seen = {}
        with mock.patch.object(app_mod, "_create_app", return_value=object()), \
             mock.patch("uvicorn.run", lambda app, host=None, port=None:
                        seen.update(host=host)):
            app_mod.serve(storage=_storage(), cfg=mock.Mock(), token="t")
        self.assertEqual(seen["host"], "127.0.0.1")


class CliFlagTest(unittest.TestCase):
    """Through `main`, so the arg->cmd_ui wiring is covered too, not just the
    parser default — that wiring is the line a new flag actually gets lost on."""

    def _host_seen(self, argv):
        import patmir.cli as cli
        seen = {}

        def fake_cmd_ui(port, runs_root, config, host="127.0.0.1"):
            seen["host"] = host
            return 0

        with mock.patch.object(cli, "cmd_ui", fake_cmd_ui):
            cli.main(argv)
        return seen.get("host")

    def test_default_is_loopback(self):
        self.assertEqual(self._host_seen(["ui"]), "127.0.0.1")

    def test_override_is_passed_through(self):
        self.assertEqual(self._host_seen(["ui", "--host", "0.0.0.0"]),
                         "0.0.0.0")


class ExtensionOriginTest(unittest.TestCase):
    """Why no auth change was needed for the LAN case."""

    def test_a_chrome_extension_origin_is_allowed(self):
        from patmir.ui.auth import _is_loopback_origin
        self.assertTrue(_is_loopback_origin("chrome-extension://abcdef"))

    def test_a_lan_page_origin_is_still_refused(self):
        # The token routes stay closed to a page served from the LAN IP; only
        # the extension and loopback pages get through.
        from patmir.ui.auth import _is_loopback_origin
        self.assertFalse(_is_loopback_origin("http://192.168.1.50:8765"))


if __name__ == "__main__":
    unittest.main()

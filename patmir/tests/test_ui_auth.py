"""Tests for `patmir.ui.auth`.

Bearer-token gate with two acceptable channels (`Authorization` header,
`?token=` query) and a loopback-origin requirement.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import HTTPException

from patmir.ui import auth


def _make_request(*, headers: dict[str, str] | None = None, qs: dict[str, str] | None = None):
    """Minimal stub of Starlette's `Request` shaped to what TokenAuth reads."""
    req = MagicMock()
    req.headers = headers or {}
    req.query_params = qs or {}
    return req


class EnsureToken(unittest.TestCase):
    def test_generates_then_reuses(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            secret_path = Path(d) / "secret"
            a = auth.ensure_token(secret_path)
            b = auth.ensure_token(secret_path)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 64)  # 32 bytes hex
            self.assertTrue(secret_path.is_file())

    def test_secret_is_chmod_600(self) -> None:
        import os
        with tempfile.TemporaryDirectory() as d:
            secret_path = Path(d) / "secret"
            auth.ensure_token(secret_path)
            mode = secret_path.stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)


class TokenAuthDependency(unittest.TestCase):
    def setUp(self) -> None:
        self.token = "deadbeef" * 8  # 64-char hex-ish
        self.dep = auth.TokenAuth(self.token)

    def test_bearer_header_accepted(self) -> None:
        req = _make_request(headers={"authorization": f"Bearer {self.token}"})
        self.dep(req)  # should not raise

    def test_query_token_accepted(self) -> None:
        req = _make_request(qs={"token": self.token})
        self.dep(req)

    def test_missing_token_401(self) -> None:
        req = _make_request()
        with self.assertRaises(HTTPException) as cm:
            self.dep(req)
        self.assertEqual(cm.exception.status_code, 401)

    def test_wrong_token_401(self) -> None:
        req = _make_request(headers={"authorization": "Bearer wrong"})
        with self.assertRaises(HTTPException) as cm:
            self.dep(req)
        self.assertEqual(cm.exception.status_code, 401)

    def test_non_loopback_origin_403(self) -> None:
        req = _make_request(
            headers={
                "authorization": f"Bearer {self.token}",
                "origin": "https://evil.example.com",
            },
        )
        with self.assertRaises(HTTPException) as cm:
            self.dep(req)
        self.assertEqual(cm.exception.status_code, 403)

    def test_extension_origin_accepted(self) -> None:
        req = _make_request(
            headers={
                "authorization": f"Bearer {self.token}",
                "origin": "chrome-extension://abcdef123",
            },
        )
        self.dep(req)


if __name__ == "__main__":
    unittest.main()

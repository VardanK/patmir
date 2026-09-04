"""Bearer-token auth for the local server.

Token lives at `~/.patmir/secret` (32-hex bytes, chmod 600, auto-generated
on first server start). The Chrome extension reads it once via the options
page and stores it in `chrome.storage.local`.

Token may arrive as either:
- `Authorization: Bearer <token>` header (preferred), or
- `?token=<token>` query param (only because `EventSource` can't set
  headers, so the SSE endpoint needs a query escape hatch).
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import HTTPException, Request, status

from patmir import paths

# Fallback for callers with no loaded config. Real callers pass the path from
# `PipelineConfig.storage.secret_path` so `[storage].state_root` is honored.
DEFAULT_SECRET_PATH = paths.defaults(None).secret_path


def ensure_token(secret_path: Path = DEFAULT_SECRET_PATH) -> str:
    """Read or generate the bearer token. Idempotent; safe to call at startup."""
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.is_file():
        token = secret_path.read_text().strip()
        if token:
            return token
    token = secrets.token_hex(32)
    secret_path.write_text(token)
    try:
        os.chmod(secret_path, 0o600)
    except OSError:
        pass
    return token


def _extract_token(request: Request) -> str | None:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if header:
        scheme, _, token = header.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()
    qs_token = request.query_params.get("token")
    if qs_token:
        return qs_token.strip()
    return None


def _is_loopback_origin(origin: str | None) -> bool:
    if not origin:
        # Missing Origin is fine for extensions + curl + same-origin.
        return True
    o = origin.lower()
    if o.startswith("chrome-extension://"):
        return True
    if o.startswith("moz-extension://"):
        return True
    if o.startswith("http://127.0.0.1") or o.startswith("http://localhost"):
        return True
    if o.startswith("http://[::1]"):
        return True
    return False


class TokenAuth:
    """FastAPI dependency. Construct once at app-factory time, reuse on every route."""

    def __init__(self, expected_token: str) -> None:
        self._expected = expected_token

    def __call__(self, request: Request) -> None:
        if not _is_loopback_origin(request.headers.get("origin")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Origin not allowed.",
            )
        token = _extract_token(request)
        if not token or not secrets.compare_digest(token, self._expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )


def _is_loopback_client(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    host = (client.host or "").strip().lower()
    # A peer address is always numeric, never a hostname. `::ffff:127.x` is
    # how an IPv4 loopback peer is reported on a dual-stack socket (`--host ::`).
    return (host == "::1" or host.startswith("127.")
            or host.startswith("::ffff:127."))


class LoopbackOrToken:
    """FastAPI dependency for the human-facing progress pages (`/`, `/run/*`).

    Those routes predate the token and are opened in a browser by navigation,
    which sends no Authorization header — so they cannot simply take
    `TokenAuth`, or the local progress UI stops working. Before this they took
    nothing at all, which was fine on `127.0.0.1` and was a leak under
    `patmir ui --host 0.0.0.0`: `/` enumerates every job id and source path,
    and `/run/<id>/file/<path>` then serves the source audio, the dub and the
    transcript to anyone on the network.

    A request from the machine itself (loopback client address) passes as
    before; anything else must carry the bearer token like the `/jobs` API.

    The test is on the PEER address as uvicorn reports it. Behind a same-host
    reverse proxy that sets `X-Forwarded-For`, uvicorn's default proxy
    middleware (trusted hosts `127.0.0.1`, chain walked right to left) rewrites
    the client to the real remote address, so the gate still demands the
    token; a proxy that does NOT set that header makes every request look
    local and opens the gate — configure the proxy, or require the token
    unconditionally, before putting one in front of this server.
    """

    def __init__(self, token_auth: TokenAuth) -> None:
        self._token_auth = token_auth

    def __call__(self, request: Request) -> None:
        if _is_loopback_client(request):
            return
        self._token_auth(request)

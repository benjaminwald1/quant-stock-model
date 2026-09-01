"""Password login for public deployments.

On loopback there is nothing to protect: only this machine can reach the
server. Exposed to the internet that stops being true, and the shared word-pair
used for LAN access is not enough — bots scan new domains within hours of the
DNS record appearing.

Passwords are stored as a PBKDF2 hash with a per-install salt, never in the
clear. Sessions are signed cookies, so the server keeps no session table and a
tampered cookie simply fails its signature.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path

log = logging.getLogger(__name__)

ITERATIONS = 240_000
SESSION_DAYS = 14
COOKIE = "qsm_session"


def _auth_path() -> Path:
    from ..config import DATA_DIR

    return DATA_DIR / "auth.json"


def _secret() -> bytes:
    """Signing key for session cookies, generated once and kept with the data."""
    env = os.environ.get("QSM_SECRET_KEY")
    if env:
        return env.encode()
    path = _auth_path().parent / "secret.key"
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return key


def hash_password(password: str, salt: bytes | None = None) -> dict:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return {"salt": base64.b64encode(salt).decode(),
            "hash": base64.b64encode(digest).decode(),
            "iterations": ITERATIONS}


def set_password(password: str, user: str = "owner") -> None:
    if len(password) < 10:
        raise ValueError("Use at least 10 characters.")
    record = {"user": user, **hash_password(password)}
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def is_configured() -> bool:
    return _auth_path().exists() or bool(os.environ.get("QSM_PASSWORD_HASH"))


def verify(password: str) -> bool:
    """Constant-time check against the stored hash."""
    raw = os.environ.get("QSM_PASSWORD_HASH")
    if raw:
        try:
            record = json.loads(base64.b64decode(raw))
        except Exception:
            return False
    else:
        path = _auth_path()
        if not path.exists():
            return False
        try:
            record = json.loads(path.read_text())
        except Exception:
            return False

    salt = base64.b64decode(record["salt"])
    expected = base64.b64decode(record["hash"])
    got = hashlib.pbkdf2_hmac("sha256", password.encode(), salt,
                              int(record.get("iterations", ITERATIONS)))
    return hmac.compare_digest(got, expected)


def issue(user: str = "owner") -> str:
    """A signed session token: payload.signature, no server-side storage."""
    expires = int(time.time()) + SESSION_DAYS * 86400
    payload = base64.urlsafe_b64encode(
        json.dumps({"u": user, "e": expires}).encode()).decode().rstrip("=")
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{sig}"


def check(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    payload, sig = token.rsplit(".", 1)
    expected = hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(sig, expected):
        return False
    try:
        pad = "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload + pad))
    except Exception:
        return False
    return int(data.get("e", 0)) > time.time()

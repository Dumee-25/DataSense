"""
HMAC signing for the anonymous session cookie.

The cookie value is "<session_uuid>.<signature>" where the signature is an
HMAC-SHA256 of the UUID keyed by SECRET_KEY. A cookie that fails verification
is treated as absent, so a visitor cannot claim another session by editing
the cookie value.

If SECRET_KEY is not configured, an ephemeral key is generated at startup —
sessions then reset on every restart, which is safe but inconvenient, so a
warning is logged.
"""

import os
import hmac
import hashlib
import secrets
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SECRET = os.getenv("SECRET_KEY", "")
if not _SECRET:
    _SECRET = secrets.token_hex(32)
    logger.warning(
        "SECRET_KEY is not set — using an ephemeral key. "
        "Session cookies will be invalidated on every restart."
    )
_KEY = _SECRET.encode()


def _signature(session_id: str) -> str:
    return hmac.new(_KEY, session_id.encode(), hashlib.sha256).hexdigest()


def sign_session(session_id: str) -> str:
    """Return the signed cookie value for a session id."""
    return f"{session_id}.{_signature(session_id)}"


def verify_session(cookie_value: Optional[str]) -> Optional[str]:
    """
    Extract the session id from a signed cookie value.
    Returns None for missing, malformed, or tampered cookies.
    """
    if not cookie_value or "." not in cookie_value:
        return None
    session_id, _, signature = cookie_value.partition(".")
    if not hmac.compare_digest(signature, _signature(session_id)):
        return None
    return session_id

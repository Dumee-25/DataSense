import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

_secret = os.getenv("SECRET_KEY", "")
if not _secret:
    logger.warning("SECRET_KEY not set — using insecure default. DO NOT USE IN PRODUCTION.")
    _secret = "dev-only-insecure-default-change-me"
SECRET_KEY = _secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "168"))  # 7 days


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

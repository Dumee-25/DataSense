from typing import Optional
from fastapi import Cookie, Depends
from sqlalchemy.orm import Session as DBSession

from database.connection import get_db
from database.models import User
from utils.auth import decode_access_token
import uuid


def get_current_user(
    db: DBSession = Depends(get_db),
    datasense_token: Optional[str] = Cookie(default=None),
) -> Optional[User]:
    """
    Optional auth dependency.
    Returns the User if a valid JWT cookie exists, None otherwise.
    Anonymous users are fully supported — this never raises an exception.
    """
    if not datasense_token:
        return None
    payload = decode_access_token(datasense_token)
    if not payload:
        return None
    try:
        uid = uuid.UUID(payload.get("sub", ""))
    except (ValueError, AttributeError):
        return None
    user = db.query(User).filter(User.id == uid, User.is_active == True).first()
    return user

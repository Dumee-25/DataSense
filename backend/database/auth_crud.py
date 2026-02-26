import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session as DBSession

from database.models import User, Job, UserRole
from utils.auth import hash_password, verify_password


def get_user_by_email(db: DBSession, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.lower().strip()).first()


def get_user_by_id(db: DBSession, user_id: str) -> Optional[User]:
    try:
        uid = uuid.UUID(user_id)
    except (ValueError, AttributeError):
        return None
    return db.query(User).filter(User.id == uid).first()


def create_user(db: DBSession, email: str, password: str, name: Optional[str] = None) -> User:
    user = User(
        email=email.lower().strip(),
        password_hash=hash_password(password),
        name=name,
        role=UserRole.analyst,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: DBSession, email: str, password: str) -> Optional[User]:
    user = get_user_by_email(db, email)
    if not user or not user.password_hash:
        return None
    if not verify_password(password, user.password_hash):
        return None
    if not user.is_active:
        return None
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    return user


def update_user_profile(
    db: DBSession,
    user_id: str,
    name: Optional[str] = None,
    new_password: Optional[str] = None,
) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    if name is not None:
        user.name = name
    if new_password:
        user.password_hash = hash_password(new_password)
    db.commit()
    db.refresh(user)
    return user


def link_session_jobs_to_user(db: DBSession, session_id: str, user_id: str) -> int:
    """
    When a user logs in, link all their anonymous session jobs to their account.
    This means their history persists even if they clear cookies.
    Returns the number of jobs linked.
    """
    try:
        sid = uuid.UUID(session_id)
        uid = uuid.UUID(user_id)
    except (ValueError, AttributeError):
        return 0

    updated = db.query(Job).filter(
        Job.session_id == sid,
        Job.user_id == None  # Only link jobs not already owned
    ).update({"user_id": uid})
    db.commit()
    return updated


def delete_user(db: DBSession, user_id: str) -> bool:
    user = get_user_by_id(db, user_id)
    if not user:
        return False
    db.delete(user)
    db.commit()
    return True

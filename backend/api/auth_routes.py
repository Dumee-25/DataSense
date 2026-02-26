import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Response, Cookie
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session as DBSession

from database.connection import get_db
from database import crud
from database import auth_crud
from utils.auth import create_access_token
from utils.dependencies import get_current_user
from database.models import User

logger = logging.getLogger(__name__)
auth_router = APIRouter(tags=["auth"])

ACCESS_TOKEN_EXPIRE_HOURS = 168  # 7 days


# ── Request / Response Models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None


def _user_response(user: User) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role.value,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None,
    }


def _set_auth_cookie(response: Response, token: str):
    response.set_cookie(
        key="datasense_token",
        value=token,
        max_age=60 * 60 * ACCESS_TOKEN_EXPIRE_HOURS,
        httponly=True,
        samesite="lax",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@auth_router.post("/register", status_code=201)
def register(
    body: RegisterRequest,
    response: Response,
    db: DBSession = Depends(get_db),
    datasense_session: Optional[str] = Cookie(default=None),
):
    """
    Create a new account.
    Any jobs from the current anonymous session are automatically linked
    to the new account so history is preserved.
    """
    existing = auth_crud.get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user = auth_crud.create_user(db, email=body.email, password=body.password, name=body.name)
    logger.info(f"New user registered: {user.email}")

    # Link anonymous session jobs → new account
    linked = 0
    if datasense_session:
        linked = auth_crud.link_session_jobs_to_user(db, datasense_session, str(user.id))
        if linked:
            logger.info(f"Linked {linked} session jobs to new user {user.email}")

    token = create_access_token(str(user.id), user.email)
    _set_auth_cookie(response, token)

    return {
        "message": "Account created successfully.",
        "user": _user_response(user),
        "session_jobs_linked": linked,
    }


@auth_router.post("/login")
def login(
    body: LoginRequest,
    response: Response,
    db: DBSession = Depends(get_db),
    datasense_session: Optional[str] = Cookie(default=None),
):
    """
    Log in with email and password.
    Anonymous session jobs are linked to the account on login.
    """
    user = auth_crud.authenticate_user(db, body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # Link any anonymous session jobs → this account
    linked = 0
    if datasense_session:
        linked = auth_crud.link_session_jobs_to_user(db, datasense_session, str(user.id))
        if linked:
            logger.info(f"Linked {linked} session jobs to {user.email} on login")

    token = create_access_token(str(user.id), user.email)
    _set_auth_cookie(response, token)

    return {
        "message": "Logged in successfully.",
        "user": _user_response(user),
        "session_jobs_linked": linked,
    }


@auth_router.post("/logout")
def logout(response: Response):
    """Clear the auth cookie."""
    response.delete_cookie("datasense_token")
    return {"message": "Logged out successfully."}


@auth_router.get("/me")
def get_me(current_user: Optional[User] = Depends(get_current_user)):
    """
    Get current user profile.
    Returns null if not logged in (anonymous session).
    """
    if not current_user:
        return {"user": None, "authenticated": False}
    return {
        "user": _user_response(current_user),
        "authenticated": True,
    }


@auth_router.patch("/me")
def update_profile(
    body: UpdateProfileRequest,
    db: DBSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Update name or password. Must be logged in."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    # If changing password, verify current password first
    if body.new_password:
        if not body.current_password:
            raise HTTPException(
                status_code=400,
                detail="current_password is required to set a new password."
            )
        if not auth_crud.authenticate_user(db, current_user.email, body.current_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect.")
        if len(body.new_password) < 8:
            raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")

    user = auth_crud.update_user_profile(
        db,
        user_id=str(current_user.id),
        name=body.name,
        new_password=body.new_password,
    )
    return {"message": "Profile updated.", "user": _user_response(user)}


@auth_router.delete("/me")
def delete_account(
    response: Response,
    db: DBSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Permanently delete the account and all associated data."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")

    auth_crud.delete_user(db, str(current_user.id))
    response.delete_cookie("datasense_token")
    logger.info(f"User deleted their account: {current_user.email}")
    return {"message": "Account deleted."}

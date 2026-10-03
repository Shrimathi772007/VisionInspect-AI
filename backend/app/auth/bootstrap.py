"""Create the first quality engineer, or promote an existing user, outside the HTTP API.

Self-registration (POST /auth/register) always creates a factory_supervisor, so the first
quality engineer has to come from here (see scripts/create_quality_engineer.py). This module
only flushes; the caller owns the transaction and commits.
"""

from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.schemas import NAME_MAX_LENGTH, PASSWORD_MAX_LENGTH, PASSWORD_MIN_LENGTH
from app.auth.security import hash_password
from app.models.user import User, UserRole

_email_adapter = TypeAdapter(EmailStr)


class BootstrapError(ValueError):
    """Invalid input or a refused operation. Messages never contain the password."""


class UserAlreadyExistsError(BootstrapError):
    pass


def validate_password(password: str) -> None:
    if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
        raise BootstrapError(
            f"Password must be between {PASSWORD_MIN_LENGTH} and {PASSWORD_MAX_LENGTH} characters"
        )


def normalize_email(email: str) -> str:
    try:
        return _email_adapter.validate_python(email)
    except ValidationError:
        raise BootstrapError(f"Invalid email address: {email!r}") from None


def find_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(select(User).where(User.email == normalize_email(email))).scalar_one_or_none()


def create_or_promote_quality_engineer(
    db: Session,
    name: str,
    email: str,
    password: str | None,
    promote_existing: bool,
) -> User:
    """Return a quality engineer for `email`.

    - email unknown: create a quality engineer with a bcrypt-hashed password (required).
    - email known and promote_existing: set that user's role to quality_engineer. The
      existing name and password are kept; `password` is ignored.
    - email known and not promote_existing: raise UserAlreadyExistsError.
    """
    email = normalize_email(email)
    existing_user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if existing_user is not None:
        if not promote_existing:
            raise UserAlreadyExistsError(
                f"A user with email {email} already exists; use --promote-existing to make them a quality engineer"
            )
        existing_user.role = UserRole.quality_engineer
        db.flush()
        return existing_user

    name = name.strip()
    if not 1 <= len(name) <= NAME_MAX_LENGTH:
        raise BootstrapError(f"Name must be between 1 and {NAME_MAX_LENGTH} characters")
    if password is None:
        raise BootstrapError("A password is required to create a new user")
    validate_password(password)

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
        role=UserRole.quality_engineer,
    )
    db.add(user)
    db.flush()
    return user

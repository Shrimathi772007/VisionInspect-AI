from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models.user import UserRole

# bcrypt only uses the first 72 bytes of a password; shared with app.auth.bootstrap.
# The 8-72 rule counts characters; a non-ASCII character takes 2-4 bytes in UTF-8, so a
# password within 72 characters can still exceed bcrypt's 72-byte limit and is refused too.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 72
PASSWORD_MAX_BYTES = 72
PASSWORD_TOO_MANY_BYTES_MESSAGE = (
    f"Password must be at most {PASSWORD_MAX_BYTES} bytes when UTF-8 encoded "
    "(accented and other non-ASCII characters count as 2-4 bytes each)"
)
NAME_MAX_LENGTH = 255


def password_exceeds_byte_limit(password: str) -> bool:
    return len(password.encode("utf-8")) > PASSWORD_MAX_BYTES


class UserCreate(BaseModel):
    # No `role` field: self-registration always creates a factory_supervisor. A `role` sent
    # by the client is ignored (Pydantic's default extra="ignore"), never an error.
    name: str = Field(min_length=1, max_length=NAME_MAX_LENGTH)
    email: EmailStr
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)

    @field_validator("password")
    @classmethod
    def password_within_bcrypt_byte_limit(cls, password: str) -> str:
        # Raising ValueError here makes FastAPI answer 422 instead of bcrypt failing with a 500.
        if password_exceeds_byte_limit(password):
            raise ValueError(PASSWORD_TOO_MANY_BYTES_MESSAGE)
        return password


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    role: UserRole
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

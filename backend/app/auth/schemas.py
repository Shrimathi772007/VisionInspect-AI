from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.user import UserRole

# bcrypt only uses the first 72 bytes of a password; shared with app.auth.bootstrap.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 72
NAME_MAX_LENGTH = 255


class UserCreate(BaseModel):
    # No `role` field: self-registration always creates a factory_supervisor. A `role` sent
    # by the client is ignored (Pydantic's default extra="ignore"), never an error.
    name: str = Field(min_length=1, max_length=NAME_MAX_LENGTH)
    email: EmailStr
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)


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

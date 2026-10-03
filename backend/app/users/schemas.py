from pydantic import BaseModel

from app.auth.schemas import UserOut
from app.models.user import UserRole

__all__ = ["UserOut", "UserRoleUpdate"]


class UserRoleUpdate(BaseModel):
    role: UserRole

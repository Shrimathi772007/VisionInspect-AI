from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import require_role
from app.database import get_db
from app.models.user import User, UserRole
from app.users import service
from app.users.schemas import UserOut, UserRoleUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
    return service.list_users(db)


@router.patch("/{user_id}/role", response_model=UserOut)
def change_user_role(
    user_id: int,
    payload: UserRoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.quality_engineer)),
):
    try:
        return service.change_user_role(db, user_id, payload.role)
    except service.UserNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    except service.LastQualityEngineerError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot change the role of the last quality engineer; promote another user first",
        )

"""User role management. Authorization is checked in the router (quality engineers only)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User, UserRole


class UserNotFoundError(Exception):
    pass


class LastQualityEngineerError(Exception):
    pass


def list_users(db: Session) -> list[User]:
    return list(db.execute(select(User).order_by(User.id)).scalars().all())


def count_quality_engineers_for_update(db: Session) -> int:
    # Locks every quality-engineer row until the transaction ends, so two concurrent demotions
    # cannot both see "2 left" and leave the system with none.
    return len(
        db.execute(
            select(User.id).where(User.role == UserRole.quality_engineer).order_by(User.id).with_for_update()
        ).all()
    )


def change_user_role(db: Session, user_id: int, role: UserRole) -> User:
    # Take the lock before reading the target, so its role is read after any concurrent change.
    quality_engineer_count = count_quality_engineers_for_update(db)

    user = db.get(User, user_id, populate_existing=True)
    if user is None:
        raise UserNotFoundError
    if user.role == role:
        return user
    if user.role == UserRole.quality_engineer and quality_engineer_count <= 1:
        raise LastQualityEngineerError

    user.role = role
    db.commit()
    db.refresh(user)
    return user

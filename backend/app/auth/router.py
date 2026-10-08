from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.rate_limit import TOO_MANY_ATTEMPTS_MESSAGE, login_rate_limiter
from app.auth.schemas import Token, UserCreate, UserLogin, UserOut
from app.auth.security import create_access_token, hash_password, verify_password
from app.database import get_db
from app.models.user import User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if existing_user is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        # Never client-chosen. Quality engineers are created with scripts/create_quality_engineer.py
        # or promoted by an existing quality engineer via PATCH /users/{user_id}/role.
        role=UserRole.factory_supervisor,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(payload: UserLogin, request: Request, db: Session = Depends(get_db)):
    # Behind nginx this is the address nginx saw (uvicorn --proxy-headers; nginx overwrites X-Forwarded-For).
    client_ip = request.client.host if request.client else "unknown"
    # Checked before the password, so a locked-out client learns nothing about whether it was right.
    retry_after = login_rate_limiter.retry_after(client_ip, payload.email)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=TOO_MANY_ATTEMPTS_MESSAGE,
            headers={"Retry-After": str(retry_after)},
        )

    invalid_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        login_rate_limiter.record_failure(client_ip, payload.email)
        raise invalid_credentials

    login_rate_limiter.reset(client_ip, payload.email)
    access_token = create_access_token({"sub": str(user.id), "role": user.role.value})
    return Token(access_token=access_token, user=user)


@router.get("/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

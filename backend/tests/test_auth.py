import pytest
from sqlalchemy import select

from app.auth.bootstrap import BootstrapError, UserAlreadyExistsError, create_or_promote_quality_engineer
from app.auth.security import verify_password
from app.database import SessionLocal
from app.models.user import User, UserRole


def _stored_role(email: str) -> UserRole:
    with SessionLocal() as session:
        return session.execute(select(User.role).where(User.email == email)).scalar_one()


def _register(client, email: str, **extra):
    return client.post(
        "/auth/register",
        json={"name": "Temp Registrant", "email": email, "password": "TempPass123", **extra},
    )


def test_register_cannot_choose_quality_engineer_role(client, temp_users):
    email = temp_users.email()
    response = _register(client, email, role="quality_engineer")
    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "factory_supervisor"
    assert "password" not in body
    assert "password_hash" not in body
    assert _stored_role(email) == UserRole.factory_supervisor


def test_register_without_role_creates_factory_supervisor(client, temp_users):
    email = temp_users.email()
    response = _register(client, email)
    assert response.status_code == 201
    assert response.json()["role"] == "factory_supervisor"
    assert _stored_role(email) == UserRole.factory_supervisor


def test_register_with_invalid_role_is_ignored(client, temp_users):
    # `role` is not part of the request schema at all, so any value is ignored rather than rejected.
    email = temp_users.email()
    response = _register(client, email, role="superuser")
    assert response.status_code == 201
    assert response.json()["role"] == "factory_supervisor"
    assert _stored_role(email) == UserRole.factory_supervisor


def test_register_login_and_me_never_expose_password_hash(client, temp_users):
    email = temp_users.email()
    register_body = _register(client, email).json()
    login_body = client.post("/auth/login", json={"email": email, "password": "TempPass123"}).json()
    me_body = client.get("/auth/me", headers={"Authorization": f"Bearer {login_body['access_token']}"}).json()
    for user_body in (register_body, login_body["user"], me_body):
        assert set(user_body.keys()) == {"id", "name", "email", "role", "created_at"}
    assert set(login_body.keys()) == {"access_token", "token_type", "user"}


def test_register_factory_supervisor(client):
    response = client.post(
        "/auth/register",
        json={
            "name": "Temp Supervisor",
            "email": "temp.supervisor.auth.test@example.com",
            "password": "TempPass456",
            "role": "factory_supervisor",
        },
    )
    assert response.status_code in (201, 409)
    if response.status_code == 201:
        body = response.json()
        assert body["role"] == "factory_supervisor"
        assert "password" not in body
        assert "password_hash" not in body


def test_duplicate_email_registration_rejected(client):
    payload = {
        "name": "Duplicate Attempt",
        "email": "temp.qe.auth.test@example.com",
        "password": "AnotherPass123",
        "role": "quality_engineer",
    }
    client.post("/auth/register", json=payload)
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 409


# bcrypt only uses 72 bytes, so the limit is checked on the UTF-8 encoding, not just characters.
MULTIBYTE_90_BYTES = "€" * 30  # 30 characters of a 3-byte character = 90 bytes


def test_register_accepts_72_ascii_characters(client, temp_users):
    email = temp_users.email()
    response = _register(client, email, password="x" * 72)
    assert response.status_code == 201
    login = client.post("/auth/login", json={"email": email, "password": "x" * 72})
    assert login.status_code == 200


def test_register_rejects_73_ascii_characters(client, temp_users):
    email = temp_users.email()
    response = _register(client, email, password="x" * 73)
    assert response.status_code == 422
    with SessionLocal() as session:
        assert session.execute(select(User).where(User.email == email)).scalar_one_or_none() is None


def test_register_rejects_multibyte_password_over_72_bytes_with_422(client, temp_users):
    assert len(MULTIBYTE_90_BYTES) == 30 and len(MULTIBYTE_90_BYTES.encode("utf-8")) == 90
    email = temp_users.email()
    response = _register(client, email, password=MULTIBYTE_90_BYTES)
    assert response.status_code == 422
    messages = " ".join(error["msg"] for error in response.json()["detail"])
    assert "at most 72 bytes" in messages
    with SessionLocal() as session:
        assert session.execute(select(User).where(User.email == email)).scalar_one_or_none() is None


def test_register_accepts_short_multibyte_password(client, temp_users):
    password = "Päss€ürdé"  # 10 characters, 15 bytes
    assert len(password.encode("utf-8")) <= 72
    email = temp_users.email()
    response = _register(client, email, password=password)
    assert response.status_code == 201
    login = client.post("/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200


def test_login_correct_credentials(client, qe_credentials):
    assert qe_credentials["token_type"] == "bearer"
    assert "access_token" in qe_credentials
    assert "password" not in qe_credentials["user"]
    assert "password_hash" not in qe_credentials["user"]


def test_login_incorrect_password(client):
    response = client.post(
        "/auth/login",
        json={"email": "temp.qe.auth.test@example.com", "password": "WrongPassword"},
    )
    assert response.status_code == 401


def test_login_nonexistent_email(client):
    response = client.post(
        "/auth/login",
        json={"email": "does.not.exist@example.com", "password": "Whatever123"},
    )
    assert response.status_code == 401


def test_me_without_token(client):
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_with_invalid_token(client):
    response = client.get("/auth/me", headers={"Authorization": "Bearer not.a.valid.token"})
    assert response.status_code == 401


def test_me_with_valid_token(client, qe_headers):
    response = client.get("/auth/me", headers=qe_headers)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"id", "name", "email", "role", "created_at"}


# ---------------------------------------------------------------------------
# Quality-engineer bootstrap (app.auth.bootstrap). Each test runs in a transaction that is
# rolled back, so nothing is written to the database.
# ---------------------------------------------------------------------------

@pytest.fixture
def rollback_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_bootstrap_creates_quality_engineer(rollback_session, temp_users):
    email = temp_users.email()
    user = create_or_promote_quality_engineer(
        rollback_session, "Bootstrap QE", email, "BootstrapPass123", promote_existing=False
    )
    stored = rollback_session.execute(select(User).where(User.email == email)).scalar_one()
    assert stored.id == user.id
    assert stored.role == UserRole.quality_engineer
    assert stored.password_hash != "BootstrapPass123"
    assert verify_password("BootstrapPass123", stored.password_hash)


def test_bootstrap_refuses_existing_email_without_promote(rollback_session, temp_users):
    email = temp_users.email()
    rollback_session.add(User(name="Existing", email=email, password_hash="x", role=UserRole.factory_supervisor))
    rollback_session.flush()
    with pytest.raises(UserAlreadyExistsError):
        create_or_promote_quality_engineer(rollback_session, "Existing", email, "BootstrapPass123", promote_existing=False)
    stored = rollback_session.execute(select(User).where(User.email == email)).scalar_one()
    assert stored.role == UserRole.factory_supervisor


def test_bootstrap_promotes_existing_user(rollback_session, temp_users):
    email = temp_users.email()
    existing = User(name="Existing", email=email, password_hash="unchanged", role=UserRole.factory_supervisor)
    rollback_session.add(existing)
    rollback_session.flush()
    user = create_or_promote_quality_engineer(rollback_session, "Ignored", email, None, promote_existing=True)
    assert user.id == existing.id
    assert user.role == UserRole.quality_engineer
    assert user.name == "Existing"
    assert user.password_hash == "unchanged"


@pytest.mark.parametrize("password", ["Short7!", "x" * 73, ""])
def test_bootstrap_rejects_password_outside_8_to_72_characters(rollback_session, temp_users, password):
    email = temp_users.email()
    with pytest.raises(BootstrapError) as exc_info:
        create_or_promote_quality_engineer(rollback_session, "Bootstrap QE", email, password, promote_existing=False)
    assert "between 8 and 72" in str(exc_info.value)
    if password:
        assert password not in str(exc_info.value)
    assert rollback_session.execute(select(User).where(User.email == email)).scalar_one_or_none() is None


def test_bootstrap_accepts_password_at_8_and_72_characters(rollback_session, temp_users):
    for password in ("x" * 8, "x" * 72):
        user = create_or_promote_quality_engineer(
            rollback_session, "Bootstrap QE", temp_users.email(), password, promote_existing=False
        )
        assert user.role == UserRole.quality_engineer


def test_bootstrap_rejects_multibyte_password_over_72_bytes(rollback_session, temp_users):
    email = temp_users.email()
    with pytest.raises(BootstrapError) as exc_info:
        create_or_promote_quality_engineer(
            rollback_session, "Bootstrap QE", email, MULTIBYTE_90_BYTES, promote_existing=False
        )
    assert "at most 72 bytes" in str(exc_info.value)
    assert MULTIBYTE_90_BYTES not in str(exc_info.value)
    assert rollback_session.execute(select(User).where(User.email == email)).scalar_one_or_none() is None


def test_bootstrap_accepts_short_multibyte_password(rollback_session, temp_users):
    user = create_or_promote_quality_engineer(
        rollback_session, "Bootstrap QE", temp_users.email(), "Päss€ürdé", promote_existing=False
    )
    assert verify_password("Päss€ürdé", user.password_hash)


def test_bootstrap_requires_password_for_new_user(rollback_session, temp_users):
    with pytest.raises(BootstrapError):
        create_or_promote_quality_engineer(rollback_session, "Bootstrap QE", temp_users.email(), None, promote_existing=True)

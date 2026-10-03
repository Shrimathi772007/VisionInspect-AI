from sqlalchemy import func, select

from app.database import SessionLocal
from app.models.user import User, UserRole
from app.users import service as users_service
from tests.conftest import QE_EMAIL, login


def _stored_role(user_id: int) -> UserRole:
    with SessionLocal() as session:
        return session.get(User, user_id).role


def _headers(client, email: str, password: str) -> dict:
    return {"Authorization": f"Bearer {login(client, email, password)['access_token']}"}


def _set_role(client, headers, user_id: int, role: str):
    return client.patch(f"/users/{user_id}/role", json={"role": role}, headers=headers)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

def test_users_endpoints_require_authentication(client):
    assert client.get("/users").status_code == 401
    assert client.patch("/users/1/role", json={"role": "quality_engineer"}).status_code == 401


def test_supervisor_cannot_list_users_or_change_roles(client, supervisor_headers, supervisor_credentials):
    assert client.get("/users", headers=supervisor_headers).status_code == 403
    own_id = supervisor_credentials["user"]["id"]
    response = _set_role(client, supervisor_headers, own_id, "quality_engineer")
    assert response.status_code == 403
    assert _stored_role(own_id) == UserRole.factory_supervisor


# ---------------------------------------------------------------------------
# GET /users
# ---------------------------------------------------------------------------

def test_quality_engineer_lists_users_ordered_by_id(client, qe_headers):
    response = client.get("/users", headers=qe_headers)
    assert response.status_code == 200
    users = response.json()
    ids = [user["id"] for user in users]
    assert ids == sorted(ids)
    assert QE_EMAIL in {user["email"] for user in users}
    for user in users:
        assert set(user.keys()) == {"id", "name", "email", "role", "created_at"}
    with SessionLocal() as session:
        assert len(users) == session.execute(select(func.count(User.id))).scalar_one()


# ---------------------------------------------------------------------------
# PATCH /users/{user_id}/role
# ---------------------------------------------------------------------------

def test_quality_engineer_promotes_supervisor_and_back(client, qe_headers, temp_users):
    user_id, email = temp_users.create(UserRole.factory_supervisor)
    target_headers = _headers(client, email, temp_users.password)
    assert client.get("/users", headers=target_headers).status_code == 403

    response = _set_role(client, qe_headers, user_id, "quality_engineer")
    assert response.status_code == 200
    assert response.json()["role"] == "quality_engineer"
    assert "password_hash" not in response.json()
    assert _stored_role(user_id) == UserRole.quality_engineer
    # Same token, new authority: roles are read from the database on every request.
    assert client.get("/users", headers=target_headers).status_code == 200

    response = _set_role(client, qe_headers, user_id, "factory_supervisor")
    assert response.status_code == 200
    assert response.json()["role"] == "factory_supervisor"
    assert _stored_role(user_id) == UserRole.factory_supervisor


def test_same_role_change_is_a_no_op(client, qe_headers, temp_users):
    user_id, email = temp_users.create(UserRole.factory_supervisor)
    response = _set_role(client, qe_headers, user_id, "factory_supervisor")
    assert response.status_code == 200
    assert response.json()["id"] == user_id
    assert response.json()["email"] == email
    assert response.json()["role"] == "factory_supervisor"
    assert _stored_role(user_id) == UserRole.factory_supervisor


def test_change_role_of_nonexistent_user_returns_404(client, qe_headers):
    with SessionLocal() as session:
        missing_id = (session.execute(select(func.max(User.id))).scalar_one() or 0) + 1_000_000
    response = _set_role(client, qe_headers, missing_id, "quality_engineer")
    assert response.status_code == 404


def test_change_role_rejects_unknown_role(client, qe_headers, temp_users):
    user_id, _ = temp_users.create(UserRole.factory_supervisor)
    response = _set_role(client, qe_headers, user_id, "superuser")
    assert response.status_code == 422
    assert _stored_role(user_id) == UserRole.factory_supervisor


def test_demoting_one_of_several_quality_engineers_works(client, qe_headers, temp_users):
    user_id, _ = temp_users.create(UserRole.quality_engineer)
    response = _set_role(client, qe_headers, user_id, "factory_supervisor")
    assert response.status_code == 200
    assert response.json()["role"] == "factory_supervisor"
    assert _stored_role(user_id) == UserRole.factory_supervisor


def test_demoting_the_last_quality_engineer_returns_409_and_changes_nothing(
    client, qe_headers, temp_users, monkeypatch
):
    # The shared development database always holds other quality engineers, and existing users
    # must not be demoted to set this up, so the locked count is pinned to 1 instead.
    user_id, _ = temp_users.create(UserRole.quality_engineer)
    monkeypatch.setattr(users_service, "count_quality_engineers_for_update", lambda db: 1)

    response = _set_role(client, qe_headers, user_id, "factory_supervisor")
    assert response.status_code == 409
    assert "last quality engineer" in response.json()["detail"]
    assert _stored_role(user_id) == UserRole.quality_engineer

    # Promoting, or a no-op on the last quality engineer, is still allowed.
    assert _set_role(client, qe_headers, user_id, "quality_engineer").status_code == 200


def test_quality_engineer_count_reflects_database(temp_users):
    with SessionLocal() as session:
        before = users_service.count_quality_engineers_for_update(session)
        session.rollback()
    temp_users.create(UserRole.quality_engineer)
    temp_users.create(UserRole.factory_supervisor)
    with SessionLocal() as session:
        assert users_service.count_quality_engineers_for_update(session) == before + 1
        session.rollback()


def test_token_issued_to_quality_engineer_loses_authority_after_demotion(client, qe_headers, temp_users):
    user_id, email = temp_users.create(UserRole.quality_engineer)
    old_headers = _headers(client, email, temp_users.password)
    assert client.get("/users", headers=old_headers).status_code == 200

    assert _set_role(client, qe_headers, user_id, "factory_supervisor").status_code == 200

    # The token still carries role=quality_engineer, but authority comes from the database.
    assert client.get("/users", headers=old_headers).status_code == 403
    assert client.delete("/products/999999999", headers=old_headers).status_code == 403
    assert client.get("/auth/me", headers=old_headers).json()["role"] == "factory_supervisor"

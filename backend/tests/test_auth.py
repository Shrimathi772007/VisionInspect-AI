def test_register_quality_engineer(client):
    response = client.post(
        "/auth/register",
        json={
            "name": "Temp QE",
            "email": "temp.qe.auth.test@example.com",
            "password": "TempPass123",
            "role": "quality_engineer",
        },
    )
    assert response.status_code in (201, 409)
    if response.status_code == 201:
        body = response.json()
        assert body["role"] == "quality_engineer"
        assert "password" not in body
        assert "password_hash" not in body


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

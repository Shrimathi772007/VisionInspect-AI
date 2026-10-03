from io import BytesIO
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete, select

from app.auth.security import hash_password
from app.database import SessionLocal
from app.main import app
from app.models.user import User, UserRole

QE_EMAIL = "pytest.qe@example.com"
QE_PASSWORD = "PytestQEPass123"
SUPERVISOR_EMAIL = "pytest.supervisor@example.com"
SUPERVISOR_PASSWORD = "PytestSupPass123"
PRODUCT_CODE = "PYTEST-WIDGET-001"


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (16, 16), color=(200, 30, 30)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer.read()


def login(client: TestClient, email: str, password: str) -> dict:
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    return response.json()


def register_or_login(client: TestClient, name: str, email: str, password: str) -> dict:
    # Self-registration always creates a factory_supervisor; quality engineers come from ensure_user.
    response = client.post(
        "/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    if response.status_code == 409:
        return login(client, email, password)

    assert response.status_code == 201
    return login(client, email, password)


def ensure_user(name: str, email: str, password: str, role: UserRole) -> int:
    """Create a user directly in the database (the API cannot create quality engineers), or reuse it.

    An existing user is never modified: if it has a different role the test setup fails loudly.
    """
    with SessionLocal() as session:
        user = session.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None:
            user = User(name=name, email=email, password_hash=hash_password(password), role=role)
            session.add(user)
            session.commit()
        assert user.role == role, f"Existing test user {email} has role {user.role.value}, expected {role.value}"
        return user.id


class TemporaryUsers:
    """Throwaway users with unique emails; every user whose email came from here is deleted afterwards."""

    password = "TempUserPass123"

    def __init__(self):
        self.emails: list[str] = []

    def email(self) -> str:
        email = f"pytest.tmp.{uuid4().hex}@example.com"
        self.emails.append(email)
        return email

    def create(self, role: UserRole, name: str = "Pytest Temporary User") -> tuple[int, str]:
        email = self.email()
        return ensure_user(name, email, self.password, role), email

    def cleanup(self) -> None:
        if self.emails:
            with SessionLocal() as session:
                session.execute(delete(User).where(User.email.in_(self.emails)))
                session.commit()


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def qe_credentials(client):
    ensure_user("Pytest QE", QE_EMAIL, QE_PASSWORD, UserRole.quality_engineer)
    return login(client, QE_EMAIL, QE_PASSWORD)


@pytest.fixture(scope="session")
def supervisor_credentials(client):
    return register_or_login(client, "Pytest Supervisor", SUPERVISOR_EMAIL, SUPERVISOR_PASSWORD)


@pytest.fixture(scope="session")
def qe_headers(qe_credentials):
    return {"Authorization": f"Bearer {qe_credentials['access_token']}"}


@pytest.fixture(scope="session")
def supervisor_headers(supervisor_credentials):
    return {"Authorization": f"Bearer {supervisor_credentials['access_token']}"}


@pytest.fixture(scope="session")
def test_product(client, qe_headers):
    response = client.post(
        "/products",
        json={"product_name": "Pytest Widget", "product_code": PRODUCT_CODE},
        headers=qe_headers,
    )
    if response.status_code == 409:
        listing = client.get("/products", headers=qe_headers)
        assert listing.status_code == 200
        for product in listing.json():
            if product["product_code"] == PRODUCT_CODE:
                return product
        raise AssertionError("Product reported as duplicate but not found in listing")

    assert response.status_code == 201
    return response.json()


@pytest.fixture
def temp_users():
    users = TemporaryUsers()
    yield users
    users.cleanup()

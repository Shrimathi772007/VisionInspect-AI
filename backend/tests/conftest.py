from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete, select

from app.auth.security import hash_password
from app.database import SessionLocal
from app.main import app
from app.models.inspection import Inspection
from app.models.product import Product
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


@pytest.fixture(autouse=True)
def _served_models_never_score_dataset_test_images(monkeypatch):
    """Every MVTec category has a served final model, and each was scored exactly once on its dataset/<category>/test/
    split. Many API tests import test/ images (for ground-truth status, severity, analytics), and an import runs
    real AI inference - so through app.inspections.service, a test/ image is refused before any model sees it and
    the inspection keeps ai_* NULL, the existing best-effort path. Tests that monkeypatch predict_image themselves
    replace this guard for their own duration."""
    import app.inspections.service as inspection_service
    from app.inspections.storage import DATASET_ROOT

    real_predict = inspection_service.predict_image
    dataset_root = Path(DATASET_ROOT).resolve()

    def guarded(image_path, category, *args, **kwargs):
        resolved = Path(image_path).resolve()
        if resolved.is_relative_to(dataset_root):
            parts = resolved.relative_to(dataset_root).parts
            if len(parts) >= 2 and parts[1] == "test":
                raise RuntimeError("test harness: dataset test/ images are never scored by a served model")
        return real_predict(image_path, category, *args, **kwargs)

    monkeypatch.setattr(inspection_service, "predict_image", guarded)


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


class CategoryProducts:
    """Throwaway products (code PYTEST-CAT-<hex>) with an optional MVTec category.

    Teardown deletes every inspection of these products through the API (which also removes
    uploaded files from storage, and never touches the dataset), then the products themselves.
    """

    def __init__(self, client: TestClient, qe_headers: dict):
        self.client = client
        self.qe_headers = qe_headers
        self.product_ids: list[int] = []

    def create(self, category: str | None = None) -> dict:
        payload = {"product_name": "Pytest Category Widget", "product_code": f"PYTEST-CAT-{uuid4().hex[:12]}"}
        if category is not None:
            payload["category"] = category
        response = self.client.post("/products", json=payload, headers=self.qe_headers)
        assert response.status_code == 201, response.text
        product = response.json()
        self.product_ids.append(product["id"])
        return product

    def upload(self, product: dict, image_bytes: bytes, filename: str = "sample.png"):
        return self.client.post(
            "/inspections/upload",
            headers=self.qe_headers,
            data={"product_id": str(product["id"])},
            files={"file": (filename, image_bytes, "image/png")},
        )

    def cleanup(self) -> None:
        if not self.product_ids:
            return
        with SessionLocal() as session:
            inspection_ids = session.execute(
                select(Inspection.id).where(Inspection.product_id.in_(self.product_ids))
            ).scalars().all()
        for inspection_id in inspection_ids:
            self.client.delete(f"/inspections/{inspection_id}", headers=self.qe_headers)
        for product_id in self.product_ids:
            self.client.delete(f"/products/{product_id}", headers=self.qe_headers)
        with SessionLocal() as session:
            leftover = session.execute(select(Product.id).where(Product.id.in_(self.product_ids))).scalars().all()
        assert not leftover, f"test products not cleaned up: {leftover}"


@pytest.fixture
def category_products(client, qe_headers):
    products = CategoryProducts(client, qe_headers)
    yield products
    products.cleanup()

"""Shared fixtures - and, BEFORE any app import, the test-isolation setup:

* Database: the suite never uses the development database. It runs against "<dev db name>_test" on the
  same server (or VISIONINSPECT_TEST_DATABASE_URL when set). pytest_sessionstart creates that database if
  it does not exist (via the "postgres" maintenance database) and runs `alembic upgrade head` on it once
  per session. The run is refused if the resolved test database is the development database.
* Storage: uploads and heatmaps go to a per-session temporary directory (UPLOAD_STORAGE_ROOT /
  HEATMAP_STORAGE_ROOT), deleted at the end of the session, so tests never write into backend/storage.

app.database.session and app.inspections.storage read these environment variables at import time
(load_dotenv never overrides a variable that is already set), so they are set here, first.
"""

import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy.engine import make_url

BACKEND_DIR = Path(__file__).resolve().parent.parent
TEST_DATABASE_URL_ENV = "VISIONINSPECT_TEST_DATABASE_URL"
load_dotenv(BACKEND_DIR / ".env")

DEV_DB = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": str(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "visioninspect_db"),
}


def _resolve_test_database() -> dict:
    """{user, password, host, port, database} of the test database (never printed: holds the password)."""
    url = os.getenv(TEST_DATABASE_URL_ENV)
    if url:
        parsed = make_url(url)
        return {
            "user": parsed.username or os.getenv("POSTGRES_USER", "postgres"),
            "password": unquote(parsed.password) if parsed.password else os.getenv("POSTGRES_PASSWORD", ""),
            "host": parsed.host or "localhost",
            "port": str(parsed.port or 5432),
            "database": parsed.database,
        }
    return {
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "host": DEV_DB["host"],
        "port": DEV_DB["port"],
        "database": f"{DEV_DB['database']}_test",
    }


def _same_database(a: dict, b: dict) -> bool:
    local = {"localhost", "127.0.0.1", "::1"}
    same_host = a["host"] == b["host"] or (a["host"] in local and b["host"] in local)
    return same_host and str(a["port"]) == str(b["port"]) and a["database"] == b["database"]


TEST_DB = _resolve_test_database()
if not TEST_DB["database"] or _same_database(TEST_DB, DEV_DB):
    raise RuntimeError(
        f"Refusing to run the tests: the test database {TEST_DB['database']!r} on {TEST_DB['host']}:{TEST_DB['port']} "
        "is the development database. Set VISIONINSPECT_TEST_DATABASE_URL to a separate database."
    )
os.environ.update({
    "POSTGRES_USER": TEST_DB["user"],
    "POSTGRES_PASSWORD": TEST_DB["password"],
    "POSTGRES_HOST": TEST_DB["host"],
    "POSTGRES_PORT": TEST_DB["port"],
    "POSTGRES_DB": TEST_DB["database"],
})

TEST_STORAGE_DIR = Path(tempfile.mkdtemp(prefix="visioninspect_test_storage_"))
(TEST_STORAGE_DIR / "uploads").mkdir()
(TEST_STORAGE_DIR / "heatmaps").mkdir()
os.environ["UPLOAD_STORAGE_ROOT"] = str(TEST_STORAGE_DIR / "uploads")
os.environ["HEATMAP_STORAGE_ROOT"] = str(TEST_STORAGE_DIR / "heatmaps")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from sqlalchemy import create_engine, delete, select, text  # noqa: E402
from sqlalchemy.engine import URL  # noqa: E402

from app.auth.security import hash_password  # noqa: E402
from app.database import SessionLocal, engine  # noqa: E402
from app.inspections.storage import HEATMAP_ROOT, STORAGE_ROOT  # noqa: E402
from app.main import app  # noqa: E402
from app.models.inspection import Inspection  # noqa: E402
from app.models.product import Product  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402


def _ensure_test_database() -> None:
    """Create the test database if it does not exist (through the "postgres" maintenance database)."""
    maintenance = URL.create(
        "postgresql+psycopg", username=TEST_DB["user"], password=TEST_DB["password"] or None,
        host=TEST_DB["host"], port=int(TEST_DB["port"]), database="postgres",
    )
    admin = create_engine(maintenance, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :name"),
                                  {"name": TEST_DB["database"]}).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{TEST_DB["database"]}"'))
    finally:
        admin.dispose()


def pytest_sessionstart(session):
    # The app's engine must point at the test database - checked again after the app import.
    if engine.url.database != TEST_DB["database"] or _same_database(
        {"host": engine.url.host, "port": str(engine.url.port), "database": engine.url.database}, DEV_DB
    ):
        raise pytest.UsageError("The application engine is not bound to the test database; refusing to run.")
    if STORAGE_ROOT.resolve() != (TEST_STORAGE_DIR / "uploads").resolve() or HEATMAP_ROOT.resolve() != (TEST_STORAGE_DIR / "heatmaps").resolve():
        raise pytest.UsageError("Upload/heatmap storage is not the per-session test directory; refusing to run.")
    _ensure_test_database()
    from alembic import command
    from alembic.config import Config

    # No ini file: migrations/env.py then skips logging.fileConfig, so the test session's loggers stay as
    # they are; env.py takes the URL from app.database.session (the test database).
    alembic_config = Config()
    alembic_config.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    command.upgrade(alembic_config, "head")


def pytest_sessionfinish(session, exitstatus):
    engine.dispose()
    shutil.rmtree(TEST_STORAGE_DIR, ignore_errors=True)

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

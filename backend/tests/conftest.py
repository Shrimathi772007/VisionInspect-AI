from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

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


def register_or_login(client: TestClient, name: str, email: str, password: str, role: str) -> dict:
    response = client.post(
        "/auth/register",
        json={"name": name, "email": email, "password": password, "role": role},
    )
    if response.status_code == 409:
        response = client.post("/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200
        return response.json()

    assert response.status_code == 201
    login_response = client.post("/auth/login", json={"email": email, "password": password})
    assert login_response.status_code == 200
    return login_response.json()


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(scope="session")
def qe_credentials(client):
    return register_or_login(client, "Pytest QE", QE_EMAIL, QE_PASSWORD, "quality_engineer")


@pytest.fixture(scope="session")
def supervisor_credentials(client):
    return register_or_login(client, "Pytest Supervisor", SUPERVISOR_EMAIL, SUPERVISOR_PASSWORD, "factory_supervisor")


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

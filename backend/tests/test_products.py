from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.database import SessionLocal
from app.models.inspection import Inspection
from app.models.product import Product
from tests.conftest import make_image_bytes


def test_create_and_list_products(client, qe_headers, test_product):
    assert test_product["product_code"] == "PYTEST-WIDGET-001"
    assert "id" in test_product

    response = client.get("/products", headers=qe_headers)
    assert response.status_code == 200
    codes = [p["product_code"] for p in response.json()]
    assert "PYTEST-WIDGET-001" in codes


def test_duplicate_product_code_rejected(client, qe_headers, test_product):
    response = client.post(
        "/products",
        json={"product_name": "Duplicate Widget", "product_code": test_product["product_code"]},
        headers=qe_headers,
    )
    assert response.status_code == 409


def test_products_require_authentication(client):
    response = client.get("/products")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /products/{product_id}
# ---------------------------------------------------------------------------

def _create_product(client, qe_headers, product_code, product_name="Delete Me Widget"):
    response = client.post(
        "/products",
        json={"product_name": product_name, "product_code": product_code},
        headers=qe_headers,
    )
    assert response.status_code == 201
    return response.json()


def test_delete_product_with_no_inspections_succeeds(client, qe_headers):
    product = _create_product(client, qe_headers, "PYTEST-DELETE-001")

    response = client.delete(f"/products/{product['id']}", headers=qe_headers)
    assert response.status_code == 204

    listing = client.get("/products", headers=qe_headers)
    assert all(p["id"] != product["id"] for p in listing.json())


def test_delete_nonexistent_product(client, qe_headers):
    response = client.delete("/products/999999", headers=qe_headers)
    assert response.status_code == 404


def test_delete_product_requires_authentication(client, qe_headers):
    product = _create_product(client, qe_headers, "PYTEST-DELETE-002")

    response = client.delete(f"/products/{product['id']}")
    assert response.status_code == 401

    # cleanup: confirm it's still there, then remove via authenticated call
    listing = client.get("/products", headers=qe_headers)
    assert any(p["id"] == product["id"] for p in listing.json())
    assert client.delete(f"/products/{product['id']}", headers=qe_headers).status_code == 204


def test_delete_product_requires_quality_engineer(client, qe_headers, supervisor_headers):
    product = _create_product(client, qe_headers, "PYTEST-DELETE-003")

    response = client.delete(f"/products/{product['id']}", headers=supervisor_headers)
    assert response.status_code == 403

    # cleanup
    assert client.delete(f"/products/{product['id']}", headers=qe_headers).status_code == 204


def test_delete_product_with_inspections_returns_409(client, qe_headers):
    product = _create_product(client, qe_headers, "PYTEST-DELETE-004")

    from tests.conftest import make_image_bytes

    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(product["id"])},
        files={"file": ("sample.png", make_image_bytes("PNG"), "image/png")},
    )
    assert upload_response.status_code == 201
    inspection_id = upload_response.json()["id"]

    response = client.delete(f"/products/{product['id']}", headers=qe_headers)
    assert response.status_code == 409

    # existing inspection and product must remain intact
    assert client.get(f"/inspections/{inspection_id}", headers=qe_headers).status_code == 200
    listing = client.get("/products", headers=qe_headers)
    assert any(p["id"] == product["id"] for p in listing.json())

    # cleanup: remove inspection first, then the product
    assert client.delete(f"/inspections/{inspection_id}", headers=qe_headers).status_code == 204
    assert client.delete(f"/products/{product['id']}", headers=qe_headers).status_code == 204


def test_users_remain_intact_after_product_deletion(client, qe_headers):
    product = _create_product(client, qe_headers, "PYTEST-DELETE-005")
    me_before = client.get("/auth/me", headers=qe_headers)
    assert me_before.status_code == 200

    assert client.delete(f"/products/{product['id']}", headers=qe_headers).status_code == 204

    me_after = client.get("/auth/me", headers=qe_headers)
    assert me_after.status_code == 200
    assert me_after.json()["id"] == me_before.json()["id"]


# ---------------------------------------------------------------------------
# Product MVTec category (selects the AI model for uploaded images)
# ---------------------------------------------------------------------------

def test_create_product_with_valid_category(client, category_products):
    product = category_products.create(category="metal_nut")
    assert product["category"] == "metal_nut"


def test_create_product_without_category(client, category_products):
    product = category_products.create(category=None)
    assert product["category"] is None


@pytest.mark.parametrize("category", ["Tile", " tile", "tile ", "plastic", "", "metal-nut"])
def test_create_product_with_invalid_category_rejected(client, qe_headers, category):
    response = client.post(
        "/products",
        json={"product_name": "Bad Category", "product_code": f"PYTEST-BADCAT-{uuid4().hex[:8]}", "category": category},
        headers=qe_headers,
    )
    assert response.status_code == 422


def test_product_creation_stays_open_to_supervisors(client, supervisor_headers, qe_headers):
    response = client.post(
        "/products",
        json={"product_name": "Supervisor Widget", "product_code": f"PYTEST-CAT-{uuid4().hex[:12]}", "category": "tile"},
        headers=supervisor_headers,
    )
    assert response.status_code == 201
    assert response.json()["category"] == "tile"
    assert client.delete(f"/products/{response.json()['id']}", headers=qe_headers).status_code == 204


def test_quality_engineer_sets_and_clears_category(client, qe_headers, category_products):
    product = category_products.create(category=None)

    response = client.patch(f"/products/{product['id']}/category", json={"category": "tile"}, headers=qe_headers)
    assert response.status_code == 200
    assert response.json()["category"] == "tile"
    assert response.json()["id"] == product["id"]

    response = client.patch(f"/products/{product['id']}/category", json={"category": None}, headers=qe_headers)
    assert response.status_code == 200
    assert response.json()["category"] is None

    with SessionLocal() as session:
        assert session.get(Product, product["id"]).category is None


def test_change_category_requires_quality_engineer(client, supervisor_headers, category_products):
    product = category_products.create(category="tile")
    response = client.patch(f"/products/{product['id']}/category", json={"category": "cable"}, headers=supervisor_headers)
    assert response.status_code == 403
    with SessionLocal() as session:
        assert session.get(Product, product["id"]).category == "tile"


def test_change_category_requires_authentication(client, category_products):
    product = category_products.create(category="tile")
    response = client.patch(f"/products/{product['id']}/category", json={"category": "cable"})
    assert response.status_code == 401


def test_change_category_of_unknown_product_returns_404(client, qe_headers):
    with SessionLocal() as session:
        missing_id = (session.execute(select(func.max(Product.id))).scalar_one() or 0) + 1_000_000
    response = client.patch(f"/products/{missing_id}/category", json={"category": "tile"}, headers=qe_headers)
    assert response.status_code == 404


@pytest.mark.parametrize("body", [{"category": "Tile"}, {"category": "plastic"}, {"category": 7}, {}])
def test_change_category_rejects_invalid_body(client, qe_headers, category_products, body):
    product = category_products.create(category="tile")
    response = client.patch(f"/products/{product['id']}/category", json=body, headers=qe_headers)
    assert response.status_code == 422
    with SessionLocal() as session:
        assert session.get(Product, product["id"]).category == "tile"


def test_changing_category_leaves_existing_inspections_unchanged(client, qe_headers, category_products, monkeypatch):
    from app.ai.inference import PredictionResult

    monkeypatch.setattr(
        "app.inspections.service.predict_image",
        lambda path, category, *a, **k: PredictionResult(
            category=category,
            prediction="defective",
            reconstruction_error=3.0,
            threshold=2.0,
            model_name="fake-model",
            input_size=(256, 256),
            processing_time_ms=5.0,
        ),
    )
    product = category_products.create(category="tile")
    inspection_id = category_products.upload(product, make_image_bytes("PNG")).json()["id"]

    def stored_fields():
        with SessionLocal() as session:
            row = session.get(Inspection, inspection_id)
            return {c.name: getattr(row, c.name) for c in Inspection.__table__.columns}

    before = stored_fields()
    assert before["ai_prediction"] == "defective"

    for category in ("cable", None):
        response = client.patch(f"/products/{product['id']}/category", json={"category": category}, headers=qe_headers)
        assert response.status_code == 200

    assert stored_fields() == before
    # The response field reflects the product's current category, not a stored copy.
    assert client.get(f"/inspections/{inspection_id}", headers=qe_headers).json()["product_category"] is None

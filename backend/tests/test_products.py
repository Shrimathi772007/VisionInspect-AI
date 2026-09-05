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

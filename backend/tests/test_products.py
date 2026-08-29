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

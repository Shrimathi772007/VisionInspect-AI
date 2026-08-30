from io import BytesIO
from pathlib import Path

from PIL import Image

from app.inspections.storage import DATASET_ROOT

BOTTLE_GOOD_FILE = "000.png"
BOTTLE_DEFECT_FILE = "000.png"


# ---------------------------------------------------------------------------
# GET /dataset/categories
# ---------------------------------------------------------------------------

def test_list_categories(client, qe_headers):
    response = client.get("/dataset/categories", headers=qe_headers)
    assert response.status_code == 200
    categories = response.json()
    assert isinstance(categories, list)
    assert "bottle" in categories
    assert "carpet" in categories
    assert categories == sorted(categories)


def test_list_categories_requires_authentication(client):
    response = client.get("/dataset/categories")
    assert response.status_code == 401


def test_list_categories_supervisor_can_browse(client, supervisor_headers):
    response = client.get("/dataset/categories", headers=supervisor_headers)
    assert response.status_code == 200
    assert "bottle" in response.json()


# ---------------------------------------------------------------------------
# GET /dataset/categories/{category}
# ---------------------------------------------------------------------------

def test_category_detail(client, qe_headers):
    response = client.get("/dataset/categories/bottle", headers=qe_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "bottle"

    splits = {s["split"]: s for s in body["splits"]}
    assert "train" in splits
    assert "test" in splits

    train_defect_types = {d["defect_type"]: d["count"] for d in splits["train"]["defect_types"]}
    assert train_defect_types.get("good", 0) > 0

    test_defect_types = {d["defect_type"]: d["count"] for d in splits["test"]["defect_types"]}
    assert test_defect_types.get("good", 0) > 0
    assert test_defect_types.get("broken_large", 0) > 0


def test_category_detail_unknown_category(client, qe_headers):
    response = client.get("/dataset/categories/not_a_real_category", headers=qe_headers)
    assert response.status_code == 404


def test_category_detail_requires_authentication(client):
    response = client.get("/dataset/categories/bottle")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /dataset/categories/{category}/images
# ---------------------------------------------------------------------------

def test_list_images_good(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/images",
        params={"split": "test", "defect_type": "good"},
        headers=qe_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "bottle"
    assert body["split"] == "test"
    assert body["defect_type"] == "good"
    assert BOTTLE_GOOD_FILE in body["filenames"]
    assert body["filenames"] == sorted(body["filenames"])


def test_list_images_defective(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/images",
        params={"split": "test", "defect_type": "broken_large"},
        headers=qe_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["filenames"]) > 0


def test_list_images_unknown_defect_type(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/images",
        params={"split": "test", "defect_type": "not_a_real_defect"},
        headers=qe_headers,
    )
    assert response.status_code == 404


def test_list_images_invalid_split(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/images",
        params={"split": "validation", "defect_type": "good"},
        headers=qe_headers,
    )
    assert response.status_code == 400


def test_list_images_requires_authentication(client):
    response = client.get(
        "/dataset/categories/bottle/images", params={"split": "test", "defect_type": "good"}
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# GET /dataset/categories/{category}/preview  (basic preprocessing)
# ---------------------------------------------------------------------------

def test_preview_returns_resized_jpeg(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "good", "filename": BOTTLE_GOOD_FILE, "max_dim": 100},
        headers=qe_headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

    img = Image.open(BytesIO(response.content))
    assert img.format == "JPEG"
    assert img.mode == "RGB"
    assert max(img.size) <= 100


def test_preview_default_max_dim(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "good", "filename": BOTTLE_GOOD_FILE},
        headers=qe_headers,
    )
    assert response.status_code == 200
    img = Image.open(BytesIO(response.content))
    assert max(img.size) <= 320


def test_preview_defective_image(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "broken_large", "filename": BOTTLE_DEFECT_FILE},
        headers=qe_headers,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_preview_nonexistent_filename(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "good", "filename": "999999.png"},
        headers=qe_headers,
    )
    assert response.status_code == 404


def test_preview_requires_authentication(client):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "good", "filename": BOTTLE_GOOD_FILE},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Path traversal / security cases
# ---------------------------------------------------------------------------

def test_preview_rejects_path_traversal_in_filename(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "good", "filename": "../../../../etc/passwd"},
        headers=qe_headers,
    )
    assert response.status_code == 400


def test_preview_rejects_path_traversal_in_defect_type(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "../../../", "filename": BOTTLE_GOOD_FILE},
        headers=qe_headers,
    )
    assert response.status_code == 400


def test_images_rejects_path_traversal_in_defect_type(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/images",
        params={"split": "test", "defect_type": "../../../"},
        headers=qe_headers,
    )
    assert response.status_code == 400


def test_preview_rejects_absolute_path_style_filename(client, qe_headers):
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "good", "filename": "/etc/passwd"},
        headers=qe_headers,
    )
    assert response.status_code == 400


def test_category_detail_rejects_traversal_category(client, qe_headers):
    response = client.get("/dataset/categories/%2e%2e%2f%2e%2e", headers=qe_headers)
    assert response.status_code in (400, 404)


def test_preview_cannot_escape_dataset_root_via_dotdot_segment(client, qe_headers):
    # Even if a segment were composed of only dots/dashes (passes the character
    # allowlist), resolve_image_path's containment check must still block escape.
    response = client.get(
        "/dataset/categories/bottle/preview",
        params={"split": "test", "defect_type": "good", "filename": ".."},
        headers=qe_headers,
    )
    assert response.status_code in (400, 404)


# ---------------------------------------------------------------------------
# POST /inspections/import
# ---------------------------------------------------------------------------

def test_import_good_image(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["product_id"] == test_product["id"]
    assert body["source"] == "mvtec_ad"
    assert body["status"] == "good"
    assert body["dataset_category"] == "bottle"
    assert body["dataset_split"] == "test"
    assert body["dataset_defect_type"] == "good"
    assert body["dataset_filename"] == BOTTLE_GOOD_FILE
    assert "image_path" not in body


def test_import_defective_image(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "broken_large",
            "filename": BOTTLE_DEFECT_FILE,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "defective"
    assert body["dataset_defect_type"] == "broken_large"
    assert "image_path" not in body


def test_import_requires_authentication(client, test_product):
    response = client.post(
        "/inspections/import",
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 401


def test_import_forbidden_for_supervisor(client, supervisor_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=supervisor_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 403


def test_import_rejects_nonexistent_product(client, qe_headers):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": 999999,
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 404


def test_import_rejects_unknown_category(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "not_a_real_category",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 404


def test_import_rejects_invalid_split(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "validation",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 400


def test_import_rejects_unknown_defect_type(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "not_a_real_defect",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 404


def test_import_rejects_nonexistent_filename(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": "999999.png",
        },
    )
    assert response.status_code == 404


def test_import_rejects_path_traversal_filename(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": "../../../../etc/passwd",
        },
    )
    assert response.status_code == 400


def test_import_rejects_path_traversal_category(client, qe_headers, test_product):
    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "../backend",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 400


def test_imported_inspection_serves_real_dataset_bytes(client, qe_headers, test_product):
    import_response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert import_response.status_code == 201
    inspection_id = import_response.json()["id"]

    image_response = client.get(f"/inspections/{inspection_id}/image", headers=qe_headers)
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

    on_disk_path = DATASET_ROOT / "bottle" / "test" / "good" / BOTTLE_GOOD_FILE
    assert image_response.content == on_disk_path.read_bytes()


def test_imported_inspection_image_requires_authentication(client, qe_headers, test_product):
    import_response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}/image")
    assert response.status_code == 401


def test_import_does_not_copy_file_into_upload_storage(client, qe_headers, test_product):
    from app.inspections.storage import STORAGE_ROOT

    existing_files = set(STORAGE_ROOT.rglob("*")) if STORAGE_ROOT.is_dir() else set()

    response = client.post(
        "/inspections/import",
        headers=qe_headers,
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": BOTTLE_GOOD_FILE,
        },
    )
    assert response.status_code == 201

    files_after = set(STORAGE_ROOT.rglob("*")) if STORAGE_ROOT.is_dir() else set()
    assert files_after == existing_files, "importing a dataset image must not write into upload storage"


def test_uploaded_inspection_has_no_dataset_metadata(client, qe_headers, test_product):
    from tests.conftest import make_image_bytes

    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", make_image_bytes("PNG"), "image/png")},
    )
    assert upload_response.status_code == 201
    body = upload_response.json()
    assert body["source"] == "upload"
    assert body["dataset_category"] is None
    assert body["dataset_split"] is None
    assert body["dataset_defect_type"] is None
    assert body["dataset_filename"] is None
    assert "image_path" not in body

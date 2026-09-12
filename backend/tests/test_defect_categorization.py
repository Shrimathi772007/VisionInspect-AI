"""Milestone 3 Phase 1: defect categorization foundation.

Inspection.defect_category is the known ground-truth defect category/type - populated
from the MVTec dataset's own defect_type directory name on import, left NULL for generic
uploads (never guessed from the autoencoder). It is a separate concept from `status`
(business ground truth) and `ai_prediction` (the anomaly detector's guess) and this suite
asserts those stay independent, alongside the existing AI/status/authorization behavior.
"""

from sqlalchemy import inspect as sa_inspect

from app.models.inspection import Inspection
from tests.conftest import make_image_bytes


def _import(client, headers, product_id, defect_type, filename="000.png", category="bottle", split="test"):
    return client.post(
        "/inspections/import",
        headers=headers,
        json={
            "product_id": product_id,
            "category": category,
            "split": split,
            "defect_type": defect_type,
            "filename": filename,
        },
    )


# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------

def test_inspection_model_has_defect_category_column():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    assert "defect_category" in columns
    assert columns["defect_category"].nullable


def test_new_inspection_row_defaults_defect_category_to_null():
    inspection = Inspection(product_id=1, image_path="bottle/test/good/000.png")
    assert inspection.defect_category is None


# ---------------------------------------------------------------------------
# MVTec import stores the dataset's ground-truth defect category
# ---------------------------------------------------------------------------

def test_import_good_stores_defect_category_good(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "good", filename="001.png")
    assert response.status_code == 201
    assert response.json()["defect_category"] == "good"


def test_import_broken_large_stores_defect_category(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "broken_large", filename="001.png")
    assert response.status_code == 201
    assert response.json()["defect_category"] == "broken_large"


def test_import_broken_small_stores_defect_category(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "broken_small", filename="001.png")
    assert response.status_code == 201
    assert response.json()["defect_category"] == "broken_small"


def test_import_contamination_stores_defect_category(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "contamination", filename="001.png")
    assert response.status_code == 201
    assert response.json()["defect_category"] == "contamination"


# ---------------------------------------------------------------------------
# Generic user uploads leave defect_category null/unknown
# ---------------------------------------------------------------------------

def test_upload_leaves_defect_category_null(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    assert response.status_code == 201
    assert response.json()["defect_category"] is None


# ---------------------------------------------------------------------------
# Independence from status and ai_prediction
# ---------------------------------------------------------------------------

def test_defect_category_independent_of_status_and_ai_prediction(client, qe_headers, test_product):
    """A broken_large import must carry all three concepts independently: the existing
    ground-truth status logic, the new defect_category, and the AI prediction - none may be
    derived from or overwritten by another."""
    response = _import(client, qe_headers, test_product["id"], "broken_large", filename="002.png")
    assert response.status_code == 201
    body = response.json()

    assert body["status"] == "defective"  # existing ground-truth-derived status, untouched
    assert body["defect_category"] == "broken_large"  # new, independent field
    assert body["dataset_defect_type"] == "broken_large"  # existing dataset reference, untouched
    assert body["ai_prediction"] in ("good", "defective", None)  # independently derived


def test_import_good_does_not_force_ai_prediction_good(client, qe_headers, test_product):
    """defect_category=good must never be copied onto ai_prediction - they are populated by
    entirely separate code paths (this endpoint vs. run_ai_inference)."""
    response = _import(client, qe_headers, test_product["id"], "good", filename="003.png")
    assert response.status_code == 201
    body = response.json()
    assert body["defect_category"] == "good"
    assert body["ai_prediction"] in ("good", "defective", None)


# ---------------------------------------------------------------------------
# Existing AI fields, status behavior, and authorization are unaffected
# ---------------------------------------------------------------------------

def test_existing_ai_fields_still_populate_alongside_defect_category(client, qe_headers, test_product):
    response = _import(client, qe_headers, test_product["id"], "good", filename="004.png")
    assert response.status_code == 201
    body = response.json()
    assert isinstance(body["ai_reconstruction_error"], float)
    assert isinstance(body["ai_threshold"], float)
    assert body["ai_model_name"] == "autoencoder"


def test_import_still_requires_quality_engineer_role(client, supervisor_headers, test_product):
    response = _import(client, supervisor_headers, test_product["id"], "good", filename="005.png")
    assert response.status_code == 403


def test_import_still_requires_authentication(client, test_product):
    response = client.post(
        "/inspections/import",
        json={
            "product_id": test_product["id"],
            "category": "bottle",
            "split": "test",
            "defect_type": "good",
            "filename": "006.png",
        },
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Inspection detail response includes defect_category
# ---------------------------------------------------------------------------

def test_get_inspection_detail_includes_defect_category(client, qe_headers, supervisor_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "contamination", filename="007.png")
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}", headers=supervisor_headers)
    assert response.status_code == 200
    assert response.json()["defect_category"] == "contamination"


def test_list_inspections_includes_defect_category(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "broken_small", filename="008.png")
    created_id = import_response.json()["id"]

    response = client.get("/inspections", headers=qe_headers)
    assert response.status_code == 200
    match = next(item for item in response.json() if item["id"] == created_id)
    assert match["defect_category"] == "broken_small"

"""Milestone 3 Phase 4: production quality reports.

GET /inspections/{id}/report composes an inspection's existing Phase 1-3 evidence into a
structured, read-only report - app.inspections.report.build_production_quality_report
implements no new business logic and no second quality-decision algorithm. The report's
`quality.decision` / `report_summary.overall_result` must always equal the existing
Inspection.quality_decision (or the existing NOT_ASSESSED vocabulary for a legacy row with
no quality_decision yet) - never a value independently recomputed here.
"""

from app.inspections.quality import FAIL, NOT_ASSESSED, PASS
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
# Endpoint existence / authorization
# ---------------------------------------------------------------------------

def test_report_endpoint_exists_for_quality_engineer(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="000.png")
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers)
    assert response.status_code == 200


def test_report_accessible_by_factory_supervisor(client, qe_headers, supervisor_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="001.png")
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}/report", headers=supervisor_headers)
    assert response.status_code == 200


def test_report_rejects_unauthenticated_access(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="002.png")
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}/report")
    assert response.status_code == 401


def test_report_for_unknown_inspection_returns_404(client, qe_headers):
    response = client.get("/inspections/99999999/report", headers=qe_headers)
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Report content - MVTec import
# ---------------------------------------------------------------------------

def test_report_includes_product_information(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="003.png")
    inspection_id = import_response.json()["id"]

    body = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers).json()
    assert body["inspection"]["product_id"] == test_product["id"]
    assert body["inspection"]["product_name"] == test_product["product_name"]


def test_report_includes_inspection_information(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="004.png")
    inspection = import_response.json()

    body = client.get(f"/inspections/{inspection['id']}/report", headers=qe_headers).json()
    assert body["inspection"]["id"] == inspection["id"]
    assert body["inspection"]["status"] == inspection["status"]
    assert body["inspection"]["source"] == inspection["source"]
    assert body["inspection"]["inspection_date"] == inspection["inspection_date"]
    assert body["inspection"]["created_at"] == inspection["created_at"]


def test_report_defect_category_included(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "broken_large", filename="005.png")
    inspection_id = import_response.json()["id"]

    body = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers).json()
    assert body["defect"]["category"] == "broken_large"


def test_report_ai_prediction_fields_included(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="006.png")
    inspection = import_response.json()

    body = client.get(f"/inspections/{inspection['id']}/report", headers=qe_headers).json()
    assert body["defect"]["ai_prediction"] == inspection["ai_prediction"]
    assert body["defect"]["ai_reconstruction_error"] == inspection["ai_reconstruction_error"]
    assert body["defect"]["ai_threshold"] == inspection["ai_threshold"]
    assert body["defect"]["ai_model_name"] == inspection["ai_model_name"]


def test_report_severity_fields_included(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "broken_small", filename="007.png")
    inspection = import_response.json()

    body = client.get(f"/inspections/{inspection['id']}/report", headers=qe_headers).json()
    assert body["severity"]["score"] == inspection["severity_score"]
    assert body["severity"]["level"] == inspection["severity_level"]
    assert body["severity"]["quality_risk"] == inspection["quality_risk"]


def test_report_quality_decision_assessment_recommendation_included(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "contamination", filename="008.png")
    inspection = import_response.json()

    body = client.get(f"/inspections/{inspection['id']}/report", headers=qe_headers).json()
    assert body["quality"]["decision"] == inspection["quality_decision"]
    assert body["quality"]["assessment"] == inspection["quality_assessment"]
    assert body["quality"]["recommendation"] == inspection["quality_recommendation"]


def test_report_overall_result_matches_existing_quality_decision(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "broken_large", filename="009.png")
    inspection = import_response.json()

    body = client.get(f"/inspections/{inspection['id']}/report", headers=qe_headers).json()
    assert body["report_summary"]["overall_result"] == inspection["quality_decision"]
    assert body["report_summary"]["overall_result"] in (PASS, FAIL, NOT_ASSESSED)


def test_report_dataset_section_present_for_mvtec_import(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "contamination", filename="010.png")
    inspection_id = import_response.json()["id"]

    body = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers).json()
    assert body["dataset"] == {
        "category": "bottle",
        "split": "test",
        "defect_type": "contamination",
        "filename": "010.png",
    }


# ---------------------------------------------------------------------------
# Report content - generic upload (no fabricated MVTec evidence)
# ---------------------------------------------------------------------------

def test_report_for_generic_upload_has_no_dataset_section(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    inspection_id = upload_response.json()["id"]

    body = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers).json()
    assert body["dataset"] is None
    assert body["defect"]["category"] is None
    assert body["defect"]["ai_prediction"] is None


def test_report_for_generic_upload_is_not_assessed_not_fabricated_pass(client, qe_headers, test_product):
    image_bytes = make_image_bytes("PNG")
    upload_response = client.post(
        "/inspections/upload",
        headers=qe_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", image_bytes, "image/png")},
    )
    inspection_id = upload_response.json()["id"]

    body = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers).json()
    assert body["report_summary"]["overall_result"] == NOT_ASSESSED
    assert body["severity"]["score"] is None
    assert body["severity"]["level"] is None


# ---------------------------------------------------------------------------
# Missing quality assessment represented safely (simulated pre-Phase-3 row)
# ---------------------------------------------------------------------------

def test_report_handles_null_quality_decision_as_not_assessed():
    """A row created before Phase 3 (quality_decision/assessment/recommendation all NULL)
    must report NOT_ASSESSED, never a fabricated PASS, and never crash."""
    from datetime import datetime, timezone

    from app.inspections.report import build_production_quality_report
    from app.models.inspection import Inspection, InspectionSource
    from app.models.product import Product

    # inspection_date/created_at are NOT NULL columns populated by a DB server_default on
    # insert - a real legacy row always has them; set them explicitly here since this
    # Inspection is never actually persisted.
    now = datetime.now(timezone.utc)
    inspection = Inspection(
        id=1,
        product_id=1,
        image_path="bottle/test/good/000.png",
        source=InspectionSource.mvtec_ad,
        status="good",
        defect_category="good",
        inspection_date=now,
        created_at=now,
    )
    product = Product(id=1, product_name="Legacy Widget", product_code="LEGACY-1")

    report = build_production_quality_report(inspection, product)
    assert report.quality.decision == NOT_ASSESSED
    assert report.report_summary.overall_result == NOT_ASSESSED
    assert report.report_summary.report_status == "PARTIAL"
    assert report.quality.assessment is None
    assert report.quality.recommendation is None


# ---------------------------------------------------------------------------
# Data integrity: image_path never exposed, existing fields unchanged
# ---------------------------------------------------------------------------

def test_report_never_exposes_image_path(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="011.png")
    inspection_id = import_response.json()["id"]

    body = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers).json()
    dumped = str(body)
    assert "image_path" not in dumped
    assert "011.png" in dumped  # filename is fine to expose (dataset section); the raw path field name is not


def test_report_does_not_mutate_existing_inspection_fields(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "broken_small", filename="012.png")
    inspection_id = import_response.json()["id"]
    before = client.get(f"/inspections/{inspection_id}", headers=qe_headers).json()

    client.get(f"/inspections/{inspection_id}/report", headers=qe_headers)
    client.get(f"/inspections/{inspection_id}/report", headers=qe_headers)  # call twice - must stay idempotent

    after = client.get(f"/inspections/{inspection_id}", headers=qe_headers).json()
    assert before == after


def test_ground_truth_and_ai_prediction_remain_distinguishable_in_report(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "broken_large", filename="013.png")
    inspection_id = import_response.json()["id"]

    body = client.get(f"/inspections/{inspection_id}/report", headers=qe_headers).json()
    # category is MVTec ground truth; ai_prediction is the independent AI guess - both
    # present as separate fields, category never mislabeled as an AI classification.
    assert body["defect"]["category"] == "broken_large"
    assert body["defect"]["ai_prediction"] in ("good", "defective", None)


# ---------------------------------------------------------------------------
# Existing endpoint unaffected
# ---------------------------------------------------------------------------

def test_existing_inspection_detail_endpoint_still_works(client, qe_headers, test_product):
    import_response = _import(client, qe_headers, test_product["id"], "good", filename="014.png")
    inspection_id = import_response.json()["id"]

    response = client.get(f"/inspections/{inspection_id}", headers=qe_headers)
    assert response.status_code == 200
    assert "image_path" not in response.json()


# ---------------------------------------------------------------------------
# Route surface: exactly one new route added
# ---------------------------------------------------------------------------

def test_phase4_adds_exactly_one_new_api_route(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    operations = sum(
        1 for methods in paths.values() for m in methods if m.lower() in ("get", "post", "put", "patch", "delete")
    )
    assert operations == 20  # 19 existing (through Phase 3) + this phase's /report route
    assert "/inspections/{inspection_id}/report" in paths

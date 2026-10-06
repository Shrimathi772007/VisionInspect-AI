"""GET /inspections/analytics/by-category (app.inspections.analytics.get_category_analytics).

The test database is shared across the session, so exact-count tests pin "today" to a date in 2001
(analytics.datetime is replaced) and insert their own rows with created_at in that period, where no other
test's inspection can exist. Rows and products are deleted afterwards.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import delete

import app.inspections.analytics as analytics
from app.database import SessionLocal
from app.dataset.categories import MVTEC_CATEGORIES
from app.inspections.analytics import UNCATEGORISED, get_category_analytics
from app.inspections.quality import FAIL, MANUAL_REVIEW, NOT_ASSESSED, PASS
from app.inspections.service import _resolve_category, resolve_category
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product

URL = "/inspections/analytics/by-category"
SUMMARY_URL = "/inspections/analytics/summary"
ROW_KEYS = {"category", "total", "ai_analysed", "ai_defective", "ai_good", "defect_rate", "manual_review",
            "pass_count", "fail_count", "manual_review_decisions", "not_assessed"}


def _at(day: int, month: int = 1, year: int = 2001) -> datetime:
    return datetime(year, month, day, 12, 0, tzinfo=timezone.utc)


def _pin_today(monkeypatch, today: datetime):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return today if tz is None else today.astimezone(tz)

    monkeypatch.setattr(analytics, "datetime", FixedDatetime)


@pytest.fixture
def seeded_2001():
    """Products (pill, no category) and inspections in Dec 2000 / Jan 2001:

        2001-01-30  import screw/test/scratch_head (pill product)  defective  review  MANUAL_REVIEW
        2001-01-30  import screw/test/good         (pill product)  good                PASS
        2001-01-30  upload                         (pill product)  defective  review  FAIL
        2001-01-30  upload                         (no category)   -                  decision NULL
        2001-01-30  import with a non-dataset path (pill product)  -                  NOT_ASSESSED
        2001-01-20  import bottle/test/good        (no category)   good                PASS
        2000-12-15  import tile/test/good          (pill product)  good                PASS   (outside 30 days)
    """
    tag = uuid4().hex[:10]
    with SessionLocal() as db:
        pill = Product(product_name="Pytest by-category pill", product_code=f"PYTEST-BYCAT-PILL-{tag}", category="pill")
        bare = Product(product_name="Pytest by-category none", product_code=f"PYTEST-BYCAT-NONE-{tag}", category=None)
        db.add_all([pill, bare])
        db.flush()
        rows = [
            (pill, InspectionSource.mvtec_ad, "screw/test/scratch_head/000.png", "defective", True, MANUAL_REVIEW, _at(30)),
            (pill, InspectionSource.mvtec_ad, "screw/test/good/001.png", "good", False, PASS, _at(30)),
            (pill, InspectionSource.upload, f"{pill.id}/{tag}a.png", "defective", True, FAIL, _at(30)),
            (bare, InspectionSource.upload, f"{bare.id}/{tag}b.png", None, None, None, _at(30)),
            (pill, InspectionSource.mvtec_ad, "not-a-dataset-path.png", None, None, NOT_ASSESSED, _at(30)),
            (bare, InspectionSource.mvtec_ad, "bottle/test/good/000.png", "good", False, PASS, _at(20)),
            (pill, InspectionSource.mvtec_ad, "tile/test/good/000.png", "good", False, PASS, _at(15, 12, 2000)),
        ]
        for product, source, path, prediction, review, decision, created in rows:
            db.add(Inspection(product_id=product.id, image_path=path, source=source, ai_prediction=prediction,
                              review_required=review, quality_decision=decision, created_at=created,
                              inspection_date=created))
        db.commit()
        product_ids = [pill.id, bare.id]
    yield
    with SessionLocal() as db:
        db.execute(delete(Inspection).where(Inspection.product_id.in_(product_ids)))
        db.execute(delete(Product).where(Product.id.in_(product_ids)))
        db.commit()


def _by_name(result) -> dict:
    return {row.category: row for row in result.categories}


def _zero(row) -> bool:
    return (row.total, row.ai_analysed, row.ai_defective, row.ai_good, row.manual_review, row.pass_count,
            row.fail_count, row.manual_review_decisions, row.not_assessed, row.defect_rate) == (0,) * 9 + (None,)


# ---------------------------------------------------------------------------
# Exact counts (pinned 2001 window)
# ---------------------------------------------------------------------------

def test_counts_resolution_and_defect_rate_for_7_days(seeded_2001, monkeypatch):
    _pin_today(monkeypatch, _at(31))
    with SessionLocal() as db:
        result = get_category_analytics(db, window_days=7)
    rows = _by_name(result)
    assert result.window_days == 7
    assert set(rows) == set(MVTEC_CATEGORIES) | {UNCATEGORISED}

    screw = rows["screw"]  # imports take the dataset path's category, not the product's (pill)
    assert (screw.total, screw.ai_analysed, screw.ai_defective, screw.ai_good) == (2, 2, 1, 1)
    assert screw.defect_rate == 0.5
    assert (screw.manual_review, screw.pass_count, screw.fail_count, screw.manual_review_decisions,
            screw.not_assessed) == (1, 1, 0, 1, 0)

    pill = rows["pill"]  # the upload takes its product's category
    assert (pill.total, pill.ai_analysed, pill.ai_defective, pill.defect_rate) == (1, 1, 1, 1.0)
    assert (pill.manual_review, pill.fail_count) == (1, 1)

    uncategorised = rows[UNCATEGORISED]  # an upload without product category + an unparseable import path
    assert (uncategorised.total, uncategorised.ai_analysed, uncategorised.defect_rate) == (2, 0, None)
    assert uncategorised.not_assessed == 2  # a NULL decision counts as not assessed, like the summary

    assert all(_zero(rows[c]) for c in MVTEC_CATEGORIES if c not in ("screw", "pill"))


def test_window_filtering(seeded_2001, monkeypatch):
    _pin_today(monkeypatch, _at(31))
    with SessionLocal() as db:
        seven, fourteen, thirty = (_by_name(get_category_analytics(db, window_days=d)) for d in (7, 14, 30))
    assert seven["bottle"].total == 0  # 2001-01-20 is 11 days back
    assert fourteen["bottle"].total == thirty["bottle"].total == 1
    assert fourteen["bottle"].defect_rate == 0.0
    assert seven["tile"].total == fourteen["tile"].total == thirty["tile"].total == 0  # 2000-12-15 is outside 30 days
    assert sum(row.total for row in thirty.values()) == 6


def test_ordering_total_desc_then_name(seeded_2001, monkeypatch):
    _pin_today(monkeypatch, _at(31))
    with SessionLocal() as db:
        names = [row.category for row in get_category_analytics(db, window_days=14).categories]
    zero = sorted(set(MVTEC_CATEGORIES) - {"screw", "uncategorised", "bottle", "pill"})
    assert names == ["screw", UNCATEGORISED, "bottle", "pill"] + zero  # 2, 2 (name), 1, 1 (name), then 0s by name


def test_all_15_categories_and_no_uncategorised_when_not_needed(seeded_2001, monkeypatch):
    _pin_today(monkeypatch, _at(25))  # window 2001-01-19..25: only the bottle import
    with SessionLocal() as db:
        result = get_category_analytics(db, window_days=7)
    assert [row.category for row in result.categories] == ["bottle"] + sorted(set(MVTEC_CATEGORIES) - {"bottle"})
    _pin_today(monkeypatch, _at(1, 6, 1999))  # nothing at all in the window
    with SessionLocal() as db:
        empty = get_category_analytics(db, window_days=30)
    assert [row.category for row in empty.categories] == sorted(MVTEC_CATEGORIES)
    assert all(_zero(row) for row in empty.categories)


def test_endpoint_returns_the_same_through_the_api(seeded_2001, monkeypatch, client, supervisor_headers):
    _pin_today(monkeypatch, _at(31))
    response = client.get(URL, params={"days": 7}, headers=supervisor_headers)  # a supervisor may read it
    assert response.status_code == 200
    body = response.json()
    assert body["window_days"] == 7
    assert all(set(row) == ROW_KEYS for row in body["categories"])
    first = body["categories"][0]
    assert first == {"category": "screw", "total": 2, "ai_analysed": 2, "ai_defective": 1, "ai_good": 1,
                     "defect_rate": 0.5, "manual_review": 1, "pass_count": 1, "fail_count": 0,
                     "manual_review_decisions": 1, "not_assessed": 0}
    by_name = {row["category"]: row for row in body["categories"]}
    assert by_name[UNCATEGORISED]["defect_rate"] is None


# ---------------------------------------------------------------------------
# Category resolution is the AI path's rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source, path, product_category, expected", [
    (InspectionSource.mvtec_ad, "screw/test/good/000.png", "pill", "screw"),
    (InspectionSource.mvtec_ad, "screw/test/good/000.png", None, "screw"),
    (InspectionSource.mvtec_ad, "not-a-dataset-path.png", "pill", None),
    (InspectionSource.mvtec_ad, "/test/good/000.png", "pill", None),
    (InspectionSource.upload, "7/abc.png", "pill", "pill"),
    (InspectionSource.upload, "7/abc.png", None, None),
])
def test_resolve_category_matches_the_inspection_wrapper(source, path, product_category, expected):
    assert resolve_category(source, path, product_category) == expected
    product = Product(product_name="x", product_code="x", category=product_category)
    inspection = Inspection(product_id=1, image_path=path, source=source)
    inspection.product = product
    assert _resolve_category(inspection) == expected


# ---------------------------------------------------------------------------
# Endpoint contract
# ---------------------------------------------------------------------------

def test_requires_a_token(client):
    assert client.get(URL).status_code == 401


@pytest.mark.parametrize("days", [7, 14, 30])
def test_allowed_windows_for_both_roles(client, qe_headers, supervisor_headers, days):
    for headers in (qe_headers, supervisor_headers):
        response = client.get(URL, params={"days": days}, headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert body["window_days"] == days
        names = [row["category"] for row in body["categories"]]
        assert set(MVTEC_CATEGORIES) <= set(names) and set(names) <= set(MVTEC_CATEGORIES) | {UNCATEGORISED}
        keys = [(-row["total"], row["category"]) for row in body["categories"]]
        assert keys == sorted(keys)
        for row in body["categories"]:
            assert row["ai_defective"] + row["ai_good"] <= row["ai_analysed"] <= row["total"]
            decisions = row["pass_count"] + row["fail_count"] + row["manual_review_decisions"] + row["not_assessed"]
            assert decisions == row["total"]
            expected = row["ai_defective"] / row["ai_analysed"] if row["ai_analysed"] else None
            assert row["defect_rate"] == expected


def test_default_window_is_the_summarys(client, qe_headers):
    body = client.get(URL, headers=qe_headers).json()
    assert body["window_days"] == analytics.TREND_WINDOW_DAYS == 14


@pytest.mark.parametrize("days", ["0", "1", "15", "31", "abc"])
def test_invalid_days_are_refused_like_the_summary(client, qe_headers, days):
    response = client.get(URL, params={"days": days}, headers=qe_headers)
    assert response.status_code == 422
    assert client.get(SUMMARY_URL, params={"days": days}, headers=qe_headers).status_code == 422


def test_summary_keys_are_unchanged(client, qe_headers):
    body = client.get(SUMMARY_URL, headers=qe_headers).json()
    assert list(body) == [
        "total_inspections", "by_status", "by_source", "ai_analyzed_count", "ai_prediction_counts", "ai_defect_rate",
        "avg_reconstruction_error", "activity_by_day", "by_product", "defect_categories", "quality_decisions",
        "severity_distribution", "operational_insights", "trend_monitoring", "performance", "manual_review_count",
        "automation_rate", "automation_counts",
    ]


def test_schema_documents_defect_rate(client):
    schema = client.get("/openapi.json").json()
    row = schema["components"]["schemas"]["CategoryAnalyticsRow"]["properties"]
    assert "ai_defective / ai_analysed" in row["defect_rate"]["description"]
    assert "not ground truth" in row["defect_rate"]["description"]
    operation = schema["paths"][URL]["get"]
    assert "not ground truth" in operation["description"]

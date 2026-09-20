"""Milestone 4: inspection performance instrumentation and the analytics built on it.

Covers: the two nullable timing columns (and that nothing was backfilled), persistence of the
measured times through the real upload/import endpoints, the `performance` block of
GET /inspections/analytics/summary, the selectable `days` window, and the optional `limit` on
GET /inspections.

Two styles are used deliberately, because the test database is a real, shared PostgreSQL
instance that every other test file also writes to (see test_inspection_analytics.py):

  * Exact-value SQL aggregation tests run in an ISOLATED session whose writes are never
    committed (always rolled back) and whose rows are dated in the year 2001, with `today`
    passed explicitly. No other test or real data touches that period, so exact averages,
    medians and window boundaries can be asserted deterministically without seeding anything.
  * API-level tests compare a "before" and "after" snapshot and assert the delta, exactly like
    the existing analytics tests.
"""

import time
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from app.ai.inference import PredictionResult
from app.database.session import SessionLocal, engine
from app.inspections.analytics import (
    ALLOWED_WINDOW_DAYS,
    _get_performance_metrics,
    get_inspection_analytics_summary,
)
from app.inspections.service import record_processing_time, run_ai_inference
from app.models.inspection import Inspection, InspectionSource
from app.models.product import Product
from tests.conftest import make_image_bytes

ANALYTICS_URL = "/inspections/analytics/summary"
TIMING_COLUMNS = ("processing_time_ms", "ai_inference_time_ms")

# A period no other test or real data ever writes to - see the module docstring.
ISOLATED_TODAY = date(2001, 1, 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db():
    """A session whose writes are visible to itself but are never committed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


def _make_product(db):
    product = Product(product_name="Isolated Performance Product", product_code=f"ISO-{uuid4().hex[:12]}")
    db.add(product)
    db.flush()
    return product


def _add_inspection(db, product, day, *, processing_ms=None, ai_ms=None, ai_prediction=None):
    inspection = Inspection(
        product_id=product.id,
        image_path="isolated/none/none/000.png",
        source=InspectionSource.upload,
        created_at=datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc),
        processing_time_ms=processing_ms,
        ai_inference_time_ms=ai_ms,
        ai_prediction=ai_prediction,
    )
    db.add(inspection)
    return inspection


class _FakeSession:
    """Minimal stand-in for a SQLAlchemy Session - only what record_processing_time calls."""

    def __init__(self, fail_commit=False):
        self.fail_commit = fail_commit
        self.committed = False
        self.rolled_back = False
        self.refreshed = None

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("simulated commit failure")
        self.committed = True

    def refresh(self, obj):
        self.refreshed = obj

    def rollback(self):
        self.rolled_back = True


def _summary(client, headers, days=None):
    url = ANALYTICS_URL if days is None else f"{ANALYTICS_URL}?days={days}"
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    return response.json()


def _upload_plain_inspection(client, headers, product_id):
    """A user-uploaded image: no derivable MVTec category, so no AI prediction is attempted."""
    response = client.post(
        "/inspections/upload",
        headers=headers,
        data={"product_id": str(product_id)},
        files={"file": ("sample.png", make_image_bytes("PNG"), "image/png")},
    )
    assert response.status_code == 201
    return response.json()


def _import_bottle(client, headers, product_id, monkeypatch, *, ai_time_ms=12.3, fail_ai=False):
    if fail_ai:

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated inference crash")

        monkeypatch.setattr("app.inspections.service.predict_image", _raise)
    else:
        fake_result = PredictionResult(
            category="bottle",
            prediction="defective",
            reconstruction_error=0.0041,
            threshold=0.003212,
            model_name="autoencoder",
            input_size=(128, 128),
            processing_time_ms=ai_time_ms,
        )
        monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake_result)

    response = client.post(
        "/inspections/import",
        headers=headers,
        json={
            "product_id": product_id,
            "category": "bottle",
            "split": "test",
            "defect_type": "broken_large",
            "filename": "000.png",
        },
    )
    assert response.status_code == 201
    return response.json()


# ---------------------------------------------------------------------------
# Schema: nullable columns, and nothing fabricated for historical rows
# ---------------------------------------------------------------------------


def test_inspection_model_has_nullable_timing_columns():
    columns = {c.name: c for c in sa_inspect(Inspection).columns}
    for name in TIMING_COLUMNS:
        assert name in columns
        assert columns[name].nullable, f"{name} must be nullable"


def test_database_timing_columns_are_nullable_with_no_default():
    database_columns = {c["name"]: c for c in sa_inspect(engine).get_columns("inspections")}
    for name in TIMING_COLUMNS:
        assert database_columns[name]["nullable"] is True
        assert database_columns[name]["default"] is None  # no server default -> nothing can be backfilled


def test_new_inspection_row_defaults_timings_to_null():
    inspection = Inspection(product_id=1, image_path="bottle/test/good/000.png", source=InspectionSource.mvtec_ad)
    assert inspection.processing_time_ms is None
    assert inspection.ai_inference_time_ms is None


def test_row_inserted_without_timings_is_stored_as_null(isolated_db):
    """A row written the way every pre-migration inspection was (no timing columns given) must
    read back NULL - proving the database itself supplies no default value."""
    product = _make_product(isolated_db)
    isolated_db.execute(
        text(
            "INSERT INTO inspections (product_id, image_path, status, source) "
            "VALUES (:pid, 'isolated/none/none/000.png', 'pending', 'upload')"
        ),
        {"pid": product.id},
    )
    row = isolated_db.execute(
        text(
            "SELECT processing_time_ms, ai_inference_time_ms FROM inspections "
            "WHERE product_id = :pid"
        ),
        {"pid": product.id},
    ).one()
    assert row.processing_time_ms is None
    assert row.ai_inference_time_ms is None


# ---------------------------------------------------------------------------
# Service layer: what gets recorded, and when it must not be
# ---------------------------------------------------------------------------


def test_record_processing_time_measures_elapsed_milliseconds():
    inspection = Inspection(product_id=1, image_path="x", source=InspectionSource.upload)
    session = _FakeSession()

    record_processing_time(inspection, session, time.perf_counter() - 0.25)

    assert 250 <= inspection.processing_time_ms < 5000  # >= the 0.25s we simulated; generous upper bound
    assert session.committed is True
    assert session.refreshed is inspection


def test_record_processing_time_never_raises_when_the_commit_fails():
    inspection = Inspection(product_id=1, image_path="x", source=InspectionSource.upload)
    session = _FakeSession(fail_commit=True)

    record_processing_time(inspection, session, time.perf_counter())  # must not raise

    assert session.rolled_back is True
    assert session.committed is False


def test_run_ai_inference_persists_the_reported_inference_time(monkeypatch):
    fake_result = PredictionResult(
        category="bottle",
        prediction="good",
        reconstruction_error=0.001,
        threshold=0.003,
        model_name="autoencoder",
        processing_time_ms=845.5,
    )
    monkeypatch.setattr("app.inspections.service.predict_image", lambda *a, **k: fake_result)
    monkeypatch.setattr(
        "app.inspections.service._resolve_absolute_path", lambda inspection: __import__("pathlib").Path("unused")
    )
    inspection = Inspection(
        id=9, product_id=1, image_path="bottle/test/good/000.png", source=InspectionSource.mvtec_ad
    )

    run_ai_inference(inspection, _FakeSession())

    assert inspection.ai_inference_time_ms == 845.5


def test_run_ai_inference_failure_leaves_the_inference_time_null(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr("app.inspections.service.predict_image", _raise)
    monkeypatch.setattr(
        "app.inspections.service._resolve_absolute_path", lambda inspection: __import__("pathlib").Path("unused")
    )
    inspection = Inspection(
        id=10, product_id=1, image_path="bottle/test/good/000.png", source=InspectionSource.mvtec_ad
    )

    run_ai_inference(inspection, _FakeSession())

    assert inspection.ai_inference_time_ms is None


def test_run_ai_inference_no_category_leaves_the_inference_time_null():
    inspection = Inspection(id=11, product_id=1, image_path="1/abc.png", source=InspectionSource.upload)

    run_ai_inference(inspection, _FakeSession())

    assert inspection.ai_inference_time_ms is None


# ---------------------------------------------------------------------------
# Persistence through the real endpoints
# ---------------------------------------------------------------------------


def test_import_persists_both_times_and_they_survive_a_reload(client, qe_headers, test_product, monkeypatch):
    created = _import_bottle(client, qe_headers, test_product["id"], monkeypatch, ai_time_ms=12.3)

    assert created["ai_inference_time_ms"] == 12.3
    assert isinstance(created["processing_time_ms"], float)
    assert created["processing_time_ms"] > 0

    reloaded = client.get(f"/inspections/{created['id']}", headers=qe_headers).json()
    assert reloaded["ai_inference_time_ms"] == 12.3
    assert reloaded["processing_time_ms"] == created["processing_time_ms"]


def test_upload_records_processing_time_but_no_ai_time(client, qe_headers, test_product):
    created = _upload_plain_inspection(client, qe_headers, test_product["id"])

    assert isinstance(created["processing_time_ms"], float)
    assert created["processing_time_ms"] > 0
    assert created["ai_prediction"] is None
    assert created["ai_inference_time_ms"] is None  # no AI ran, so there is nothing to time


def test_failed_ai_inference_still_records_processing_time_but_no_ai_time(
    client, qe_headers, test_product, monkeypatch
):
    created = _import_bottle(client, qe_headers, test_product["id"], monkeypatch, fail_ai=True)

    assert created["ai_prediction"] is None
    assert created["ai_inference_time_ms"] is None
    assert isinstance(created["processing_time_ms"], float)  # the handler still did measurable work


def test_timings_appear_in_the_list_endpoint_too(client, qe_headers, test_product):
    created = _upload_plain_inspection(client, qe_headers, test_product["id"])

    listing = client.get("/inspections?limit=1", headers=qe_headers).json()

    assert listing[0]["id"] == created["id"]
    assert listing[0]["processing_time_ms"] == created["processing_time_ms"]


def test_image_path_is_still_never_exposed(client, qe_headers, test_product):
    created = _upload_plain_inspection(client, qe_headers, test_product["id"])

    assert "image_path" not in created


def test_report_is_unchanged_and_carries_no_timing_fields(client, qe_headers, test_product, monkeypatch):
    created = _import_bottle(client, qe_headers, test_product["id"], monkeypatch)

    report = client.get(f"/inspections/{created['id']}/report", headers=qe_headers).json()

    assert set(report.keys()) == {"inspection", "dataset", "defect", "severity", "quality", "report_summary"}
    assert set(report["report_summary"].keys()) == {"overall_result", "report_status", "summary"}
    serialized = str(report)
    assert "processing_time_ms" not in serialized
    assert "ai_inference_time_ms" not in serialized


# ---------------------------------------------------------------------------
# Analytics: exact SQL aggregation (isolated, rolled-back, dated in 2001)
# ---------------------------------------------------------------------------


def test_performance_metrics_exact_values(isolated_db):
    product = _make_product(isolated_db)
    day = ISOLATED_TODAY - timedelta(days=1)
    for ms in (100.0, 200.0, 300.0, 400.0):
        _add_inspection(isolated_db, product, day, processing_ms=ms)
    isolated_db.flush()

    performance = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7)

    stats = performance.processing_time
    assert stats.count == 4
    assert stats.avg_ms == pytest.approx(250.0)
    assert stats.median_ms == pytest.approx(250.0)  # even count: mean of the two middle values
    assert stats.min_ms == 100.0
    assert stats.max_ms == 400.0
    assert performance.inspections_in_window == 4
    assert performance.window_days == 7


def test_median_of_an_odd_count_is_the_middle_value_regardless_of_insert_order(isolated_db):
    product = _make_product(isolated_db)
    for ms in (9.0, 1.0, 5.0):
        _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=ms)
    isolated_db.flush()

    stats = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7).processing_time

    assert stats.median_ms == pytest.approx(5.0)
    assert stats.avg_ms == pytest.approx(5.0)


def test_untimed_inspections_are_ignored_not_read_as_zero(isolated_db):
    product = _make_product(isolated_db)
    _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=100.0)
    _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=300.0)
    for _ in range(3):  # historical-style rows with no measurement
        _add_inspection(isolated_db, product, ISOLATED_TODAY)
    isolated_db.flush()

    performance = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7)

    assert performance.inspections_in_window == 5
    assert performance.processing_time.count == 2  # only the two real measurements
    assert performance.processing_time.avg_ms == pytest.approx(200.0)  # NOT diluted to 80 by the NULL rows
    assert performance.processing_time.min_ms == 100.0


def test_no_timed_inspections_reports_none_never_zero(isolated_db):
    product = _make_product(isolated_db)
    for _ in range(3):
        _add_inspection(isolated_db, product, ISOLATED_TODAY)
    isolated_db.flush()

    performance = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7)

    for stats in (performance.processing_time, performance.ai_inference_time):
        assert stats.count == 0
        assert stats.avg_ms is None
        assert stats.median_ms is None
        assert stats.min_ms is None
        assert stats.max_ms is None


def test_empty_window_reports_zero_counts_and_no_rate(isolated_db):
    performance = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7)

    assert performance.inspections_in_window == 0
    assert performance.ai_analyzed_in_window == 0
    assert performance.ai_analyzed_rate is None  # a rate over zero inspections is undefined, not 0%
    assert performance.processing_time.count == 0


def test_a_measured_zero_is_a_real_value_not_missing(isolated_db):
    product = _make_product(isolated_db)
    _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=0.0)
    isolated_db.flush()

    stats = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7).processing_time

    assert stats.count == 1
    assert stats.min_ms == 0.0  # 0.0 is a measurement; only None means "not measured"
    assert stats.avg_ms == 0.0


def test_ai_inference_time_is_aggregated_independently_of_processing_time(isolated_db):
    product = _make_product(isolated_db)
    _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=1000.0, ai_ms=10.0, ai_prediction="good")
    _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=3000.0, ai_ms=30.0, ai_prediction="defective")
    _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=500.0)  # e.g. an upload: no AI
    isolated_db.flush()

    performance = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7)

    assert performance.processing_time.count == 3
    assert performance.ai_inference_time.count == 2
    assert performance.ai_inference_time.avg_ms == pytest.approx(20.0)
    assert performance.ai_inference_time.median_ms == pytest.approx(20.0)
    assert performance.ai_inference_time.min_ms == 10.0
    assert performance.ai_inference_time.max_ms == 30.0
    assert performance.ai_analyzed_in_window == 2
    assert performance.inspections_in_window == 3
    assert performance.ai_analyzed_rate == pytest.approx(2 / 3)


def test_window_boundaries_are_applied_in_sql(isolated_db):
    product = _make_product(isolated_db)
    _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=1.0)  # day 0
    _add_inspection(isolated_db, product, ISOLATED_TODAY - timedelta(days=6), processing_ms=2.0)  # oldest day in a 7-day window
    _add_inspection(isolated_db, product, ISOLATED_TODAY - timedelta(days=7), processing_ms=4.0)  # just outside 7, inside 14
    _add_inspection(isolated_db, product, ISOLATED_TODAY - timedelta(days=13), processing_ms=8.0)  # oldest day in a 14-day window
    _add_inspection(isolated_db, product, ISOLATED_TODAY - timedelta(days=14), processing_ms=16.0)  # just outside 14, inside 30
    _add_inspection(isolated_db, product, ISOLATED_TODAY + timedelta(days=1), processing_ms=32.0)  # the future: never included
    isolated_db.flush()

    seven = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7)
    fourteen = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 14)
    thirty = _get_performance_metrics(isolated_db, ISOLATED_TODAY, 30)

    assert seven.processing_time.count == 2 and seven.processing_time.max_ms == 2.0
    assert fourteen.processing_time.count == 4 and fourteen.processing_time.max_ms == 8.0
    assert thirty.processing_time.count == 5 and thirty.processing_time.max_ms == 16.0


# ---------------------------------------------------------------------------
# Analytics: no per-row queries
# ---------------------------------------------------------------------------


def _count_statements(fn):
    statements = []

    def _before(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", _before)
    try:
        fn()
    finally:
        event.remove(engine, "before_cursor_execute", _before)
    return len(statements)


def test_performance_metrics_use_exactly_one_query(isolated_db):
    product = _make_product(isolated_db)
    for i in range(20):
        _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=float(i))
    isolated_db.flush()

    assert _count_statements(lambda: _get_performance_metrics(isolated_db, ISOLATED_TODAY, 7)) == 1


def test_summary_query_count_does_not_grow_with_the_number_of_inspections(isolated_db):
    product = _make_product(isolated_db)
    isolated_db.flush()

    before = _count_statements(lambda: get_inspection_analytics_summary(isolated_db))
    for i in range(60):
        _add_inspection(isolated_db, product, ISOLATED_TODAY, processing_ms=float(i), ai_ms=1.0, ai_prediction="good")
    isolated_db.flush()
    after = _count_statements(lambda: get_inspection_analytics_summary(isolated_db))

    assert after == before  # aggregation happens in SQL - no N+1, no per-inspection queries


# ---------------------------------------------------------------------------
# Analytics API: shape, window, validation, deltas
# ---------------------------------------------------------------------------


def test_performance_block_shape(client, qe_headers):
    performance = _summary(client, qe_headers)["performance"]

    assert set(performance.keys()) == {
        "window_days",
        "inspections_in_window",
        "ai_analyzed_in_window",
        "ai_analyzed_rate",
        "processing_time",
        "ai_inference_time",
    }
    for name in ("processing_time", "ai_inference_time"):
        assert set(performance[name].keys()) == {"count", "avg_ms", "median_ms", "min_ms", "max_ms"}
    assert performance["ai_analyzed_rate"] is None or 0 <= performance["ai_analyzed_rate"] <= 1


def test_existing_summary_fields_are_all_still_present(client, qe_headers):
    body = _summary(client, qe_headers)

    previous_top_level_keys = {
        "total_inspections", "by_status", "by_source", "ai_analyzed_count", "ai_prediction_counts",
        "ai_defect_rate", "avg_reconstruction_error", "activity_by_day", "by_product",
        "defect_categories", "quality_decisions", "severity_distribution", "operational_insights",
        "trend_monitoring",
    }
    assert previous_top_level_keys.issubset(body.keys())
    assert set(body.keys()) == previous_top_level_keys | {"performance"}


def test_default_window_is_identical_to_requesting_14_days(client, qe_headers):
    default = _summary(client, qe_headers)
    explicit = _summary(client, qe_headers, days=14)

    assert default["trend_monitoring"] == explicit["trend_monitoring"]
    assert default["performance"]["window_days"] == 14
    assert default["by_product"] == explicit["by_product"]


@pytest.mark.parametrize("days", ALLOWED_WINDOW_DAYS)
def test_days_selects_the_trend_and_performance_window_only(client, qe_headers, days):
    body = _summary(client, qe_headers, days=days)

    assert body["trend_monitoring"]["period_days"] == days
    assert len(body["trend_monitoring"]["daily"]) == days
    assert body["performance"]["window_days"] == days
    for trend in body["trend_monitoring"]["category_trends"]:
        assert len(trend["daily"]) == days
    # The overview activity series is a fixed 14-day chart and does not follow the window.
    assert len(body["activity_by_day"]) == 14


def test_a_longer_window_never_contains_fewer_inspections(client, qe_headers):
    counts = [_summary(client, qe_headers, days=d)["performance"]["inspections_in_window"] for d in (7, 14, 30)]

    assert counts == sorted(counts)


@pytest.mark.parametrize("bad_days", ["0", "-1", "5", "31", "100", "abc", "7.5"])
def test_unsupported_days_are_rejected(client, qe_headers, bad_days):
    response = client.get(f"{ANALYTICS_URL}?days={bad_days}", headers=qe_headers)

    assert response.status_code == 422


def test_unsupported_days_error_names_the_allowed_values(client, qe_headers):
    response = client.get(f"{ANALYTICS_URL}?days=5", headers=qe_headers)

    assert response.status_code == 422
    assert "7, 14, 30" in response.json()["detail"]


def test_get_inspection_analytics_summary_rejects_unsupported_window_directly(isolated_db):
    with pytest.raises(ValueError):
        get_inspection_analytics_summary(isolated_db, window_days=5)


def test_import_increments_timed_and_ai_analyzed_counts(client, qe_headers, test_product, monkeypatch):
    before = _summary(client, qe_headers)["performance"]

    _import_bottle(client, qe_headers, test_product["id"], monkeypatch, ai_time_ms=42.0)

    after = _summary(client, qe_headers)["performance"]
    assert after["inspections_in_window"] == before["inspections_in_window"] + 1
    assert after["ai_analyzed_in_window"] == before["ai_analyzed_in_window"] + 1
    assert after["processing_time"]["count"] == before["processing_time"]["count"] + 1
    assert after["ai_inference_time"]["count"] == before["ai_inference_time"]["count"] + 1
    for name in ("processing_time", "ai_inference_time"):
        stats = after[name]
        assert stats["min_ms"] <= stats["median_ms"] <= stats["max_ms"]
        assert stats["min_ms"] <= stats["avg_ms"] <= stats["max_ms"]


def test_upload_increments_only_the_processing_time_count(client, qe_headers, test_product):
    before = _summary(client, qe_headers)["performance"]

    _upload_plain_inspection(client, qe_headers, test_product["id"])

    after = _summary(client, qe_headers)["performance"]
    assert after["processing_time"]["count"] == before["processing_time"]["count"] + 1
    assert after["ai_inference_time"]["count"] == before["ai_inference_time"]["count"]  # no AI ran
    assert after["ai_analyzed_in_window"] == before["ai_analyzed_in_window"]


# ---------------------------------------------------------------------------
# Optional `limit` on GET /inspections
# ---------------------------------------------------------------------------


def test_list_without_limit_returns_every_inspection_as_before(client, qe_headers):
    listing = client.get("/inspections", headers=qe_headers).json()

    assert len(listing) == _summary(client, qe_headers)["total_inspections"]


def test_limit_returns_only_the_newest_inspections_in_order(client, qe_headers, test_product):
    _upload_plain_inspection(client, qe_headers, test_product["id"])
    everything = client.get("/inspections", headers=qe_headers).json()

    limited = client.get("/inspections?limit=3", headers=qe_headers).json()

    assert len(limited) == 3
    assert [i["id"] for i in limited] == [i["id"] for i in everything[:3]]


def test_limit_larger_than_the_table_returns_everything(client, qe_headers):
    everything = client.get("/inspections", headers=qe_headers).json()

    limited = client.get("/inspections?limit=1000", headers=qe_headers).json()

    assert len(limited) == min(len(everything), 1000)


@pytest.mark.parametrize("bad_limit", ["0", "-5", "1001", "abc"])
def test_invalid_limit_is_rejected(client, qe_headers, bad_limit):
    assert client.get(f"/inspections?limit={bad_limit}", headers=qe_headers).status_code == 422


# ---------------------------------------------------------------------------
# Authorization (unchanged behavior, exercised with the new parameters)
# ---------------------------------------------------------------------------


def test_analytics_with_days_requires_authentication(client):
    assert client.get(f"{ANALYTICS_URL}?days=7").status_code == 401


def test_list_with_limit_requires_authentication(client):
    assert client.get("/inspections?limit=5").status_code == 401


def test_analytics_with_days_is_readable_by_a_factory_supervisor(client, supervisor_headers):
    assert client.get(f"{ANALYTICS_URL}?days=30", headers=supervisor_headers).status_code == 200


def test_list_with_limit_is_readable_by_a_factory_supervisor(client, supervisor_headers):
    assert client.get("/inspections?limit=5", headers=supervisor_headers).status_code == 200


def test_report_requires_authentication(client, qe_headers, test_product):
    created = _upload_plain_inspection(client, qe_headers, test_product["id"])

    assert client.get(f"/inspections/{created['id']}/report").status_code == 401


def test_report_is_readable_by_a_factory_supervisor(client, qe_headers, supervisor_headers, test_product):
    created = _upload_plain_inspection(client, qe_headers, test_product["id"])

    response = client.get(f"/inspections/{created['id']}/report", headers=supervisor_headers)

    assert response.status_code == 200
    assert response.json()["report_summary"]["report_status"] == "COMPLETE"


def test_a_factory_supervisor_still_cannot_create_inspections(client, supervisor_headers, test_product):
    response = client.post(
        "/inspections/upload",
        headers=supervisor_headers,
        data={"product_id": str(test_product["id"])},
        files={"file": ("sample.png", make_image_bytes("PNG"), "image/png")},
    )

    assert response.status_code == 403

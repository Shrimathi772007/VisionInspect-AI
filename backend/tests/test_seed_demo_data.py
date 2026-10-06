"""scripts/seed_demo_data.py - pure parts only (argument handling, deterministic image selection and ordering,
timestamp spreading, the --reset confirmation rule, password handling). Nothing here touches a database
(the one DB-reading test runs against the separate test database and changes nothing)."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "seed_demo_data.py"
_spec = importlib.util.spec_from_file_location("seed_demo_data_under_test", SCRIPT)
seed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed)

REQUIRED = ["--qe-email", "qe@example.com", "--qe-name", "QE", "--supervisor-email", "sup@example.com",
            "--supervisor-name", "Sup"]
FILES = {
    "good": [f"{i:03d}.png" for i in (5, 0, 3, 1, 2, 4)],
    "crack": ["001.png", "000.png", "002.png"],
    "oil": ["000.png"],
    "glue_strip": ["001.png", "000.png"],
}


# ---------------------------------------------------------------------------
# Arguments and passwords
# ---------------------------------------------------------------------------

def test_defaults_and_flags():
    args = seed.parse_args(REQUIRED)
    assert (args.days, args.per_category, args.reset, args.yes, args.dry_run) == (14, 8, False, False, False)
    args = seed.parse_args(REQUIRED + ["--days", "7", "--per-category", "4", "--reset", "--yes", "--dry-run"])
    assert (args.days, args.per_category, args.reset, args.yes, args.dry_run) == (7, 4, True, True, True)


@pytest.mark.parametrize("missing", ["--qe-email", "--qe-name", "--supervisor-email", "--supervisor-name"])
def test_required_arguments(missing, capsys):
    argv = list(REQUIRED)
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit):
        seed.parse_args(argv)


@pytest.mark.parametrize("bad", [["--days", "0"], ["--per-category", "0"], ["--days", "x"]])
def test_invalid_numbers_are_refused(bad):
    with pytest.raises(SystemExit):
        seed.parse_args(REQUIRED + bad)


@pytest.mark.parametrize("flag", ["--password", "--qe-password", "--supervisor-password", "--pw"])
def test_a_password_is_never_accepted_on_the_command_line(flag):
    with pytest.raises(SystemExit):
        seed.parse_args(REQUIRED + [flag, "secret123"])
    assert not any("password" in action.dest for action in seed.build_parser()._actions)


def test_password_from_environment_or_typed_twice(monkeypatch):
    monkeypatch.setenv(seed.QE_PASSWORD_ENV, "from-env-password")
    assert seed.password_from(seed.QE_PASSWORD_ENV, "qe", prompt=lambda _p: pytest.fail("must not prompt")) == "from-env-password"

    monkeypatch.delenv(seed.QE_PASSWORD_ENV)
    prompts = []
    answers = iter(["typed-password", "typed-password"])

    def fake_getpass(prompt):
        prompts.append(prompt)
        return next(answers)

    assert seed.password_from(seed.QE_PASSWORD_ENV, "qe@example.com", prompt=fake_getpass) == "typed-password"
    assert len(prompts) == 2 and all("typed-password" not in p for p in prompts)

    mismatched = iter(["one-password", "two-password"])
    with pytest.raises(SystemExit) as exc:
        seed.password_from(seed.QE_PASSWORD_ENV, "qe@example.com", prompt=lambda _p: next(mismatched))
    assert "one-password" not in str(exc.value) and "two-password" not in str(exc.value)


def test_existing_users_are_left_untouched_and_no_password_is_needed(monkeypatch):
    """With both accounts already present, the seed script must not ask for, read or use any password and
    must not modify the users (runs against the separate test database; the users are removed afterwards)."""
    from uuid import uuid4

    from sqlalchemy import delete, select

    from app.auth.security import hash_password
    from app.database import SessionLocal
    from app.models.user import User, UserRole

    def no_password(*_args, **_kwargs):
        raise AssertionError("a password was requested for an existing user")

    monkeypatch.setattr(seed, "password_from", no_password)
    monkeypatch.setattr(seed.getpass, "getpass", no_password)
    monkeypatch.delenv(seed.QE_PASSWORD_ENV, raising=False)
    monkeypatch.delenv(seed.SUPERVISOR_PASSWORD_ENV, raising=False)

    qe_email, sup_email = f"seed.qe.{uuid4().hex}@example.com", f"seed.sup.{uuid4().hex}@example.com"
    with SessionLocal() as db:
        db.add_all([
            User(name="Existing QE", email=qe_email, password_hash=hash_password("ExistingQE123"), role=UserRole.quality_engineer),
            User(name="Existing Sup", email=sup_email, password_hash=hash_password("ExistingSup123"), role=UserRole.factory_supervisor),
        ])
        db.commit()
        before = {u.email: (u.name, u.role, u.password_hash) for u in db.scalars(select(User).where(User.email.in_([qe_email, sup_email])))}
    try:
        args = seed.parse_args(["--qe-email", qe_email, "--qe-name", "Other Name", "--supervisor-email", sup_email,
                                "--supervisor-name", "Other Sup"])
        with SessionLocal() as db:
            result = seed.ensure_users(db, args, dry_run=False)
            assert result["qe"] == "kept existing user (role quality_engineer)"
            assert result["supervisor"] == "kept existing user (role factory_supervisor)"
            assert result["qe_user"].email == qe_email
        with SessionLocal() as db:
            after = {u.email: (u.name, u.role, u.password_hash) for u in db.scalars(select(User).where(User.email.in_([qe_email, sup_email])))}
        assert after == before  # names, roles and password hashes untouched
    finally:
        with SessionLocal() as db:
            db.execute(delete(User).where(User.email.in_([qe_email, sup_email])))
            db.commit()


@pytest.mark.parametrize("account", ["qe", "supervisor"])
def test_password_over_72_bytes_is_refused_before_any_user_is_created(monkeypatch, account):
    """bcrypt only uses 72 bytes: a 30-character password of 3-byte characters (90 bytes) gets the same
    BootstrapError as the API's 422, and no user is written (runs against the separate test database)."""
    from uuid import uuid4

    from sqlalchemy import delete, select

    from app.auth.bootstrap import BootstrapError
    from app.auth.security import hash_password
    from app.database import SessionLocal
    from app.models.user import User, UserRole

    too_long = "€" * 30
    assert len(too_long.encode("utf-8")) == 90
    qe_email, sup_email = f"seed.qe.{uuid4().hex}@example.com", f"seed.sup.{uuid4().hex}@example.com"
    if account == "supervisor":  # the QE already exists, so only the supervisor password is read
        with SessionLocal() as db:
            db.add(User(name="Existing QE", email=qe_email, password_hash=hash_password("ExistingQE123"),
                        role=UserRole.quality_engineer))
            db.commit()
    monkeypatch.setattr(seed, "password_from", lambda *_args, **_kwargs: too_long)
    try:
        args = seed.parse_args(["--qe-email", qe_email, "--qe-name", "QE", "--supervisor-email", sup_email,
                                "--supervisor-name", "Sup"])
        with SessionLocal() as db:
            with pytest.raises(BootstrapError) as exc:
                seed.ensure_users(db, args, dry_run=False)
        assert "at most 72 bytes" in str(exc.value)
        assert too_long not in str(exc.value)
        with SessionLocal() as db:
            created = db.scalars(select(User.email).where(User.email.in_([qe_email, sup_email]))).all()
        assert created == ([qe_email] if account == "supervisor" else [])
    finally:
        with SessionLocal() as db:
            db.execute(delete(User).where(User.email.in_([qe_email, sup_email])))
            db.commit()


def test_reset_needs_yes():
    assert seed.require_yes_for_reset(True, False) is False
    assert seed.require_yes_for_reset(True, True) is True
    assert seed.require_yes_for_reset(False, True) is False


def test_reset_without_yes_changes_nothing(capsys):
    """Runs main() against the (separate) test database: it prints the plan and refuses."""
    from sqlalchemy import func, select

    from app.database import SessionLocal
    from app.models.inspection import Inspection

    with SessionLocal() as db:
        before = db.scalar(select(func.count()).select_from(Inspection))
    assert seed.main(REQUIRED + ["--reset"]) == 1
    out = capsys.readouterr().out
    assert "Refusing to reset without --yes. Nothing was changed." in out
    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(Inspection)) == before


def test_invalid_email_is_refused_before_anything_happens(capsys):
    argv = ["--qe-email", "qe@demo.local", "--qe-name", "QE", "--supervisor-email", "sup@example.com", "--supervisor-name", "S"]
    assert seed.main(argv) == 2
    assert "Invalid email address" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Deterministic image selection and ordering
# ---------------------------------------------------------------------------

def test_selection_is_deterministic_balanced_and_alternating():
    first = seed.select_images(FILES, 8)
    assert first == seed.select_images({k: list(reversed(v)) for k, v in FILES.items()}, 8)  # input order irrelevant
    assert len(first) == 8
    kinds = ["good" if t == "good" else "defect" for t, _ in first]
    assert kinds == ["good", "defect"] * 4
    assert [f for t, f in first if t == "good"] == ["000.png", "001.png", "002.png", "003.png"]
    # defects: round-robin over sorted types (crack, glue_strip, oil), each taking its next sorted file
    assert [(t, f) for t, f in first if t != "good"] == [
        ("crack", "000.png"), ("glue_strip", "000.png"), ("oil", "000.png"), ("crack", "001.png"),
    ]


@pytest.mark.parametrize("per_category", [1, 3, 7, 12])
def test_selection_sizes_and_uniqueness(per_category):
    picks = seed.select_images(FILES, per_category)
    available = sum(len(v) for v in FILES.values())
    assert len(picks) == min(per_category, available)
    assert len(set(picks)) == len(picks)
    goods = sum(t == "good" for t, _ in picks)
    assert goods >= len(picks) / 2  # roughly half good (ceil), more only when defects run out


def test_selection_tops_up_when_one_side_runs_out():
    only_one_defect = {"good": ["0.png", "1.png", "2.png", "3.png"], "crack": ["0.png"]}
    picks = seed.select_images(only_one_defect, 4)
    assert sorted(picks) == [("crack", "0.png"), ("good", "0.png"), ("good", "1.png"), ("good", "2.png")]
    assert seed.select_images({"good": ["0.png"]}, 3) == [("good", "0.png")]


def test_interleave_mixes_categories_and_kinds():
    plan = {c: [(c, "g0"), (c, "d0"), (c, "g1"), (c, "d1")] for c in ("a", "b", "c")}
    order = seed.interleave(plan)
    assert order[:3] == [("a", "g0"), ("b", "d0"), ("c", "g0")]  # every 2nd category starts with a defect
    assert sorted(order) == sorted(item for items in plan.values() for item in items)


# ---------------------------------------------------------------------------
# Timestamp spreading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("count, days", [(1, 1), (124, 14), (10, 3), (500, 30)])
def test_timestamps_are_monotonic_and_within_the_window(count, days):
    now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    stamps = seed.spread_timestamps(count, days, now)
    assert len(stamps) == count
    assert all(a < b for a, b in zip(stamps, stamps[1:]))
    assert all(now - timedelta(days=days) < s < now for s in stamps)
    assert stamps == seed.spread_timestamps(count, days, now)  # deterministic


def test_timestamps_cover_most_days():
    now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
    stamps = seed.spread_timestamps(124, 14, now)
    assert len({s.date() for s in stamps}) >= 14
    assert seed.spread_timestamps(0, 14, now) == []


def test_demo_product_naming():
    assert seed.demo_product_code("metal_nut") == "DEMO-METAL_NUT-001"
    assert seed.demo_product_name("metal_nut") == "Metal Nut Demo Product"


# ---------------------------------------------------------------------------
# --reset-ids
# ---------------------------------------------------------------------------

def test_reset_ids_flags():
    args = seed.parse_args(REQUIRED)
    assert (args.reset_ids, args.extra_uploads) == (False, 25)
    args = seed.parse_args(REQUIRED + ["--reset", "--reset-ids", "--yes", "--extra-uploads", "0"])
    assert (args.reset, args.reset_ids, args.yes, args.extra_uploads) == (True, True, True, 0)
    with pytest.raises(SystemExit):
        seed.parse_args(REQUIRED + ["--reset-ids", "--yes"])  # --reset-ids needs --reset
    with pytest.raises(SystemExit):
        seed.parse_args(REQUIRED + ["--extra-uploads", "-1"])


def test_reset_ids_without_yes_changes_nothing(capsys):
    """Runs main() against the (separate) test database: it prints the plan (users never touched) and refuses;
    users (hashes included), products, inspections and both id sequences are unchanged."""
    from sqlalchemy import text

    from app.database import SessionLocal

    def snapshot():
        with SessionLocal() as db:
            return {
                "users": db.execute(text("SELECT id, name, email, role, password_hash FROM users ORDER BY id")).all(),
                "products": db.execute(text("SELECT id, product_code FROM products ORDER BY id")).all(),
                "inspections": db.execute(text("SELECT count(*), max(id) FROM inspections")).one(),
                "sequences": [db.execute(text(f"SELECT last_value, is_called FROM {t}_id_seq")).one()
                              for t in ("inspections", "products", "users")],
            }

    before = snapshot()
    assert seed.main(REQUIRED + ["--reset", "--reset-ids", "--extra-uploads", "3"]) == 1
    out = capsys.readouterr().out
    assert "--reset-ids would also delete the" in out and "users are never touched" in out
    assert "Refusing to reset without --yes. Nothing was changed." in out
    assert snapshot() == before


class _Result:
    def __init__(self, value, rowcount):
        self._value, self.rowcount = value, rowcount

    def scalar(self):
        return self._value


class _RecordingSession:
    """Stands in for a Session: records every SQL statement; table counts come from `remaining`."""

    def __init__(self, remaining):
        self.statements, self.remaining, self.committed = [], remaining, False

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append((sql, params))
        if sql.startswith("SELECT count(*) FROM "):
            value = self.remaining[sql.rsplit(" ", 1)[1]]
        elif "pg_get_serial_sequence" in sql:
            value = f"public.{params['table']}_id_seq"
        else:
            value = None
        return _Result(value, rowcount=15)

    def commit(self):
        self.committed = True


def test_reset_ids_touches_only_demo_products_and_the_two_sequences():
    db = _RecordingSession({"inspections": 0, "products": 0})
    stats = seed.reset_ids(db)
    assert stats == {"demo_products": 15, "inspections_sequence": "restarted at 1", "products_sequence": "restarted at 1"}
    assert db.committed
    sql = " ".join(s for s, _ in db.statements).lower()
    for forbidden in ("users", "defects", "truncate", "drop", "alter", "update"):
        assert forbidden not in sql
    deletes = [(s, p) for s, p in db.statements if s.startswith("DELETE")]
    assert deletes == [("DELETE FROM products WHERE product_code LIKE :pattern", {"pattern": "DEMO-%"})]
    setvals = [(s, p["sequence"]) for s, p in db.statements if "setval" in s]
    assert [name for _, name in setvals] == ["public.inspections_id_seq", "public.products_id_seq"]
    assert all("1, false" in s for s, _ in setvals)


def test_reset_ids_keeps_a_sequence_whose_table_is_not_empty():
    db = _RecordingSession({"inspections": 0, "products": 2})
    stats = seed.reset_ids(db)
    assert stats["products_sequence"] == "kept (2 rows left)"
    assert [p["sequence"] for s, p in db.statements if "setval" in s] == ["public.inspections_id_seq"]


# ---------------------------------------------------------------------------
# Painted demo marks and the extra-upload plan
# ---------------------------------------------------------------------------

TRAIN_GOOD = {c: [f"{i:03d}.png" for i in range(30)] for c in (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper")}


def _png_bytes(image) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("size_class", sorted(seed.SIZE_CLASSES))
@pytest.mark.parametrize("size", [(800, 800), (840, 840), (900, 900), (1024, 1024)])
def test_painted_area_fraction_is_within_its_size_class(size_class, size):
    import random

    lo, hi = seed.SIZE_CLASSES[size_class]
    rng = random.Random(42)
    for _ in range(60):
        spec = seed.make_paint_spec(size_class, rng)
        assert spec.size_class == size_class and spec.shape in seed.SHAPES_BY_CLASS[size_class]
        assert len(spec.boxes) == (3 if size_class == "scattered_spots" else 1)
        assert all(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 for x0, y0, x1, y1 in spec.boxes)
        assert lo <= seed.painted_area_fraction(spec, size) <= hi


def test_size_classes_match_the_requested_sizes():
    assert seed.SIZE_CLASSES["small_spot"][0] < 0.01 < seed.SIZE_CLASSES["small_spot"][1]
    assert seed.SIZE_CLASSES["medium_patch"] == (0.05, 0.08)
    assert seed.SIZE_CLASSES["large_block"] == (0.15, 0.25)
    assert seed.SHAPES_BY_CLASS["linear_stripe"] == ("bar",)


def test_linear_stripes_are_long_and_thin():
    import random

    rng = random.Random(42)
    for _ in range(50):
        (x0, y0, x1, y1), = seed.make_paint_spec("linear_stripe", rng).boxes
        w, h = x1 - x0, y1 - y0
        assert max(w, h) / min(w, h) >= 10


@pytest.mark.parametrize("mode, colour, black", [("RGB", (200, 180, 160), (0, 0, 0)), ("L", 200, 0)])
def test_paint_defect_is_deterministic_and_only_blackens_the_mark(mode, colour, black):
    import random

    from PIL import Image

    image = Image.new(mode, (256, 256), colour)
    spec = seed.make_paint_spec("medium_patch", random.Random(42))
    assert spec == seed.make_paint_spec("medium_patch", random.Random(42))
    painted = seed.paint_defect(image, spec)
    assert _png_bytes(painted) == _png_bytes(seed.paint_defect(image, spec))
    assert (painted.mode, painted.size) == (mode, image.size)
    assert image.getcolors() == [(256 * 256, colour)]  # the source image is not modified
    mask = seed.paint_mask(spec, image.size)
    for xy in [(x, y) for x in range(0, 256, 8) for y in range(0, 256, 8)]:
        assert painted.getpixel(xy) == (black if mask.getpixel(xy) else colour)


def test_extra_upload_plan_is_deterministic_and_has_the_required_mix():
    plan = seed.plan_extra_uploads(25, TRAIN_GOOD)
    assert plan == seed.plan_extra_uploads(25, TRAIN_GOOD)
    assert plan == seed.plan_extra_uploads(25, {c: list(reversed(v)) for c, v in TRAIN_GOOD.items()})
    assert len(plan) == 25 and all(p.kind == "upload" and p.source_category == p.category for p in plan)
    painted = [p for p in plan if p.paint is not None]
    plain = [p for p in plan if p.paint is None]
    assert all(p.painted for p in painted) and not any(p.painted for p in plain)
    painted_categories = {p.category for p in painted}
    assert {"wood", "carpet", "screw", "pill"} <= painted_categories
    assert len(painted_categories - {"wood", "carpet", "screw", "pill"}) >= 10
    assert 4 <= len(plain) <= 6 and {"tile", "bottle"} <= {p.category for p in plain}
    assert len({p.category for p in plain}) >= 3
    assert {p.paint.size_class for p in painted} == set(seed.SIZE_CLASSES)  # every size class occurs
    assert all(p.filename in TRAIN_GOOD[p.category] for p in plan)  # train/good sources only
    picks = [(p.category, p.filename) for p in plan]
    assert len(set(picks)) == len(picks)  # no source image used twice
    assert seed.plan_extra_uploads(25, TRAIN_GOOD, seed=7) != plan  # the seed is what fixes the choices


@pytest.mark.parametrize("count", [0, 3, 40])
def test_extra_upload_plan_sizes(count):
    plan = seed.plan_extra_uploads(count, TRAIN_GOOD)
    assert len(plan) == count
    if count >= 4:
        assert {"wood", "carpet", "screw", "pill"} <= {p.category for p in plan if p.paint is not None}


def test_upload_labels():
    import random

    spec = seed.make_paint_spec("large_block", random.Random(42))
    item = seed.PlannedInspection("upload", "pill", filename="214.png", painted=True, source_category="pill", paint=spec)
    assert seed.upload_label(item) == "pill_train_good_214_painted_large_block_square.png"
    legacy = seed.PlannedInspection("upload", "tile", filename="000.png", painted=True, source_category="tile")
    assert seed.upload_label(legacy) == "tile_train_good_000_painted_bar.png"


# ---------------------------------------------------------------------------
# Timeline: imports and uploads interleaved, timestamps monotonic
# ---------------------------------------------------------------------------

def test_merge_timeline_spreads_uploads_and_keeps_order():
    imports = [("import", i) for i in range(120)]
    uploads = [("upload", j) for j in range(29)]
    order = seed.merge_timeline(imports, uploads)
    assert len(order) == 149
    assert [x for x in order if x[0] == "import"] == imports
    assert [x for x in order if x[0] == "upload"] == uploads
    positions = [i for i, x in enumerate(order) if x[0] == "upload"]
    assert all(4 <= b - a <= 6 for a, b in zip(positions, positions[1:]))  # evenly spaced
    assert seed.merge_timeline(imports, []) == imports
    assert seed.merge_timeline([], uploads) == uploads


@pytest.mark.parametrize("days", [7, 14])
def test_every_day_has_imports_and_uploads(days):
    now = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
    order = seed.merge_timeline(["import"] * 120, ["upload"] * 29)
    stamps = seed.spread_timestamps(len(order), days, now)
    assert all(a < b for a, b in zip(stamps, stamps[1:]))
    assert all(now - timedelta(days=days) < s < now for s in stamps)
    kinds_by_day = {}
    for kind, stamp in zip(order, stamps):
        kinds_by_day.setdefault(stamp.date(), set()).add(kind)
    assert len(kinds_by_day) >= days
    assert all(kinds == {"import", "upload"} for kinds in kinds_by_day.values())

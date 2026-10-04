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

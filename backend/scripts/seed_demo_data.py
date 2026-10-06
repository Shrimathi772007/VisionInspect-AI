"""Seed a clean demonstration database: demo users, one product per MVTec category, and inspections created
through the application's own import/upload code paths (real AI, confidence, localization, severity,
quality decision and heatmap).

    python scripts/seed_demo_data.py --qe-email qe@demo.local --qe-name "Demo QE" \
        --supervisor-email supervisor@demo.local --supervisor-name "Demo Supervisor" [--days 14] [--per-category 8]
    python scripts/seed_demo_data.py ... --dry-run          # print the plan, write nothing, ask nothing
    python scripts/seed_demo_data.py ... --reset            # print what would be wiped, then stop
    python scripts/seed_demo_data.py ... --reset --yes      # wipe ALL inspections (and their files), then seed
    python scripts/seed_demo_data.py ... --reset --reset-ids --yes [--extra-uploads 25]
                                                            # also delete the DEMO-* products and restart the
                                                            # inspection and product ids at 1, then seed

Run `alembic upgrade head` on the target database first. The database is the one configured for the app
(backend/.env: POSTGRES_*), exactly like the API.

DEMO DATA, NOT ACCURACY RESULTS. Inspections are imported from the MVTec AD "test" split (the one-shot final
evaluation is finished, so test images may be used for demonstrations). Nothing here measures model
quality.

Timestamps are demo timestamps spread over N days (--days): after each inspection is created through the
normal path, its inspection_date and created_at are set to a deterministic, evenly spaced point in the last N
days (oldest first), so the dashboard's trend charts show a series. The uploads are spread evenly between the
imports (merge_timeline), so every day has both kinds, and inspections are created in timestamp order, so ids
and dates increase together.

What it creates (all idempotent - existing rows are kept and reported):
  * a quality engineer (app.auth.bootstrap.create_or_promote_quality_engineer) and a factory supervisor (the
    app's password hashing). Passwords are read with getpass (typed twice) or from the environment variables
    VISIONINSPECT_SEED_QE_PASSWORD / VISIONINSPECT_SEED_SUPERVISOR_PASSWORD. They are never accepted on the
    command line and never printed or logged.
  * 15 products "<Category> Demo Product", codes DEMO-<CATEGORY>-001, category set (existing codes skipped).
  * --per-category imports per category through POST /inspections/import's handler: half "good", the rest
    round-robin across the defect types (all lists sorted; no randomness).
  * 4 upload-style inspections (no ground truth) through the upload pipeline: tile and bottle train/good
    copies, a tile copy with a painted black bar, and a wood train/good copy (wood's model is
    NOT_PRODUCTION_READY, so it goes to MANUAL_REVIEW).
  * --extra-uploads N (default 25) more upload-style inspections (no ground truth, so only these show AI-only
    severity scores and review cases) through the same upload pipeline, for the demo products. Each is a copy
    of a dataset/<category>/train/good image (train/good only), most with a painted black mark: a small spot
    (~1% of the image), a linear stripe, a medium patch (~5-8%), a large block (~15-25%) or three scattered
    spots, drawn as a bar, a blob or a square. The rest are plain copies (good). Wood, carpet, screw and pill
    (NOT_PRODUCTION_READY models, so MANUAL_REVIEW) always get a painted mark, at least 10 other categories
    get one, and tile and bottle get plain copies (plan_extra_uploads). Fully deterministic: one
    random.Random(42) chooses the source files, mark positions and sizes; nothing else is random.
    Painted defects are synthetic demo marks, not real defects; results are demonstration data, not accuracy.
If the demo products already have inspections, no inspections are added (use --reset --yes to start again).

--reset deletes every inspection in the database plus only the files that belong to them (their uploaded
images under storage/uploads and heatmaps under storage/heatmaps); users and products are kept. Without
--yes it prints the counts and stops.

--reset-ids (only with --reset) additionally deletes the DEMO-* products (never users) and restarts the
inspections and products id sequences at 1 (setval(seq, 1, false); TRUNCATE is not used because defects has
a foreign key to inspections and is never touched). The products sequence is only restarted when no other
product is left, so ids can never collide. Without --yes it prints the plan and stops.
"""

import argparse
import asyncio
import getpass
import math
import os
import random
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

QE_PASSWORD_ENV = "VISIONINSPECT_SEED_QE_PASSWORD"
SUPERVISOR_PASSWORD_ENV = "VISIONINSPECT_SEED_SUPERVISOR_PASSWORD"
DEMO_SPLIT = "test"
GOOD = "good"
UPLOAD_FILES = (  # (product category, source category, painted bar?)
    ("tile", "tile", False),
    ("bottle", "bottle", False),
    ("tile", "tile", True),
    ("wood", "wood", False),
)

EXTRA_UPLOADS_DEFAULT = 25
EXTRA_UPLOADS_SEED = 42
# Painted mark size classes: target fraction of the image area (min, max) the mark covers.
SIZE_CLASSES = {
    "small_spot": (0.008, 0.012),
    "linear_stripe": (0.015, 0.03),
    "medium_patch": (0.05, 0.08),
    "large_block": (0.15, 0.25),
    "scattered_spots": (0.015, 0.024),  # three separate spots, together
}
SHAPES_BY_CLASS = {
    "small_spot": ("blob", "square"),
    "linear_stripe": ("bar",),
    "medium_patch": ("blob", "square"),
    "large_block": ("square",),
    "scattered_spots": ("blob", "square"),
}
# The 25-item mix (cycled when N > 25, first N when smaller): (category, size class or None for plain).
# Wood, carpet, screw and pill come first with marks big enough to be flagged; plain tile and bottle early.
EXTRA_UPLOAD_TEMPLATE = (
    ("wood", "medium_patch"), ("carpet", "medium_patch"), ("screw", "large_block"), ("pill", "large_block"),
    ("tile", None), ("bottle", None),
    ("bottle", "small_spot"), ("cable", "linear_stripe"), ("capsule", "medium_patch"), ("grid", "large_block"),
    ("hazelnut", "scattered_spots"), ("leather", "small_spot"), ("metal_nut", "linear_stripe"),
    ("tile", "medium_patch"), ("toothbrush", "large_block"), ("transistor", "scattered_spots"),
    ("zipper", "linear_stripe"),
    ("cable", None), ("hazelnut", None), ("zipper", None),
    ("leather", "linear_stripe"), ("tile", "small_spot"), ("carpet", "large_block"), ("wood", "scattered_spots"),
    ("bottle", "linear_stripe"),
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no database)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed demonstration data (demo data, not accuracy results).")
    parser.add_argument("--qe-email", required=True)
    parser.add_argument("--qe-name", required=True)
    parser.add_argument("--supervisor-email", required=True)
    parser.add_argument("--supervisor-name", required=True)
    parser.add_argument("--days", type=int, default=14, help="spread demo timestamps over this many days (default 14)")
    parser.add_argument("--per-category", type=int, default=8, help="dataset imports per MVTec category (default 8)")
    parser.add_argument("--reset", action="store_true", help="wipe ALL inspections first (needs --yes)")
    parser.add_argument("--reset-ids", action="store_true",
                        help="with --reset: also delete the DEMO-* products and restart inspection/product ids at 1")
    parser.add_argument("--yes", action="store_true", help="confirm --reset")
    parser.add_argument("--extra-uploads", type=int, default=EXTRA_UPLOADS_DEFAULT,
                        help=f"extra upload-style inspections with painted demo marks (default {EXTRA_UPLOADS_DEFAULT})")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.days < 1:
        build_parser().error("--days must be at least 1")
    if args.per_category < 1:
        build_parser().error("--per-category must be at least 1")
    if args.extra_uploads < 0:
        build_parser().error("--extra-uploads must be 0 or more")
    if args.reset_ids and not args.reset:
        build_parser().error("--reset-ids needs --reset")
    return args


def select_images(files_by_type: dict[str, list[str]], per_category: int) -> list[tuple[str, str]]:
    """Deterministic (defect_type, filename) picks for one category: ceil(n/2) "good" images (sorted, from the
    start) and the rest round-robin across the sorted defect types, each taking its next sorted file; when one
    side runs out the other fills up. The result alternates good / defective (good first), so a demo period
    never shows all good images first and all defects later. No randomness."""
    good = sorted(files_by_type.get(GOOD, []))
    defect_types = sorted(t for t, files in files_by_type.items() if t != GOOD and files)
    defect_files = {t: sorted(files_by_type[t]) for t in defect_types}

    defects = []
    cursor = {t: 0 for t in defect_types}
    while len(defects) < per_category:
        added = False
        for defect_type in defect_types:
            if len(defects) >= per_category:
                break
            if cursor[defect_type] < len(defect_files[defect_type]):
                defects.append((defect_type, defect_files[defect_type][cursor[defect_type]]))
                cursor[defect_type] += 1
                added = True
        if not added:
            break

    n_good = min(len(good), max(math.ceil(per_category / 2), per_category - len(defects)))
    goods = [(GOOD, name) for name in good[:n_good]]
    defects = defects[: per_category - len(goods)]
    picks = []
    for i in range(max(len(goods), len(defects))):
        if i < len(goods):
            picks.append(goods[i])
        if i < len(defects):
            picks.append(defects[i])
    return picks


def interleave(plan_by_category: dict[str, list]) -> list:
    """Display order: slot 0 of every category (sorted), then slot 1, ... - so every day of the demo period
    shows several categories. Every second category (by sorted position) has its items swapped in pairs
    (1, 0, 3, 2, ...), so with select_images' good/defective alternation each slot mixes good and defective
    images instead of whole days of one kind."""
    lists = []
    for position, category in enumerate(sorted(plan_by_category)):
        items = list(plan_by_category[category])
        if position % 2 == 1:
            for i in range(0, len(items) - 1, 2):
                items[i], items[i + 1] = items[i + 1], items[i]
        lists.append(items)
    order = []
    longest = max((len(items) for items in lists), default=0)
    for slot in range(longest):
        for items in lists:
            if slot < len(items):
                order.append(items[slot])
    return order


def spread_timestamps(count: int, days: int, now: datetime) -> list[datetime]:
    """`count` strictly increasing timestamps, evenly spaced inside (now - days, now), oldest first."""
    if count <= 0:
        return []
    span = timedelta(days=days)
    step = span / (count + 1)
    start = now - span
    return [start + step * (i + 1) for i in range(count)]


def merge_timeline(imports: list, uploads: list) -> list:
    """Imports and uploads in one display order, the uploads spread evenly between the imports (upload j at
    position floor((j + 0.5) * total / len(uploads))), so every part of the demo period has both kinds.
    Relative order inside each list is kept."""
    total = len(imports) + len(uploads)
    if not uploads:
        return list(imports)
    upload_slots = {math.floor((j + 0.5) * total / len(uploads)) for j in range(len(uploads))}
    pending_imports, pending_uploads = iter(imports), iter(uploads)
    return [next(pending_uploads) if i in upload_slots else next(pending_imports) for i in range(total)]


@dataclass(frozen=True)
class PaintSpec:
    """A synthetic demo mark: shape ("bar", "blob" or "square") and boxes (x0, y0, x1, y1) as fractions of the
    image width/height. Synthetic demo marks, not real defects."""
    size_class: str
    shape: str
    boxes: tuple[tuple[float, float, float, float], ...]


def _place(rng: random.Random, w: float, h: float, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float, float, float]:
    x0 = rng.uniform(lo, hi - w) if hi - w > lo else (1 - w) / 2
    y0 = rng.uniform(lo, hi - h) if hi - h > lo else (1 - h) / 2
    return (x0, y0, x0 + w, y0 + h)


def _shape_dims(shape: str, fraction: float, rng: random.Random) -> tuple[float, float]:
    """(w, h) as image fractions so the shape covers `fraction` of the image area."""
    if shape == "square":
        side = math.sqrt(fraction)
        return side, side
    if shape == "blob":  # ellipse: area = pi/4 * w * h
        ratio = rng.uniform(0.7, 1.4)
        w = math.sqrt(4 * fraction / math.pi * ratio)
        return w, 4 * fraction / (math.pi * w)
    length = rng.uniform(0.55, 0.75)  # bar: long and thin (aspect >= 10)
    thickness = fraction / length
    return (length, thickness) if rng.random() < 0.5 else (thickness, length)


def make_paint_spec(size_class: str, rng: random.Random) -> PaintSpec:
    """A deterministic (given rng's state) mark of one size class; the covered area is drawn from the middle
    of the class's range."""
    lo, hi = SIZE_CLASSES[size_class]
    fraction = rng.uniform(lo + (hi - lo) * 0.15, hi - (hi - lo) * 0.15)
    shape = rng.choice(SHAPES_BY_CLASS[size_class])
    if size_class != "scattered_spots":
        return PaintSpec(size_class, shape, (_place(rng, *_shape_dims(shape, fraction, rng)),))
    boxes = []  # three spots, one in each of three separate thirds of the image (left/middle/right columns)
    for column in range(3):
        w, h = _shape_dims(shape, fraction / 3, rng)
        x_lo, x_hi = 0.05 + column / 3, (column + 1) / 3 - 0.05
        x0 = rng.uniform(x_lo, max(x_lo, x_hi - w))
        y0 = rng.uniform(0.1, 0.9 - h)
        boxes.append((x0, y0, x0 + w, y0 + h))
    return PaintSpec(size_class, shape, tuple(boxes))


def _pixel_box(box: tuple[float, float, float, float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [round(x0 * width), round(y0 * height), round(x1 * width) - 1, round(y1 * height) - 1]


def paint_mask(spec: PaintSpec, size: tuple[int, int]):
    """The mark as a PIL "L" mask (255 = painted) for an image of `size`."""
    from PIL import Image, ImageDraw

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for box in spec.boxes:
        pixels = _pixel_box(box, *size)
        if spec.shape == "blob":
            draw.ellipse(pixels, fill=255)
        else:
            draw.rectangle(pixels, fill=255)
    return mask


def painted_area_fraction(spec: PaintSpec, size: tuple[int, int]) -> float:
    histogram = paint_mask(spec, size).histogram()
    return histogram[255] / (size[0] * size[1])


def paint_defect(image, spec: PaintSpec):
    """A copy of `image` with the mark painted black (same size and mode)."""
    painted = image.copy()
    black = 0 if painted.mode in ("L", "1", "I", "F") else (0,) * len(painted.getbands())
    painted.paste(black, (0, 0), paint_mask(spec, painted.size))
    return painted


def plan_extra_uploads(count: int, files_by_category: dict[str, list[str]], seed: int = EXTRA_UPLOADS_SEED) -> list:
    """`count` upload-style items from EXTRA_UPLOAD_TEMPLATE (cycled), each with a train/good source file and,
    unless plain, a PaintSpec. One random.Random(seed) picks the files (no file used twice per category while
    any is left), the marks and the final order; the same inputs always give the same plan."""
    rng = random.Random(seed)
    used = {category: set() for category in files_by_category}
    items = []
    for i in range(count):
        category, size_class = EXTRA_UPLOAD_TEMPLATE[i % len(EXTRA_UPLOAD_TEMPLATE)]
        files = sorted(files_by_category[category])
        unused = [f for f in files if f not in used[category]] or files
        filename = rng.choice(unused)
        used[category].add(filename)
        spec = make_paint_spec(size_class, rng) if size_class else None
        items.append(PlannedInspection("upload", category, filename=filename, painted=spec is not None,
                                       source_category=category, paint=spec))
    rng.shuffle(items)
    return items


def password_from(env_name: str, prompt_label: str, prompt=getpass.getpass) -> str:
    """The password from `env_name`, or typed twice through getpass. Never echoed, printed or logged."""
    value = os.environ.get(env_name)
    if value:
        return value
    first = prompt(f"Password for {prompt_label}: ")
    second = prompt(f"Repeat the password for {prompt_label}: ")
    if first != second:
        raise SystemExit(f"The two passwords for {prompt_label} did not match.")
    return first


def require_yes_for_reset(reset: bool, yes: bool) -> bool:
    """True when the reset may proceed; False (after which the caller stops) when --yes is missing."""
    return bool(reset and yes)


# ---------------------------------------------------------------------------
# Database work
# ---------------------------------------------------------------------------

@dataclass
class PlannedInspection:
    kind: str  # "import" or "upload"
    category: str  # product category
    defect_type: str | None = None
    filename: str | None = None
    painted: bool = False
    source_category: str | None = None
    paint: PaintSpec | None = None  # extra uploads: the synthetic demo mark (None = plain copy)


def demo_product_code(category: str) -> str:
    return f"DEMO-{category.upper()}-001"


def demo_product_name(category: str) -> str:
    return f"{category.replace('_', ' ').title()} Demo Product"


def build_plan(per_category: int, extra_uploads: int = 0) -> list[PlannedInspection]:
    from app.dataset.categories import MVTEC_CATEGORIES
    from app.dataset.service import list_defect_types, list_images

    by_category = {}
    for category in sorted(MVTEC_CATEGORIES):
        files = {entry["defect_type"]: list_images(category, DEMO_SPLIT, entry["defect_type"])
                 for entry in list_defect_types(category, DEMO_SPLIT)}
        by_category[category] = [PlannedInspection("import", category, t, f) for t, f in select_images(files, per_category)]
    uploads = [PlannedInspection("upload", product, filename="000.png", painted=painted, source_category=source)
               for product, source, painted in UPLOAD_FILES]
    train_good = {category: list_images(category, "train", GOOD) for category in sorted(MVTEC_CATEGORIES)}
    uploads += plan_extra_uploads(extra_uploads, train_good)
    return merge_timeline(interleave(by_category), uploads)


def upload_label(item: PlannedInspection) -> str:
    stem = Path(item.filename).stem
    if item.paint is not None:
        return f"{item.source_category}_train_good_{stem}_painted_{item.paint.size_class}_{item.paint.shape}.png"
    return f"{item.source_category}_train_good_{stem}{'_painted_bar' if item.painted else ''}.png"


def print_reset_plan(db, reset_ids: bool = False) -> tuple[int, int, int]:
    from sqlalchemy import func, select

    from app.models.inspection import Inspection, InspectionSource

    total = db.scalar(select(func.count()).select_from(Inspection))
    uploads = db.scalar(select(func.count()).select_from(Inspection).where(Inspection.source == InspectionSource.upload))
    heatmaps = db.scalar(select(func.count()).select_from(Inspection).where(Inspection.heatmap_path.is_not(None)))
    print(f"--reset would delete ALL {total} inspections: {uploads} uploads (their image files under storage/uploads), "
          f"{total - uploads} dataset imports (dataset files are never touched) and {heatmaps} heatmap files. "
          f"Users are kept{'' if reset_ids else ' and products are kept'}.")
    if reset_ids:
        from app.models.product import Product

        demo = db.scalar(select(func.count()).select_from(Product).where(Product.product_code.like(DEMO_CODE_PATTERN)))
        other = db.scalar(select(func.count()).select_from(Product).where(Product.product_code.not_like(DEMO_CODE_PATTERN)))
        products_note = "(no other products exist)" if not other else f"- NOT done: {other} non-demo products exist"
        print(f"--reset-ids would also delete the {demo} DEMO-* products (users are never touched), restart the "
              f"inspections id sequence at 1 and the products id sequence at 1 {products_note}.")
    return total, uploads, heatmaps


DEMO_CODE_PATTERN = "DEMO-%"
RESET_SEQUENCE_TABLES = ("inspections", "products")


def reset_ids(db) -> dict:
    """After reset_inspections: delete the DEMO-* products and restart the inspections and products id sequences
    at 1. Only these two tables are touched (never users or defects); a sequence is restarted only when its
    table is empty, so ids can never collide."""
    from sqlalchemy import text

    stats = {"demo_products": db.execute(text("DELETE FROM products WHERE product_code LIKE :pattern"),
                                         {"pattern": DEMO_CODE_PATTERN}).rowcount}
    for table in RESET_SEQUENCE_TABLES:
        remaining = db.execute(text(f"SELECT count(*) FROM {table}")).scalar()
        if remaining:
            stats[f"{table}_sequence"] = f"kept ({remaining} rows left)"
            continue
        sequence = db.execute(text("SELECT pg_get_serial_sequence(:table, 'id')"), {"table": table}).scalar()
        db.execute(text("SELECT setval(CAST(:sequence AS regclass), 1, false)"), {"sequence": sequence})
        stats[f"{table}_sequence"] = "restarted at 1"
    db.commit()
    return stats


def reset_inspections(db) -> dict:
    """Delete every inspection and only the files that belong to them (upload images, heatmaps)."""
    from sqlalchemy import select

    from app.inspections.storage import STORAGE_ROOT, delete_heatmap_file, delete_upload_file
    from app.models.inspection import Inspection, InspectionSource

    stats = Counter()
    product_ids = set()
    for inspection in db.scalars(select(Inspection)).all():
        if inspection.source == InspectionSource.upload:
            try:
                delete_upload_file(inspection.image_path)
                stats["upload_files"] += 1
                product_ids.add(inspection.product_id)
            except Exception:  # noqa: BLE001 - an unresolvable path is skipped, never followed
                stats["upload_files_skipped"] += 1
        if inspection.heatmap_path:
            try:
                delete_heatmap_file(inspection.heatmap_path)
                stats["heatmap_files"] += 1
            except Exception:  # noqa: BLE001
                stats["heatmap_files_skipped"] += 1
        db.delete(inspection)
        stats["inspections"] += 1
    db.commit()
    for product_id in product_ids:
        folder = STORAGE_ROOT / str(product_id)
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            stats["empty_upload_folders"] += 1
    return dict(stats)


def ensure_users(db, args, dry_run: bool) -> dict:
    from sqlalchemy import select

    from app.auth.bootstrap import create_or_promote_quality_engineer, normalize_email, validate_password
    from app.auth.security import hash_password
    from app.models.user import User, UserRole

    result = {}
    qe = db.scalar(select(User).where(User.email == normalize_email(args.qe_email)))
    if qe is not None:
        result["qe"] = f"kept existing user (role {qe.role.value})"
    elif dry_run:
        result["qe"] = "would create quality engineer"
    else:
        password = password_from(QE_PASSWORD_ENV, args.qe_email)
        validate_password(password)
        qe = create_or_promote_quality_engineer(db, args.qe_name, args.qe_email, password, promote_existing=False)
        db.commit()
        result["qe"] = "created quality engineer"

    supervisor_email = normalize_email(args.supervisor_email)
    supervisor = db.scalar(select(User).where(User.email == supervisor_email))
    if supervisor is not None:
        result["supervisor"] = f"kept existing user (role {supervisor.role.value})"
    elif dry_run:
        result["supervisor"] = "would create factory supervisor"
    else:
        password = password_from(SUPERVISOR_PASSWORD_ENV, args.supervisor_email)
        validate_password(password)
        db.add(User(name=args.supervisor_name.strip(), email=supervisor_email, password_hash=hash_password(password),
                    role=UserRole.factory_supervisor))
        db.commit()
        result["supervisor"] = "created factory supervisor"
    result["qe_user"] = qe
    return result


def ensure_products(db, dry_run: bool) -> tuple[dict, list[str], list[str]]:
    from sqlalchemy import select

    from app.dataset.categories import MVTEC_CATEGORIES
    from app.models.product import Product

    products, created, kept = {}, [], []
    for category in sorted(MVTEC_CATEGORIES):
        code = demo_product_code(category)
        product = db.scalar(select(Product).where(Product.product_code == code))
        if product is None:
            created.append(code)
            if not dry_run:
                product = Product(product_name=demo_product_name(category), product_code=code, category=category)
                db.add(product)
                db.commit()
                db.refresh(product)
        else:
            kept.append(code)
        products[category] = product
    return products, created, kept


def painted_copy(source: Path, target: Path) -> Path:
    from PIL import Image, ImageDraw

    with Image.open(source) as im:
        image = im.convert("RGB")
    width, height = image.size
    ImageDraw.Draw(image).line([(0, 0), (width, height)], fill=(0, 0, 0), width=max(4, round(0.07 * min(width, height))))
    image.save(target)
    return target


def write_upload_copy(item: PlannedInspection, source: Path, target: Path) -> Path:
    """The upload file for one planned upload: a byte copy of the train/good image, the legacy diagonal bar
    (base tile upload) or the item's synthetic demo mark (extra uploads)."""
    from PIL import Image

    if item.paint is not None:
        with Image.open(source) as im:
            im.load()
            paint_defect(im, item.paint).save(target)
    elif item.painted:
        painted_copy(source, target)
    else:
        target.write_bytes(source.read_bytes())
    return target


def create_inspections(db, plan: list[PlannedInspection], products: dict, qe_user, timestamps: list[datetime]) -> list:
    """Create every planned inspection through the API's own handlers, in plan (= timestamp) order so ids and
    dates increase together, then set its demo timestamp."""
    from starlette.datastructures import UploadFile

    from app.inspections.router import _process_upload, import_dataset_inspection
    from app.inspections.schemas import DatasetImportRequest
    from app.inspections.storage import DATASET_ROOT

    created = [None] * len(plan)
    order = range(len(plan))
    with tempfile.TemporaryDirectory(prefix="visioninspect_seed_") as tmp:
        for count, index in enumerate(order, start=1):
            item = plan[index]
            product = products[item.category]
            if item.kind == "import":
                payload = DatasetImportRequest(product_id=product.id, category=item.category, split=DEMO_SPLIT,
                                               defect_type=item.defect_type, filename=item.filename)
                inspection = import_dataset_inspection(payload, db=db, current_user=qe_user)
            else:
                source = DATASET_ROOT / item.source_category / "train" / GOOD / item.filename
                name = upload_label(item)
                local = write_upload_copy(item, source, Path(tmp) / name)
                with open(local, "rb") as handle:
                    upload = UploadFile(file=handle, filename=name, size=local.stat().st_size)
                    inspection = asyncio.run(_process_upload(db, product.id, upload, time.perf_counter()))
            inspection.inspection_date = timestamps[index]
            inspection.created_at = timestamps[index]
            db.commit()
            db.refresh(inspection)
            created[index] = inspection
            label = f"{item.defect_type}/{item.filename}" if item.kind == "import" else name
            print(f"  [{count:>3}/{len(plan)}] id {inspection.id:>5} {item.kind:<6} {item.category:<11} {label:<58} -> "
                  f"{inspection.ai_prediction or 'no AI':<9} {inspection.quality_decision}")
    return created


def print_summary(db) -> None:
    from sqlalchemy import func, select

    from app.inspections.analytics import get_inspection_analytics_summary
    from app.models.inspection import Inspection
    from app.models.product import Product

    def table(title, rows):
        print(f"\n{title}")
        for key, value in rows:
            print(f"  {str(key):<24} {value}")

    by_category = db.execute(select(Product.category, func.count(Inspection.id)).join(Inspection, Inspection.product_id == Product.id)
                             .group_by(Product.category).order_by(Product.category)).all()
    table("Inspections by category", by_category)
    for column, title in ((Inspection.source, "by source"), (Inspection.quality_decision, "by quality decision"),
                          (Inspection.ai_prediction, "by AI prediction"), (Inspection.review_required, "by review_required"),
                          (Inspection.severity_level, "by severity level"),
                          (Inspection.localization["defect_form"].as_string(), "by defect form")):
        rows = db.execute(select(column, func.count()).group_by(column).order_by(func.count().desc())).all()
        table(f"Inspections {title}", [(getattr(k, "value", k), v) for k, v in rows])
    min_id, max_id = db.execute(select(func.min(Inspection.id), func.max(Inspection.id))).one()
    print(f"\nInspection ids: {min_id} .. {max_id}")
    summary = get_inspection_analytics_summary(db)
    rate = summary.automation_rate
    print(f"\nAutomation rate: {rate:.1%}" if rate is not None else "\nAutomation rate: n/a")
    print(f"Automation counts: {summary.automation_counts.model_dump()}  manual_review_count: {summary.manual_review_count}")
    days_with_data = sum(1 for day in summary.trend_monitoring.daily if day.total > 0)
    print(f"Trend window: {len(summary.trend_monitoring.daily)} days, {days_with_data} with inspections")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    from sqlalchemy import func, select

    from app.auth.bootstrap import BootstrapError, normalize_email
    from app.database import SessionLocal
    from app.models.inspection import Inspection

    try:  # the same validation the API applies, before anything is read or written
        normalize_email(args.qe_email)
        normalize_email(args.supervisor_email)
    except BootstrapError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("DEMO DATA, NOT ACCURACY RESULTS. Timestamps are demo timestamps spread over "
          f"{args.days} days.{' (dry run: nothing is written)' if args.dry_run else ''}")
    with SessionLocal() as db:
        print(f"Database: {db.get_bind().url.database}")
        if args.reset:
            print_reset_plan(db, args.reset_ids)
            if not require_yes_for_reset(args.reset, args.yes):
                print("Refusing to reset without --yes. Nothing was changed.")
                return 1
            if args.dry_run:
                print("(dry run: not resetting)")
            else:
                print("Reset done:", reset_inspections(db))
                if args.reset_ids:
                    print("Id reset done:", reset_ids(db))

        users = ensure_users(db, args, args.dry_run)
        print(f"QE {args.qe_email}: {users['qe']}; supervisor {args.supervisor_email}: {users['supervisor']}")
        products, created_codes, kept_codes = ensure_products(db, args.dry_run)
        print(f"Products: {len(created_codes)} {'would be ' if args.dry_run else ''}created, {len(kept_codes)} kept")

        plan = build_plan(args.per_category, args.extra_uploads)
        timestamps = spread_timestamps(len(plan), args.days, datetime.now(timezone.utc).replace(microsecond=0))
        existing = 0
        product_ids = [p.id for p in products.values() if p is not None]
        if product_ids:
            existing = db.scalar(select(func.count()).select_from(Inspection).where(Inspection.product_id.in_(product_ids)))
        print(f"Plan: {sum(p.kind == 'import' for p in plan)} dataset imports ({DEMO_SPLIT} split) + "
              f"{sum(p.kind == 'upload' for p in plan)} uploads, {timestamps[0]:%Y-%m-%d %H:%M} .. {timestamps[-1]:%Y-%m-%d %H:%M} UTC")
        if args.dry_run:
            for item, stamp in zip(plan, timestamps):
                label = f"{item.defect_type}/{item.filename}" if item.kind == "import" else upload_label(item)
                print(f"  {stamp:%Y-%m-%d %H:%M}  {item.kind:<6} {item.category:<11} {label}")
            return 0
        if existing:
            print(f"The demo products already have {existing} inspections; not adding more (use --reset --yes to start again).")
        else:
            create_inspections(db, plan, products, users["qe_user"], timestamps)
        print_summary(db)
    return 0


if __name__ == "__main__":
    sys.exit(main())

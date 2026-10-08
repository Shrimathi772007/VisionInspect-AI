"""Localization evaluation of the served models: box AP@0.5 and pixel AUROC (rules: PROTOCOL.md in this folder).

    python scripts/experiments/localization_map/run_box_eval.py --out-dir DIR [--categories a,b] [--allow-test]
                                                                  [--dataset-root ROOT]

Run from backend/. Evaluation only: runs predict_image exactly as served and the localization steps of
app.inspections.service without writing anything but DIR. Never imports or connects to the database. Refuses to read
the real dataset's test/ split without --allow-test; any other dataset root must carry the SYNTHETIC_DATASET marker
written by build_synthetic_dataset.py. Resumable per category (see PROTOCOL.md, "Outputs").
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parents[2]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PROTOCOL_PATH = HERE / "PROTOCOL.md"
SYNTHETIC_MARKER = "SYNTHETIC_DATASET"
IOU_MATCH = 0.5
PRED_MIN_AREA_FRACTION = 0.0025  # app.ai.inference.localization.MIN_COMPONENT_AREA_FRACTION (asserted at run time)
WRN50_CATEGORIES = ("bottle", "capsule", "carpet", "grid", "hazelnut", "pill", "screw", "wood", "zipper")
RESNET18_CATEGORIES = ("cable", "leather", "metal_nut", "tile", "toothbrush", "transistor")
ALL_CATEGORIES = tuple(sorted(WRN50_CATEGORIES + RESNET18_CATEGORIES))
VIEWS = ("components", "merged")

# Pixel AUROC histogram (PROTOCOL.md): log10(amap / threshold) over [-4, 3] in 14,000 bins + underflow + overflow.
HIST_LOG_MIN, HIST_LOG_MAX, HIST_BINS = -4.0, 3.0, 14_000
HIST_SIZE = HIST_BINS + 2  # index 0 = underflow (r <= 1e-4, incl. r <= 0), index HIST_BINS + 1 = overflow (r > 1e3)

APP_FILES = ("app/ai/inference/predict.py", "app/ai/inference/localization.py", "app/ai/inference/serving.py")


class TestSplitRefused(RuntimeError):
    """The dataset root is the real dataset and --allow-test was not given (or the root is not marked synthetic)."""


# ---------------------------------------------------------------------------------------------------------------
# Pure metrics (unit-tested on hand-checked cases)
# ---------------------------------------------------------------------------------------------------------------

def iou(a, b) -> float:
    """IoU of two [x0, y0, x1, y1] boxes in continuous coordinates; 0 when either has no area."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_boxes_from_mask(mask: np.ndarray, merged: bool = False) -> list[dict]:
    """GT boxes of a defect mask (pixel > 0 = defect): one per 8-connected component (no minimum area), or one merged
    bounding box. Each: {"box": [x0, y0, x1, y1] in pixels, "area_fraction": defect pixels / image pixels}."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    total = binary.shape[0] * binary.shape[1]
    if not binary.any():
        return []
    if merged:
        ys, xs = np.nonzero(binary)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        return [{"box": [x0, y0, x1, y1], "area_fraction": float(binary.sum()) / total}]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for label in range(1, count):
        x, y, w, h, pixels = (int(v) for v in stats[label])
        boxes.append({"box": [x, y, x + w, y + h], "area_fraction": pixels / total})
    return boxes


def match_category(images: list[dict]) -> dict:
    """PROTOCOL.md matching. `images`: [{"pred": [(box, confidence), ...], "gt": [box, ...]}] in image order.
    Returns {"tp": [0/1 per prediction in confidence order], "n_gt", "n_pred", "n_tp", "ap"}."""
    flat = [(conf, i, k, box) for i, image in enumerate(images) for k, (box, conf) in enumerate(image["pred"])]
    flat.sort(key=lambda item: (-item[0], item[1], item[2]))  # stable on (image order, box order)
    matched = [set() for _ in images]
    tp = []
    for _conf, i, _k, box in flat:
        best_iou, best_j = 0.0, None
        for j, gt in enumerate(images[i]["gt"]):
            if j in matched[i]:
                continue
            value = iou(box, gt)
            if value > best_iou:
                best_iou, best_j = value, j
        if best_j is not None and best_iou >= IOU_MATCH:
            matched[i].add(best_j)
            tp.append(1)
        else:
            tp.append(0)
    n_gt = sum(len(image["gt"]) for image in images)
    return {"tp": tp, "n_gt": n_gt, "n_pred": len(flat), "n_tp": int(sum(tp)), "ap": average_precision(tp, n_gt)}


def average_precision(tp: list[int], n_gt: int) -> float | None:
    """All-point interpolated AP of predictions already sorted by confidence. None without GT; 0 without predictions."""
    if n_gt == 0:
        return None
    if not tp:
        return 0.0
    tps = np.cumsum(np.asarray(tp, dtype=float))
    fps = np.cumsum(1.0 - np.asarray(tp, dtype=float))
    recall = np.concatenate([[0.0], tps / n_gt, [1.0]])
    precision = np.concatenate([[0.0], tps / (tps + fps), [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def histogram_bins(ratio: np.ndarray) -> np.ndarray:
    """Histogram bin index (0 .. HIST_SIZE - 1) of each score ratio amap / threshold."""
    ratio = np.asarray(ratio, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log10(ratio)
    safe = np.where(np.isfinite(logs), logs, HIST_LOG_MIN - 1.0)  # non-finite -> underflow below, no cast warning
    idx = np.floor((safe - HIST_LOG_MIN) / (HIST_LOG_MAX - HIST_LOG_MIN) * HIST_BINS).astype(np.int64) + 1
    idx = np.minimum(idx, HIST_BINS)  # r == 1e3 belongs to the last regular bin
    idx = np.where(~np.isfinite(logs) | (ratio <= 10.0 ** HIST_LOG_MIN), 0, idx)
    idx = np.where(np.isfinite(logs) & (ratio > 10.0 ** HIST_LOG_MAX), HIST_BINS + 1, idx)
    return np.clip(idx, 0, HIST_SIZE - 1)


def histogram_add(pos: np.ndarray, neg: np.ndarray, ratio: np.ndarray, labels: np.ndarray) -> None:
    bins = histogram_bins(ratio.ravel())
    labels = np.asarray(labels, dtype=bool).ravel()
    pos += np.bincount(bins[labels], minlength=HIST_SIZE)
    neg += np.bincount(bins[~labels], minlength=HIST_SIZE)


def histogram_auroc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """(sum pos_b x neg_below_b + 0.5 pos_b x neg_b) / (P x N); None when either class is empty."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    p, n = pos.sum(), neg.sum()
    if p == 0 or n == 0:
        return None
    neg_below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float((np.sum(pos * neg_below) + 0.5 * np.sum(pos * neg)) / (p * n))


def mask_to_map(mask: np.ndarray, region, side: int) -> tuple[np.ndarray, int]:
    """(S x S boolean mask in map space, defect pixels outside the analysed region). PROTOCOL.md, "Pixel AUROC"."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height, width = binary.shape
    x0, y0, x1, y1 = region
    left, top, right, bottom = round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)
    inside = binary[top:bottom, left:right]
    outside = int(binary.sum() - inside.sum())
    resized = cv2.resize(inside, (side, side), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool), outside


def category_metrics(records: list[dict], pos: np.ndarray, neg: np.ndarray, gt_outside_pixels: int) -> dict:
    """Metrics of one category from its per-image records (PROTOCOL.md, "Matching and AP@0.5")."""
    out = {"n_images": len(records), "n_defective": sum(r["defect_type"] != "good" for r in records),
           "n_good": sum(r["defect_type"] == "good" for r in records),
           "missed_images": sum(r["defect_type"] != "good" and not r["boxes"] for r in records),
           "good_images_with_boxes": sum(r["defect_type"] == "good" and bool(r["boxes"]) for r in records)}
    for view in VIEWS:
        images = [{"pred": [(b["box_px"], b["confidence"]) for b in r["boxes"]], "gt": [g["box"] for g in r["gt"][view]]}
                  for r in records]
        matched = match_category(images)
        gts = [g for r in records for g in r["gt"][view]]
        out[view] = {
            "ap50": matched["ap"], "gt_boxes": matched["n_gt"], "pred_boxes": matched["n_pred"], "tp": matched["n_tp"],
            "gt_below_min_area": sum(g["area_fraction"] < PRED_MIN_AREA_FRACTION for g in gts),
            "gt_outside_region": sum(g.get("outside_region", False) for g in gts),
        }
    out["pixel_auroc"] = histogram_auroc(pos, neg)
    out["pixel_counts"] = {"positive": int(pos.sum()), "negative": int(neg.sum()),
                           "gt_pixels_outside_region": int(gt_outside_pixels)}
    return out


def mean_of(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(per_category: dict) -> dict:
    def means(key):
        get = (lambda m: m[key]["ap50"]) if key in VIEWS else (lambda m: m[key])
        return {name: mean_of(get(per_category[c]) for c in group if c in per_category)
                for name, group in (("all_15", ALL_CATEGORIES), ("wrn50_9", WRN50_CATEGORIES),
                                    ("resnet18_6", RESNET18_CATEGORIES))}
    return {"map50_components": means("components"), "map50_merged": means("merged"),
            "pixel_auroc": means("pixel_auroc"), "categories_evaluated": sorted(per_category)}


# ---------------------------------------------------------------------------------------------------------------
# Guard, hashes, I/O
# ---------------------------------------------------------------------------------------------------------------

def real_dataset_roots() -> set[Path]:
    roots = {(REPO_ROOT / "dataset").resolve()}
    if os.environ.get("DATASET_ROOT"):
        roots.add(Path(os.environ["DATASET_ROOT"]).resolve())
    return roots


def check_dataset_root(root: Path, allow_test: bool) -> str:
    """'real' or 'synthetic'; raises TestSplitRefused unless the root may be read."""
    root = Path(root).resolve()
    if root in real_dataset_roots():
        if not allow_test:
            raise TestSplitRefused(
                f"Refusing to read the real test/ split under {root} without --allow-test "
                "(second scoring of the final test sets, localization only; see PROTOCOL.md)."
            )
        return "real"
    if (root / SYNTHETIC_MARKER).is_file():
        return "synthetic"
    raise TestSplitRefused(f"{root} is neither the project dataset nor marked {SYNTHETIC_MARKER}; refusing.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_path_for(config) -> Path:
    match = re.search(r"backend/ai_models/\S+?\.json", config.provenance)
    if not match:
        raise RuntimeError(f"No lock file in the provenance of {config.category}: {config.provenance!r}")
    return REPO_ROOT / match.group(0)


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def list_images(dataset_root: Path, category: str) -> list[tuple[str, Path, Path | None]]:
    test_dir = dataset_root / category / "test"
    items = []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for image in sorted(defect_dir.glob("*.png")):
            mask = None
            if defect_dir.name != "good":
                mask = dataset_root / category / "ground_truth" / defect_dir.name / f"{image.stem}_mask.png"
                if not mask.is_file():
                    raise FileNotFoundError(f"Defective image without a mask: {image} (expected {mask})")
            items.append((defect_dir.name, image, mask))
    if not items:
        raise FileNotFoundError(f"No test images under {test_dir}")
    return items


# ---------------------------------------------------------------------------------------------------------------
# Served prediction + localization (PROTOCOL.md, "Predictions (exactly as served)")
# ---------------------------------------------------------------------------------------------------------------

def served_localization(image_path: Path, category: str, predict=None):
    """(result, amap, region, localization dict, (W, H)) - the steps of app.inspections.service._localize_and_render
    without the heatmap file."""
    from app.ai.inference import DEFECTIVE_PREDICTION, localization, predict_image
    from app.ai.preprocessing.patchcore_preprocess import decode_image

    result = (predict or predict_image)(image_path, category, return_patch_scores=True)
    if result.patch_scores is None or result.input_mode is None:
        raise RuntimeError(f"{category}: the served model returned no patch grid; it cannot be localized.")
    height, width = decode_image(image_path).shape[:2]
    amap = localization.anomaly_map(result.patch_scores, result.input_size[0])
    region = localization.map_region(result.input_mode, width, height)
    if result.prediction == DEFECTIVE_PREDICTION:
        located = localization.localize(amap, result.threshold, region)
    else:
        located = localization.empty_localization()
    return result, amap, region, located, (width, height)


def evaluate_category(category: str, dataset_root: Path, predict=None, log=print) -> dict:
    """Per-image records, histogram and metrics of one category."""
    from app.ai.inference import localization

    assert localization.MIN_COMPONENT_AREA_FRACTION == PRED_MIN_AREA_FRACTION
    items = list_images(dataset_root, category)
    pos, neg = np.zeros(HIST_SIZE, dtype=np.int64), np.zeros(HIST_SIZE, dtype=np.int64)
    records, outside_pixels, started = [], 0, time.perf_counter()
    for n, (defect_type, image_path, mask_path) in enumerate(items, 1):
        t0 = time.perf_counter()
        result, amap, region, located, (width, height) = served_localization(image_path, category, predict)
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                raise ValueError(f"Mask {mask_path} is unreadable or not {width}x{height}")
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0, x1, y1 = region
        gt = {}
        for view in VIEWS:
            boxes = gt_boxes_from_mask(mask, merged=(view == "merged"))
            for g in boxes:
                b = g["box"]
                g["outside_region"] = (b[0] < round(x0 * width) or b[1] < round(y0 * height)
                                       or b[2] > round(x1 * width) or b[3] > round(y1 * height))
            gt[view] = boxes
        map_mask, outside = mask_to_map(mask, region, amap.shape[0])
        outside_pixels += outside
        histogram_add(pos, neg, amap / float(result.threshold), map_mask)
        boxes = [{"box_px": [b["x"] * width, b["y"] * height, (b["x"] + b["w"]) * width, (b["y"] + b["h"]) * height],
                  "confidence": b["peak"] / float(result.threshold), "peak": b["peak"],
                  "area_fraction": b["area_fraction"]} for b in located["boxes"]]
        records.append({"defect_type": defect_type, "filename": image_path.name, "prediction": result.prediction,
                        "score": float(result.reconstruction_error), "threshold": float(result.threshold),
                        "model_name": result.model_name, "image_size": [width, height],
                        "analysed_region": [round(v, 6) for v in region], "mask_rule": located.get("mask_rule"),
                        "boxes": boxes, "gt": gt})
        log(f"[{category}] {n}/{len(items)} {defect_type}/{image_path.name} pred={result.prediction} "
            f"boxes={len(boxes)} gt={len(gt['components'])} {time.perf_counter() - t0:.2f}s")
    metrics = category_metrics(records, pos, neg, outside_pixels)
    nonzero = lambda h: {str(i): int(v) for i, v in enumerate(h) if v}  # noqa: E731
    return {"category": category, "records": records, "histogram": {"positive": nonzero(pos), "negative": nonzero(neg)},
            "metrics": metrics, "seconds": round(time.perf_counter() - started, 1)}


def current_hashes(categories) -> dict:
    from app.ai.inference.serving import get_serving_config

    locks = {}
    for category in categories:
        path = lock_path_for(get_serving_config(category))
        locks[category] = {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)}
    return {"script_sha256": sha256_file(Path(__file__)), "protocol_sha256": sha256_file(PROTOCOL_PATH), "locks": locks}


def run(out_dir: Path, categories, dataset_root: Path, allow_test: bool, predict=None, log=print) -> dict:
    kind = check_dataset_root(dataset_root, allow_test)
    out_dir = Path(out_dir)
    hashes = current_hashes(categories)
    per_category, timings = {}, {}
    for category in categories:
        path = out_dir / "categories" / f"{category}.json"
        expected = {"script_sha256": hashes["script_sha256"], "protocol_sha256": hashes["protocol_sha256"],
                    "lock_sha256": hashes["locks"][category]["sha256"], "dataset_kind": kind}
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("provenance") != expected:
                raise RuntimeError(f"{path} was produced with different hashes or data; refusing to mix results.")
            log(f"[{category}] reusing finished result {path}")
        else:
            log(f"[{category}] start ({kind} dataset)")
            saved = evaluate_category(category, Path(dataset_root), predict, log)
            saved["provenance"] = expected
            write_json_atomic(path, saved)
            log(f"[{category}] done in {saved['seconds']}s")
        per_category[category], timings[category] = saved["metrics"], saved["seconds"]

    import torch

    report = {"protocol": "PROTOCOL.md", "dataset_kind": kind, "per_category": per_category,
              "summary": summarize(per_category), "seconds_per_category": timings,
              "note": ("second scoring of final test sets for localization only; no model or threshold changed; "
                       "anomaly-map boxes, not a trained detector")}
    manifest = {**hashes, "app_files": {p: sha256_file(BACKEND_DIR / p) for p in APP_FILES},
                "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                "torch": torch.__version__, "dataset_kind": kind, "categories": list(categories)}
    write_json_atomic(out_dir / "report.json", report)
    write_json_atomic(out_dir / "manifest.json", manifest)
    return report


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Box AP@0.5 and pixel AUROC of the served models (PROTOCOL.md).")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--categories", default=",".join(ALL_CATEGORIES))
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--allow-test", action="store_true",
                        help="read the real dataset's test/ split (second scoring, localization only)")
    args = parser.parse_args(argv)
    args.categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = sorted(set(args.categories) - set(ALL_CATEGORIES))
    if unknown:
        parser.error(f"unknown categories: {unknown}")
    return args


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = run(args.out_dir, args.categories, args.dataset_root, args.allow_test,
                     log=lambda msg: print(msg, flush=True))
    except TestSplitRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Write app/ai/box_eval_addendum.json (tracked) from a finished run_box_eval.py output folder (not tracked).

    python scripts/experiments/localization_map/write_addendum.py --raw-dir DIR --raw-dir-label LABEL

Copies the per-category metrics from DIR/report.json, adds the fraction of GT defect pixels outside the analysed area
(from the per-image records in DIR/categories/), and records the SHA-256 of DIR/report.json and DIR/manifest.json.
Computes nothing else. Refuses a synthetic run or an incomplete one (fewer than 15 categories).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parents[2]
ADDENDUM_PATH = BACKEND_DIR / "app" / "ai" / "box_eval_addendum.json"
NOTE = ("second scoring of final test sets for localization only; no model or threshold changed; anomaly-map boxes, "
        "not a trained detector")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gt_pixel_totals(category_file: Path) -> int:
    """Total GT defect pixels of a category at the original image resolution (sum over its component boxes)."""
    data = json.loads(category_file.read_text(encoding="utf-8"))
    return sum(round(g["area_fraction"] * r["image_size"][0] * r["image_size"][1])
               for r in data["records"] for g in r["gt"]["components"])


def build(raw_dir: Path, raw_dir_label: str) -> dict:
    raw_dir = Path(raw_dir)
    report = json.loads((raw_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    if report["dataset_kind"] != "real" or manifest["dataset_kind"] != "real":
        raise SystemExit("Refusing: not a run on the real test split.")
    if len(report["per_category"]) != 15:
        raise SystemExit(f"Refusing: {len(report['per_category'])} categories, expected 15.")
    categories = {}
    for category, m in sorted(report["per_category"].items()):
        outside = m["pixel_counts"]["gt_pixels_outside_region"]
        total = gt_pixel_totals(raw_dir / "categories" / f"{category}.json")
        categories[category] = {
            "box_ap50": m["components"]["ap50"], "box_ap50_merged": m["merged"]["ap50"], "pixel_auroc": m["pixel_auroc"],
            "n_images": m["n_images"], "n_defective": m["n_defective"], "n_good": m["n_good"],
            "gt_boxes": m["components"]["gt_boxes"], "pred_boxes": m["components"]["pred_boxes"],
            "tp": m["components"]["tp"], "gt_boxes_merged": m["merged"]["gt_boxes"], "tp_merged": m["merged"]["tp"],
            "missed_images": m["missed_images"], "good_images_with_boxes": m["good_images_with_boxes"],
            "gt_boxes_below_min_area": m["components"]["gt_below_min_area"],
            "gt_boxes_outside_analysed_area": m["components"]["gt_outside_region"],
            "gt_pixels_outside_analysed_area": outside,
            "gt_pixels_outside_analysed_area_fraction": outside / total if total else 0.0,
            "lock_sha256": manifest["locks"][category]["sha256"],
        }
    return {
        "note": NOTE,
        "protocol": "backend/scripts/experiments/localization_map/PROTOCOL.md",
        "protocol_sha256": manifest["protocol_sha256"],
        "script": "backend/scripts/experiments/localization_map/run_box_eval.py",
        "script_sha256": manifest["script_sha256"],
        "date": manifest["date"],
        "raw_output": {"path": raw_dir_label, "report_json_sha256": sha256_file(raw_dir / "report.json"),
                       "manifest_json_sha256": sha256_file(raw_dir / "manifest.json")},
        "means": report["summary"],
        "seconds_per_category": report["seconds_per_category"],
        "categories": categories,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--raw-dir-label", required=True, help="path name recorded in the addendum")
    parser.add_argument("--out", type=Path, default=ADDENDUM_PATH)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(build(args.raw_dir, args.raw_dir_label), indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

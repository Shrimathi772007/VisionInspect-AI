"""Build a SYNTHETIC dataset for a dry run of run_box_eval.py, from train/good images only.

    python scripts/experiments/localization_map/build_synthetic_dataset.py --out DIR [--categories tile,bottle]
                                                                            [--per-split 2]

For each category: test/good/ = copies of train/good images; test/synthetic_patch/ = other train/good copies with a
painted rectangle, and ground_truth/synthetic_patch/<stem>_mask.png marking it. Never reads test/ or ground_truth/
of the real dataset. Writes the SYNTHETIC_DATASET marker that run_box_eval.py requires for a non-project root.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SYNTHETIC_MARKER = "SYNTHETIC_DATASET"


def paint_defect(image: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """A filled rectangle (about 4% of the image) at a position that depends on `index`, and its mask."""
    height, width = image.shape[:2]
    w, h = width // 5, height // 5
    x = int((0.2 + 0.4 * (index % 2)) * width)
    y = int((0.25 + 0.3 * ((index // 2) % 2)) * height)
    painted = image.copy()
    colour = (20, 200, 240) if index % 2 == 0 else (240, 40, 200)
    cv2.rectangle(painted, (x, y), (x + w - 1, y + h - 1), colour, thickness=-1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return painted, mask


def build(out: Path, categories, per_split: int, source_root: Path = REPO_ROOT / "dataset") -> None:
    out = Path(out)
    if out.resolve() == source_root.resolve():
        raise SystemExit("Refusing to write into the real dataset.")
    for category in categories:
        train = sorted((source_root / category / "train" / "good").glob("*.png"))
        if len(train) < 2 * per_split:
            raise SystemExit(f"{category}: needs {2 * per_split} train/good images")
        good_dir = out / category / "test" / "good"
        defect_dir = out / category / "test" / "synthetic_patch"
        mask_dir = out / category / "ground_truth" / "synthetic_patch"
        for d in (good_dir, defect_dir, mask_dir):
            d.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(train[:per_split]):
            shutil.copyfile(src, good_dir / f"{i:03d}.png")
        for i, src in enumerate(train[per_split:2 * per_split]):
            image = cv2.imread(str(src), cv2.IMREAD_COLOR)
            painted, mask = paint_defect(image, i)
            cv2.imwrite(str(defect_dir / f"{i:03d}.png"), painted)
            cv2.imwrite(str(mask_dir / f"{i:03d}_mask.png"), mask)
    (out / SYNTHETIC_MARKER).write_text("Synthetic dry-run data built from train/good copies only.\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--categories", default="tile,bottle")
    parser.add_argument("--per-split", type=int, default=2)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    build(args.out, [c.strip() for c in args.categories.split(",") if c.strip()], args.per_split)
    print(f"Synthetic dataset written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

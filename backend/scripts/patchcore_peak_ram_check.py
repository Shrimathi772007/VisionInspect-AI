"""Peak-RAM / timing check for the WideResNet-50-2 PatchCore detector on train/good images only.

Fits a 10% greedy-coreset memory bank on the first N (default 100) images of
dataset/<category>/train/good (sorted by filename) and prints extraction time, coreset time, bank shape and
the peak process RAM. Nothing is written anywhere - in particular nothing under ai_models/. Only the
train/good folder is ever listed or read; the category's evaluation split is never touched.

Peak RAM comes from the Windows process counters (GetProcessMemoryInfo, via ctypes - nothing to install).
On other platforms the peak is reported as unavailable.

Usage (from backend/):
    python scripts/patchcore_peak_ram_check.py bottle [--count 100] [--mode crop224|full256]
"""

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.models.patchcore import PatchCoreConfig, PatchCoreDetector  # noqa: E402
from app.ai.preprocessing.patchcore_preprocess import MODES, preprocess_for_patchcore  # noqa: E402
from app.dataset.categories import MVTEC_CATEGORIES  # noqa: E402
from app.inspections.storage import DATASET_ROOT  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def peak_ram_mb() -> float | None:
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
        ]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return round(counters.PeakWorkingSetSize / 2**20, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("category", choices=MVTEC_CATEGORIES)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--mode", choices=MODES, default="crop224")
    args = parser.parse_args()

    train_good = DATASET_ROOT / args.category / "train" / "good"
    files = sorted(p for p in train_good.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)[: args.count]
    if len(files) < args.count:
        print(f"Only {len(files)} train/good images available; using all of them.")

    started = time.perf_counter()
    detector = PatchCoreDetector(PatchCoreConfig(preprocessing=args.mode))
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    features = detector.extract_features_into((preprocess_for_patchcore(p, args.mode) for p in files), len(files))
    extract_s = time.perf_counter() - started

    started = time.perf_counter()
    bank = detector.fit_from_features(features)
    coreset_s = time.perf_counter() - started

    print(json.dumps({
        "category": args.category,
        "mode": args.mode,
        "images": len(files),
        "backbone_load_seconds": round(load_s, 1),
        "extraction_seconds": round(extract_s, 1),
        "extraction_ms_per_image": round(extract_s * 1000 / len(files), 1),
        "features_shape": list(features.shape),
        "features_mb": round(features.numel() * 4 / 2**20, 1),
        "coreset_seconds": round(coreset_s, 1),
        "bank_shape": list(bank.shape),
        "bank_mb": round(bank.numel() * 4 / 2**20, 1),
        "peak_process_ram_mb": peak_ram_mb(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

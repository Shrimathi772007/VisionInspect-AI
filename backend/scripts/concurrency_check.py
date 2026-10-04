"""Concurrency check of POST /inspections/upload against an ALREADY RUNNING backend.

Starts nothing itself. Uploads copies of dataset/<category>/train/good/ images (never the test split) for a
tile product and a bottle product - both must already exist with those categories - at concurrency 1, 5 and
10 (threads), 10 requests per level, alternating tile and bottle, and prints min / median / p95 / max
latency and error counts per level as a table, plus the CPU count. The same numbers are written as JSON to
a file under the system temp directory (never into the repository).

The bearer token of a THROWAWAY quality engineer is read from the environment variable named by
--token-env (default VERIFYCHECK_QE_TOKEN) and is never printed or written. Every upload creates a real
inspection for the given products; deleting them afterwards is the caller's job.

    python scripts/concurrency_check.py --base-url http://127.0.0.1:8000 --tile-product-id 12 --bottle-product-id 13
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", str(BACKEND_DIR.parent / "dataset"))).resolve()
LEVELS = (1, 5, 10)
REQUESTS_PER_LEVEL = 10


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile (p95 of 10 values is the largest)."""
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * fraction // 1))  # ceil(n * fraction)
    return ordered[int(rank) - 1]


def train_good_copy(category: str, workdir: Path) -> Path:
    source = DATASET_ROOT / category / "train" / "good" / "000.png"
    target = workdir / f"{category}_train_good_000.png"
    target.write_bytes(source.read_bytes())
    return target


def upload_once(client: httpx.Client, headers: dict, product_id: int, image: Path) -> tuple[float, int | None]:
    start = time.perf_counter()
    try:
        with open(image, "rb") as handle:
            response = client.post("/inspections/upload", headers=headers, data={"product_id": str(product_id)},
                                   files={"file": (image.name, handle, "image/png")})
        status = response.status_code
    except httpx.HTTPError:
        status = None
    return (time.perf_counter() - start) * 1000, status


def run_level(base_url: str, headers: dict, jobs: list[tuple[int, Path]], concurrency: int) -> dict:
    with httpx.Client(base_url=base_url, timeout=300) as client, ThreadPoolExecutor(max_workers=concurrency) as pool:
        started = time.perf_counter()
        results = list(pool.map(lambda job: upload_once(client, headers, *job), jobs))
        wall = (time.perf_counter() - started) * 1000
    latencies = [ms for ms, status in results if status == 201]
    errors = sum(1 for _ms, status in results if status != 201)
    statuses = sorted({str(status) for _ms, status in results if status != 201})
    summary = {"concurrency": concurrency, "requests": len(jobs), "ok": len(latencies), "errors": errors,
               "error_statuses": statuses, "wall_ms": round(wall, 1)}
    if latencies:
        summary.update(min_ms=round(min(latencies), 1), median_ms=round(statistics.median(latencies), 1),
                       p95_ms=round(percentile(latencies, 0.95), 1), max_ms=round(max(latencies), 1))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tile-product-id", type=int, required=True)
    parser.add_argument("--bottle-product-id", type=int, required=True)
    parser.add_argument("--token-env", default="VERIFYCHECK_QE_TOKEN")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        print(f"Set {args.token_env} to a throwaway quality engineer's bearer token.", file=sys.stderr)
        return 2
    headers = {"Authorization": f"Bearer {token}"}

    workdir = Path(tempfile.mkdtemp(prefix="concurrency_check_"))
    images = {"tile": train_good_copy("tile", workdir), "bottle": train_good_copy("bottle", workdir)}
    products = {"tile": args.tile_product_id, "bottle": args.bottle_product_id}

    # Warm-up (not measured): loads both models once so the first level does not measure cold loading.
    with httpx.Client(base_url=args.base_url, timeout=300) as client:
        for category in ("tile", "bottle"):
            upload_once(client, headers, products[category], images[category])

    rows = []
    for level in LEVELS:
        jobs = [(products[c], images[c]) for c in (("tile", "bottle") * REQUESTS_PER_LEVEL)[:REQUESTS_PER_LEVEL]]
        rows.append(run_level(args.base_url, headers, jobs, level))

    cpu_count = os.cpu_count()
    print(f"CPU count: {cpu_count}   requests per level: {REQUESTS_PER_LEVEL} (alternating tile / bottle)")
    print(f"{'concurrency':>11} {'ok':>4} {'errors':>6} {'min ms':>9} {'median ms':>10} {'p95 ms':>9} {'max ms':>9} {'wall ms':>9}")
    for row in rows:
        print(f"{row['concurrency']:>11} {row['ok']:>4} {row['errors']:>6} {row.get('min_ms', '-'):>9} "
              f"{row.get('median_ms', '-'):>10} {row.get('p95_ms', '-'):>9} {row.get('max_ms', '-'):>9} {row['wall_ms']:>9}")

    out = Path(tempfile.gettempdir()) / f"concurrency_check_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    out.write_text(json.dumps({"cpu_count": cpu_count, "levels": rows, "warmup_uploads": 2}, indent=2), encoding="utf-8")
    print(f"JSON written to {out}")
    for image in images.values():
        image.unlink(missing_ok=True)
    workdir.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())

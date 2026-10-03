import time, statistics
from pathlib import Path
import app.ai.inference.predict as P
from app.ai.inference.serving import get_serving_config, load_serving_model, clear_model_cache
from app.ai.training.artifacts import get_model_path
from app.inspections.storage import DATASET_ROOT

cfg = get_serving_config("bottle")
print("model path:", get_model_path("bottle", cfg.artifact_name)); print("threshold:", repr(cfg.threshold), "model_name:", cfg.model_name)

# count image decodes per call
n = {"c": 0}; orig = P.process_image
def counting(*a, **k): n["c"] += 1; return orig(*a, **k)
P.process_image = counting
import app.ai.training.dataset as D
def boom(*a, **k): raise AssertionError("training set scanned")
D.discover_train_samples = boom

samples = [("good/000.png", "test/good/000.png"), ("good/001.png","test/good/001.png"),
           ("contamination/000.png","test/contamination/000.png"), ("broken_large/000.png","test/broken_large/000.png"),
           ("broken_small/000.png","test/broken_small/000.png")]
clear_model_cache()
for i, (name, rel) in enumerate(samples):
    n["c"] = 0
    r = P.predict_image(DATASET_ROOT / "bottle" / rel, "bottle")
    print(f"{rel:32s} pred={r.prediction:9s} err={r.reconstruction_error:.6f} thr={r.threshold:.10f} decodes={n['c']} time={r.processing_time_ms:.1f}ms" + ("  (cold: model load + MD5 verify)" if i == 0 else ""))

warm = []
for _ in range(20):
    warm.append(P.predict_image(DATASET_ROOT / "bottle/test/good/000.png", "bottle").processing_time_ms)
print(f"warm predict_image x20: mean {statistics.mean(warm):.1f} ms, median {statistics.median(warm):.1f} ms, min {min(warm):.1f}, max {max(warm):.1f}")

# Old behaviour cost (measured here for comparison only; nothing served): reprocess all train/good for a threshold
D.discover_train_samples = None
import importlib; importlib.reload(D)
from app.ai.evaluation.evaluate import compute_reconstruction_error
from app.ai.evaluation.threshold import compute_threshold
from app.ai.training.artifacts import load_model_for_category
model = load_model_for_category("bottle")  # Phase 1 model, read-only
train = D.discover_train_samples("bottle")
t = time.perf_counter()
errs = [compute_reconstruction_error(model, P._image_to_tensor(s.path, (128,128))) for s in train]
old_ms = (time.perf_counter() - t) * 1000
print(f"OLD-style threshold scan over {len(train)} train/good images: {old_ms:.0f} ms (threshold K=3 -> {compute_threshold(errs):.6f})")

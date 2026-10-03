import statistics, time, torch
from app.ai.evaluation.evaluate import compute_reconstruction_error
from app.ai.training import discover_train_samples
from app.ai.training.artifacts import load_model
from app.ai.training.dataset import load_sample
from app.ai.training.model import build_model
from app.ai.training.validation_split import split_train_validation

split = split_train_validation(discover_train_samples("hazelnut"), 0.8, 42)  # validation images ONLY
tensors = []
prep = []
for s in split.validation:
    t0 = time.perf_counter()
    r = load_sample(s, target_size=(128, 128))
    tensors.append(torch.from_numpy(r.preprocessing.normalized_image).permute(2, 0, 1).contiguous())
    prep.append((time.perf_counter() - t0) * 1000)
paths = {"phase3_baseline": "ai_models/hazelnut/phase3_validation/autoencoder.pt",
         "phase4_selected": "ai_models/hazelnut/phase4_optimization/selected_candidate/autoencoder.pt"}
models = {k: load_model(p, build_model()) for k, p in paths.items()}
res = {k: [] for k in models}
for rep in range(7):                      # alternate models each repeat to share machine load
    for k, m in models.items():
        t0 = time.perf_counter()
        for x in tensors: compute_reconstruction_error(m, x)
        res[k].append((time.perf_counter() - t0) * 1000 / len(tensors))
print("preprocess ms/image (128x128 path, validation):", round(statistics.median(prep), 2))
for k, v in res.items():
    print(k, "forward-pass ms/image per repeat:", [round(x, 2) for x in v], "median", round(statistics.median(v), 2))
load = {k: [] for k in paths}
for _ in range(7):
    for k, p in paths.items():
        t0 = time.perf_counter(); load_model(p, build_model()); load[k].append((time.perf_counter() - t0) * 1000)
for k, v in load.items(): print(k, "model load ms median", round(statistics.median(v), 2))

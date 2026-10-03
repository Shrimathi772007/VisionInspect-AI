p = "app/ai/evaluation/category_phase1.py"
s = open(p, encoding="utf-8").read()

# 1) helpers: dataset fingerprint + gate classification
anchor = "def describe(values) -> dict:"
helpers = '''def dataset_fingerprint(samples, counts: dict) -> str:
    """Content fingerprint of the training data (every train/good file's SHA-256, in sorted order) plus the counts."""
    h = hashlib.sha256()
    for sample in samples:
        h.update(f"{sample.path.name}:{hashlib.sha256(sample.path.read_bytes()).hexdigest()}\\n".encode())
    h.update(json.dumps({k: counts[k] for k in ("train_good", "test_good", "test_defective", "test_total")}, sort_keys=True).encode())
    return h.hexdigest()


def gate_classification(recall: float, f1: float, fpr: float) -> str:
    """Project evidence gates (interpretation only): EXCELLENT / GOOD / ACCEPTABLE / NOT PRODUCTION READY."""
    if recall >= 0.90 and f1 >= 0.85 and fpr <= 0.10:
        return "EXCELLENT"
    if recall >= 0.85 and f1 >= 0.80 and fpr <= 0.10:
        return "GOOD"
    if recall >= 0.75 and f1 >= 0.70 and fpr <= 0.15:
        return "ACCEPTABLE"
    return "NOT PRODUCTION READY"


'''
assert anchor in s and "def dataset_fingerprint" not in s
s = s.replace(anchor, helpers + anchor, 1)

# 2) json import already present? ensure os import for cpu count
if "\nimport os\n" not in s:
    s = s.replace("import json\nimport platform", "import json\nimport os\nimport platform", 1)

# 3) record extra fields in the returned dict
old_env = '''"python": platform.python_version(), "torch_threads": torch.get_num_threads(), "platform": platform.platform()},'''
new_env = '''"python": platform.python_version(), "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(), "cpu_count": os.cpu_count(),
                        "processor": platform.processor(), "platform": platform.platform()},
        "dataset_fingerprint_sha256": dataset_fingerprint(train_samples, counts),
        "training_files": [s.path.name for s in train_samples], "training_files_sha256": hashlib.sha256("\\n".join(s.path.name for s in train_samples).encode()).hexdigest(),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "gate_classification": gate_classification(metrics["recall"], metrics["f1_score"], metrics["false_defect_detection_rate"]),'''
assert old_env in s
s = s.replace(old_env, new_env, 1)

# 4) extend compare_runs
old_cmp = '''        "loss_history_identical": a["training"]["loss_history"] == b["training"]["loss_history"],'''
new_cmp = '''        "dataset_fingerprint_identical": a["dataset_fingerprint_sha256"] == b["dataset_fingerprint_sha256"],
        "training_order_identical": a["training_files"] == b["training_files"] and a["training_files_sha256"] == b["training_files_sha256"],
        "model_configuration_identical": a["config"] == b["config"] and a["parameter_count"] == b["parameter_count"],
        "loss_history_identical": a["training"]["loss_history"] == b["training"]["loss_history"],'''
assert old_cmp in s
s = s.replace(old_cmp, new_cmp, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched")

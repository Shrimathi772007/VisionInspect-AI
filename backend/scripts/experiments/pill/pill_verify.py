import json, pathlib, sys
sys.path.insert(0, ".")
from app.ai.evaluation import model_family_final as fin, model_family_pipeline as pipe, model_family_study as mfs
from app.ai.evaluation.category_phase1 import file_hashes

root = pathlib.Path("ai_models/pill/model_family_study")
L = json.loads((root / "selection_lock.json").read_bytes().decode())
S = json.loads((root / "reports/model_family_study.json").read_bytes().decode())
R = json.loads((root / "reports/final_test_result.json").read_bytes().decode())
print("lock digest valid          :", pipe.lock_digest(L) == L["lock_digest"])
re_sel = mfs.select_configuration({c: r["summary"] for c, r in S["candidates"].items()}, mfs.registry())
print("re-selection == lock       :", re_sel["winner"] == f"{L['selected_candidate']}|{L['selected_policy']}" and re_sel["threshold"] == L["selected_threshold"])
print("lock unchanged since final :", fin.verify_lock_unchanged_since_final_test("pill"))
print("result -> lock file sha    :", R["selection_lock_file_sha256"] == file_hashes(root / "selection_lock.json")["sha256"], "| digest", R["selection_lock_digest"] == L["lock_digest"])
sel = file_hashes(root / "selected_candidate/model_state.pt")["sha256"]
print("selected artifact          :", sel == L["selected_artifact"]["sha256"] == R["artifact_sha256"] == S["candidates"]["gauss_l23_256"]["artifact"]["sha256"], sel)
print("backbone                   :", file_hashes(fin.pretrained_weights_path())["sha256"] == L["backbone"]["sha256"] == R["backbone_sha256"], fin.pretrained_weights_path().stat().st_size)
print("candidate hashes match study:", all(file_hashes(root / "candidates" / c / "model_state.pt")["sha256"] == S["candidates"][c]["artifact"]["sha256"] for c in S["candidates"]))
print("phase1 pill artifacts      :", pipe.prior_phase_reference("pill") == L["prior_phase_artifacts"])
print("candidate sha256 (8):")
for c in S["candidates"]:
    a = S["candidates"][c]["artifact"]
    print("  ", c, a["sha256"], a["size_bytes"])
print("selected model_state md5/size:", L["selected_artifact"]["md5"], L["selected_artifact"]["size_bytes"])
print("threshold_position:", R["threshold_position"])
print("selected_configuration:", R["selected_configuration"])

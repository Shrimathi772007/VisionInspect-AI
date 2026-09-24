"""Read-only parity check of the Phase 4 engine against the real, protected Hazelnut Phase 3 baseline.

Kept in its own module because test_ai_category_phase4.py's module-scoped fixture redirects the dataset
and artifact roots to a synthetic category for the whole module.
"""

import json

import pytest

from app.ai.evaluation.category_phase3 import PHASE3_SPLIT_SEED, PHASE3_TRAINING_FRACTION, resolve_outputs
from app.ai.evaluation.category_phase4 import Phase3Reference, run_candidate
from app.ai.training import category_phase4_candidates as cands
from app.ai.training import discover_train_samples
from app.ai.training.validation_split import split_train_validation

# ---------------------------------------------------------------------------

_P3 = resolve_outputs("hazelnut")


@pytest.mark.skipif(
    not (_P3.model_path.is_file() and _P3.report_path.is_file()), reason="Hazelnut Phase 3 artifacts not present"
)
def test_engine_baseline_candidate_reproduces_hazelnut_phase3_validation_statistics(tmp_path):
    report = json.loads(_P3.report_path.read_text(encoding="utf-8"))
    split = split_train_validation(discover_train_samples("hazelnut"), PHASE3_TRAINING_FRACTION, PHASE3_SPLIT_SEED)
    assert (len(split.training), len(split.validation)) == (313, 78)
    ref = Phase3Reference(
        model_path=_P3.model_path,
        training_duration_s=report["training_result"]["duration_seconds"],
        final_loss=report["training_result"]["final_loss"],
        num_training_images=report["training_result"]["num_training_images"],
    )
    outcome = run_candidate(cands.make_candidate("hazelnut", "s1", (128, 128), 15, 1e-3), split, tmp_path / "unused.pt", ref)

    stats = report["validation_statistics"]
    assert outcome.validation_mean == pytest.approx(stats["mean_error"], rel=1e-9)
    assert outcome.validation_std == pytest.approx(stats["std_error"], rel=1e-9)
    assert outcome.selection.selected.threshold == report["threshold_selection"]["selected_threshold"]
    assert outcome.model_path == _P3.model_path and not (tmp_path / "unused.pt").exists()

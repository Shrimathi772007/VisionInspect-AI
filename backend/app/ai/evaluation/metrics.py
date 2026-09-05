"""Real classification metrics via scikit-learn.

"Defective" (label 1) is treated as the positive class - in an anomaly
detection setting, that's the class we actually care about detecting.
"""

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

GOOD_LABEL = 0
DEFECTIVE_LABEL = 1


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> ClassificationMetrics:
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, pos_label=DEFECTIVE_LABEL, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=DEFECTIVE_LABEL, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=DEFECTIVE_LABEL, zero_division=0))

    # labels=[0, 1] fixes row/column order: row 0 = actual good, row 1 = actual
    # defective; col 0 = predicted good, col 1 = predicted defective.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[GOOD_LABEL, DEFECTIVE_LABEL]).ravel()

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )

"""Offline evaluation.

Compare predicted affect labels against ground-truth labels and compute
standard classification metrics. This is separate from the live inference
path because it requires labelled data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Aggregate evaluation metrics."""

    accuracy: float
    per_label_f1: dict[str, float]
    support: dict[str, int]
    num_samples: int


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def evaluate(y_true: list[str], y_pred: list[str]) -> EvaluationResult:
    """Compute accuracy and per-label F1.

    Parameters
    ----------
    y_true, y_pred:
        Parallel lists of label strings of equal length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: {len(y_true)} truths vs {len(y_pred)} predictions"
        )
    n = len(y_true)
    if n == 0:
        return EvaluationResult(accuracy=0.0, per_label_f1={}, support={}, num_samples=0)

    correct = sum(t == p for t, p in zip(y_true, y_pred))

    labels = sorted(set(y_true) | set(y_pred))
    per_label_f1: dict[str, float] = {}
    support: dict[str, int] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        per_label_f1[label] = _f1(tp, fp, fn)
        support[label] = sum(t == label for t in y_true)

    return EvaluationResult(
        accuracy=correct / n,
        per_label_f1=per_label_f1,
        support=support,
        num_samples=n,
    )

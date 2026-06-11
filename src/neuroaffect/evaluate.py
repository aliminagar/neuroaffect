"""Stage 4 — offline evaluation.

Two layers:

1. :func:`evaluate` — pure metrics from two parallel label lists: overall
   accuracy, macro-F1, and per-class precision/recall/F1 + a confusion matrix.
   No heavy dependencies, so it stays trivially unit-testable.
2. :func:`evaluate_dataset` — run the Stage 2 classifier against a real
   labelled test set (FER-2013) and report **honest** metrics. Whatever the
   model scores is what we report; nothing here is synthetic.

:func:`plot_confusion_matrix` renders the confusion matrix to a PNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    """Aggregate evaluation metrics."""

    accuracy: float
    num_samples: int
    macro_f1: float = 0.0
    per_label_precision: dict[str, float] = field(default_factory=dict)
    per_label_recall: dict[str, float] = field(default_factory=dict)
    per_label_f1: dict[str, float] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    # confusion[true_label][pred_label] = count
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)


def _div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def evaluate(
    y_true: list[str],
    y_pred: list[str],
    *,
    labels: list[str] | None = None,
) -> EvaluationResult:
    """Compute accuracy, macro-F1, and per-label precision/recall/F1.

    Parameters
    ----------
    y_true, y_pred:
        Parallel lists of label strings of equal length.
    labels:
        Fixed label set/order to report over. Defaults to the sorted union of
        the labels seen. Pass an explicit set (e.g. all 7 affect labels) so a
        class that never appears still shows up with zero support.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: {len(y_true)} truths vs {len(y_pred)} predictions"
        )
    n = len(y_true)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    if n == 0:
        return EvaluationResult(accuracy=0.0, num_samples=0, labels=list(labels))

    correct = sum(t == p for t, p in zip(y_true, y_pred))

    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in confusion and p in confusion[t]:
            confusion[t][p] += 1

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    support: dict[str, int] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision[label] = _div(tp, tp + fp)
        recall[label] = _div(tp, tp + fn)
        f1[label] = _div(2 * tp, 2 * tp + fp + fn)
        support[label] = sum(t == label for t in y_true)

    macro_f1 = _div(sum(f1.values()), len(labels))

    return EvaluationResult(
        accuracy=correct / n,
        num_samples=n,
        macro_f1=macro_f1,
        per_label_precision=precision,
        per_label_recall=recall,
        per_label_f1=f1,
        support=support,
        labels=list(labels),
        confusion=confusion,
    )


def plot_confusion_matrix(
    result: EvaluationResult,
    out_path: str,
    *,
    normalize: bool = True,
    title: str | None = None,
    dpi: int = 120,
) -> str:
    """Render ``result.confusion`` as a heatmap PNG (row-normalised by default)."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np

    labels = result.labels
    counts = np.array(
        [[result.confusion[t][p] for p in labels] for t in labels], dtype=float
    )
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    else:
        matrix = counts

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() or 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="row-normalised rate" if normalize else "count")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        title
        or f"FER confusion — acc {result.accuracy:.3f}, "
        f"macro-F1 {result.macro_f1:.3f} (n={result.num_samples})"
    )

    thresh = (matrix.max() or 1) / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            cell = f"{matrix[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            ax.text(
                j, i, cell, ha="center", va="center", fontsize=8,
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_dataset(
    dataset: str = "fer2013",
    *,
    limit: int | None = None,
    temperature: float = 1.0,
) -> tuple[EvaluationResult, dict]:
    """Run the Stage 2 classifier on a labelled test set and score it.

    Classifies each (pre-cropped, aligned) face image directly — this measures
    the classifier in isolation, the standard FER benchmark protocol, rather
    than the end-to-end detect->classify path.

    Returns ``(result, info)`` where ``info`` records the dataset and how many
    of the available samples were actually used.
    """
    if dataset != "fer2013":
        raise ValueError(f"unknown dataset: {dataset!r} (supported: 'fer2013')")

    from neuroaffect.classification import AFFECT_LABELS, classify_face
    from neuroaffect.datasets import load_fer2013_test
    from neuroaffect.detection import BoundingBox

    samples, total = load_fer2013_test(limit=limit)

    y_true: list[str] = []
    y_pred: list[str] = []
    for image, true_label in samples:
        h, w = image.shape[:2]
        pred = classify_face(image, BoundingBox(0, 0, w, h), temperature=temperature)
        y_true.append(true_label)
        y_pred.append(pred.label)

    result = evaluate(y_true, y_pred, labels=sorted(AFFECT_LABELS))
    info = {
        "dataset": dataset,
        "samples_used": len(y_true),
        "total_available": total,
        "limited": limit is not None and limit < total,
    }
    return result, info

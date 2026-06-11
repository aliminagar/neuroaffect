"""Stage 3 — aggregation.

Combine per-face predictions across a frame (or a sequence of frames) into
a compact summary: dominant affect and the mean score per label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from neuroaffect.classification import AFFECT_LABELS, AffectPrediction


@dataclass
class AffectSummary:
    """Aggregated affect over a set of predictions."""

    num_faces: int
    dominant_label: str
    label_counts: dict[str, int] = field(default_factory=dict)
    mean_scores: dict[str, float] = field(default_factory=dict)


def aggregate(predictions: list[AffectPrediction]) -> AffectSummary:
    """Aggregate a flat list of per-face predictions.

    The dominant label is the most frequently predicted one; ties are broken
    by the higher mean score. ``mean_scores`` averages each label's
    probability across all predictions.
    """
    if not predictions:
        return AffectSummary(num_faces=0, dominant_label="none")

    counts = Counter(p.label for p in predictions)

    mean_scores: dict[str, float] = {}
    for label in AFFECT_LABELS:
        total = sum(p.scores.get(label, 0.0) for p in predictions)
        mean_scores[label] = total / len(predictions)

    dominant = max(counts, key=lambda lbl: (counts[lbl], mean_scores.get(lbl, 0.0)))

    return AffectSummary(
        num_faces=len(predictions),
        dominant_label=dominant,
        label_counts=dict(counts),
        mean_scores=mean_scores,
    )

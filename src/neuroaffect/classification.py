"""Stage 2 — affect classification.

Given a detected face crop, predict its affect (emotion) label and the
probability distribution over the supported labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroaffect.detection import BoundingBox, Frame

# Canonical affect label set (Ekman's basic emotions + neutral).
AFFECT_LABELS: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprise",
    "fear",
    "disgust",
)


@dataclass(frozen=True)
class AffectPrediction:
    """Predicted affect for a single face."""

    box: BoundingBox
    label: str
    scores: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.scores.get(self.label, 0.0)


def classify_face(frame: Frame, box: BoundingBox) -> AffectPrediction:
    """Classify the affect of the face inside ``box``.

    Stub implementation: returns a uniform distribution over
    :data:`AFFECT_LABELS`. Replace the body with a real classifier
    (e.g. a CNN over the normalized face crop).
    """
    face = box.crop(frame)
    if face.size == 0:
        raise ValueError("bounding box does not overlap the frame")

    # TODO: replace with real model inference over ``face``.
    uniform = 1.0 / len(AFFECT_LABELS)
    scores = {label: uniform for label in AFFECT_LABELS}
    label = max(scores, key=scores.get)
    return AffectPrediction(box=box, label=label, scores=scores)


def classify_faces(
    frame: Frame, boxes: list[BoundingBox]
) -> list[AffectPrediction]:
    """Classify every detected face in a frame."""
    return [classify_face(frame, box) for box in boxes]

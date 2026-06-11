import numpy as np
import pytest

from neuroaffect.classification import AFFECT_LABELS, classify_face, classify_faces
from neuroaffect.detection import BoundingBox


def _frame():
    return np.zeros((32, 32, 3), dtype=np.uint8)


def test_classify_face_returns_full_distribution():
    pred = classify_face(_frame(), BoundingBox(0, 0, 16, 16))
    assert set(pred.scores) == set(AFFECT_LABELS)
    assert pred.label in AFFECT_LABELS
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-6


def test_classify_face_rejects_empty_crop():
    with pytest.raises(ValueError):
        classify_face(_frame(), BoundingBox(100, 100, 10, 10))


def test_classify_faces_maps_over_boxes():
    boxes = [BoundingBox(0, 0, 8, 8), BoundingBox(8, 8, 8, 8)]
    preds = classify_faces(_frame(), boxes)
    assert len(preds) == 2

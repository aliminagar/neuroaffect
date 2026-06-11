import numpy as np
import pytest

from neuroaffect.classification import (
    AFFECT_LABELS,
    AffectPrediction,
    _normalize_label,
    _scores_from_logits,
    _softmax,
    classify_face,
    classify_faces,
)
from neuroaffect.detection import BoundingBox


def _frame():
    return np.zeros((32, 32, 3), dtype=np.uint8)


# --- pure helpers: no torch / transformers / network needed ----------------


def test_softmax_sums_to_one_and_is_ordered():
    p = _softmax(np.array([1.0, 2.0, 3.0]), temperature=1.0)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[2] > p[1] > p[0]


def test_softmax_temperature_flattens_distribution():
    logits = np.array([1.0, 2.0, 3.0])
    sharp = _softmax(logits, temperature=1.0)
    flat = _softmax(logits, temperature=10.0)
    # Higher temperature -> closer to uniform -> lower peak probability.
    assert flat.max() < sharp.max()


def test_softmax_rejects_nonpositive_temperature():
    with pytest.raises(ValueError):
        _softmax(np.array([1.0, 2.0]), temperature=0.0)


def test_normalize_label_handles_aliases_and_case():
    assert _normalize_label("Happy") == "happy"
    assert _normalize_label("anger") == "angry"
    assert _normalize_label("surprised") == "surprise"
    assert _normalize_label("contempt") is None  # not in our label set


def test_scores_from_logits_maps_to_canonical_labels():
    id2label = {0: "sad", 1: "disgust", 2: "angry", 3: "neutral",
                4: "fear", 5: "surprise", 6: "happy"}
    # Make "happy" (index 6) clearly dominant.
    logits = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    scores = _scores_from_logits(logits, id2label, temperature=1.0)
    assert set(scores) == set(AFFECT_LABELS)
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    assert max(scores, key=scores.get) == "happy"


def test_scores_from_logits_drops_unknown_label_and_renormalizes():
    # An extra "contempt" class should be discarded, remainder renormalised.
    id2label = {0: "happy", 1: "sad", 2: "contempt"}
    logits = np.array([1.0, 1.0, 5.0])  # most mass on the dropped class
    scores = _scores_from_logits(logits, id2label, temperature=1.0)
    assert set(scores) == set(AFFECT_LABELS)  # only canonical labels survive
    assert "contempt" not in scores
    assert abs(sum(scores.values()) - 1.0) < 1e-6  # renormalised after the drop
    assert abs(scores["happy"] - scores["sad"]) < 1e-9  # equal surviving logits


def test_affect_prediction_confidence():
    scores = {label: 0.0 for label in AFFECT_LABELS}
    scores["happy"] = 1.0
    pred = AffectPrediction(box=BoundingBox(0, 0, 8, 8), label="happy", scores=scores)
    assert pred.confidence == 1.0


# --- end-to-end against the real model (needs torch/transformers + weights) -


@pytest.mark.integration
def test_classify_face_returns_full_distribution():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pred = classify_face(_frame(), BoundingBox(0, 0, 16, 16))
    assert set(pred.scores) == set(AFFECT_LABELS)
    assert pred.label in AFFECT_LABELS
    assert abs(sum(pred.scores.values()) - 1.0) < 1e-6


def test_classify_face_rejects_empty_crop():
    # Validation happens before any model load — no heavy deps required.
    with pytest.raises(ValueError):
        classify_face(_frame(), BoundingBox(100, 100, 10, 10))


@pytest.mark.integration
def test_classify_faces_maps_over_boxes():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    boxes = [BoundingBox(0, 0, 8, 8), BoundingBox(8, 8, 8, 8)]
    preds = classify_faces(_frame(), boxes)
    assert len(preds) == 2

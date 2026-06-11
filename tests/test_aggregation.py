from neuroaffect.aggregation import aggregate
from neuroaffect.classification import AFFECT_LABELS, AffectPrediction
from neuroaffect.detection import BoundingBox


def _pred(label):
    scores = {lbl: 0.0 for lbl in AFFECT_LABELS}
    scores[label] = 1.0
    return AffectPrediction(box=BoundingBox(0, 0, 1, 1), label=label, scores=scores)


def test_aggregate_empty():
    summary = aggregate([])
    assert summary.num_faces == 0
    assert summary.dominant_label == "none"


def test_aggregate_picks_most_frequent_label():
    preds = [_pred("happy"), _pred("happy"), _pred("sad")]
    summary = aggregate(preds)
    assert summary.num_faces == 3
    assert summary.dominant_label == "happy"
    assert summary.label_counts == {"happy": 2, "sad": 1}


def test_aggregate_mean_scores_average_over_predictions():
    preds = [_pred("happy"), _pred("sad")]
    summary = aggregate(preds)
    assert summary.mean_scores["happy"] == 0.5
    assert summary.mean_scores["sad"] == 0.5

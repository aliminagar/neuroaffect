import pytest

from neuroaffect.evaluate import evaluate


def test_evaluate_perfect_predictions():
    y = ["happy", "sad", "neutral"]
    result = evaluate(y, list(y))
    assert result.accuracy == 1.0
    assert all(f1 == 1.0 for f1 in result.per_label_f1.values())
    assert result.num_samples == 3


def test_evaluate_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate(["happy"], ["happy", "sad"])


def test_evaluate_empty():
    result = evaluate([], [])
    assert result.accuracy == 0.0
    assert result.num_samples == 0


def test_evaluate_half_correct():
    y_true = ["happy", "sad"]
    y_pred = ["happy", "happy"]
    result = evaluate(y_true, y_pred)
    assert result.accuracy == 0.5
    assert result.support == {"happy": 1, "sad": 1}

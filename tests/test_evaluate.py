import pytest

from neuroaffect.evaluate import evaluate, plot_confusion_matrix


def test_evaluate_perfect_predictions():
    y = ["happy", "sad", "neutral"]
    result = evaluate(y, list(y))
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
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


def test_evaluate_precision_recall_f1_values():
    # happy: tp=2, fp=1, fn=0  -> P=2/3, R=1, F1=0.8
    # sad:   tp=0, fp=0, fn=1  -> P=0,   R=0, F1=0
    y_true = ["happy", "happy", "sad"]
    y_pred = ["happy", "happy", "happy"]
    result = evaluate(y_true, y_pred)
    assert result.per_label_precision["happy"] == pytest.approx(2 / 3)
    assert result.per_label_recall["happy"] == 1.0
    assert result.per_label_f1["happy"] == pytest.approx(0.8)
    assert result.per_label_f1["sad"] == 0.0
    # macro-F1 averages over the (two) labels present.
    assert result.macro_f1 == pytest.approx((0.8 + 0.0) / 2)


def test_evaluate_fixed_labels_include_absent_class():
    # 'fear' never appears but must still be reported (zero support).
    result = evaluate(["happy"], ["happy"], labels=["happy", "fear"])
    assert result.support["fear"] == 0
    assert result.per_label_f1["fear"] == 0.0
    assert "fear" in result.confusion and result.confusion["fear"] == {"happy": 0, "fear": 0}


def test_evaluate_confusion_matrix_counts():
    y_true = ["happy", "happy", "sad"]
    y_pred = ["happy", "sad", "sad"]
    result = evaluate(y_true, y_pred, labels=["happy", "sad"])
    assert result.confusion["happy"] == {"happy": 1, "sad": 1}
    assert result.confusion["sad"] == {"happy": 0, "sad": 1}


def test_plot_confusion_matrix_writes_png(tmp_path):
    result = evaluate(
        ["happy", "sad", "happy"], ["happy", "sad", "sad"], labels=["happy", "sad"]
    )
    out = tmp_path / "cm.png"
    plot_confusion_matrix(result, str(out))
    assert out.exists() and out.stat().st_size > 0


# --- dataset loading / end-to-end benchmark (needs network + torch) --------


@pytest.mark.integration
def test_load_fer2013_subsample_size_and_labels():
    pytest.importorskip("huggingface_hub")
    from neuroaffect.classification import AFFECT_LABELS
    from neuroaffect.datasets import load_fer2013_test

    samples, total = load_fer2013_test(limit=12)
    assert total > 7000  # full FER-2013 test split
    assert len(samples) == 12
    for image, label in samples:
        assert image.ndim == 3 and image.shape[2] == 3
        assert label in AFFECT_LABELS  # FER class names normalised to ours


@pytest.mark.integration
def test_evaluate_dataset_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from neuroaffect.classification import AFFECT_LABELS
    from neuroaffect.evaluate import evaluate_dataset

    result, info = evaluate_dataset("fer2013", limit=16)
    assert info["samples_used"] == result.num_samples == 16
    assert info["total_available"] > 7000
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.labels == sorted(AFFECT_LABELS)


def test_evaluate_dataset_rejects_unknown():
    from neuroaffect.evaluate import evaluate_dataset

    with pytest.raises(ValueError):
        evaluate_dataset("raf_db")

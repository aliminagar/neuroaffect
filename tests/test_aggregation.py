import math

import pytest

from neuroaffect.aggregation import (
    AffectTimeline,
    FrameAffect,
    _draw_overlay,
    _running_readout,
    aggregate,
    aggregate_timeline,
    analyze_video,
    label_distribution,
    normalized_label_entropy,
    plot_timeline,
    shannon_entropy,
    valence_of,
)
from neuroaffect.classification import AFFECT_LABELS, AffectPrediction
from neuroaffect.detection import BoundingBox


def _pred(label):
    scores = {lbl: 0.0 for lbl in AFFECT_LABELS}
    scores[label] = 1.0
    return AffectPrediction(box=BoundingBox(0, 0, 1, 1), label=label, scores=scores)


# --- single-frame aggregate() (unchanged) ----------------------------------


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


# --- temporal helpers: pure, no torch / cv2 / network ----------------------


def _scores(label):
    d = {lbl: 0.0 for lbl in AFFECT_LABELS}
    d[label] = 1.0
    return d


def _frame(t, label):
    return FrameAffect(time=t, scores=_scores(label), dominant=label, face_score=0.9)


def _timeline(labels, dt=0.5, duration=None):
    frames = [_frame(i * dt, lbl) for i, lbl in enumerate(labels)]
    return AffectTimeline(
        frames=frames,
        sample_fps=1.0 / dt,
        duration_seconds=duration if duration is not None else len(labels) * dt,
    )


def test_shannon_entropy_known_values():
    assert shannon_entropy([1.0]) == 0.0
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert shannon_entropy([1 / 7] * 7) == pytest.approx(math.log2(7))


def test_shannon_entropy_no_negative_zero():
    # One-hot must give a clean +0.0, not -0.0.
    assert math.copysign(1.0, shannon_entropy([1.0])) == 1.0


def test_label_distribution_normalizes_over_all_labels():
    dist = label_distribution(["happy", "happy", "sad"])
    assert set(dist) == set(AFFECT_LABELS)
    assert dist["happy"] == pytest.approx(2 / 3)
    assert dist["sad"] == pytest.approx(1 / 3)
    assert dist["angry"] == 0.0


def test_normalized_label_entropy_flat_vs_labile():
    assert normalized_label_entropy(["happy"] * 5) == 0.0  # flat
    assert normalized_label_entropy([]) == 0.0
    # All seven labels equally -> maximally labile -> 1.0.
    assert normalized_label_entropy(list(AFFECT_LABELS)) == pytest.approx(1.0)
    # Two labels evenly -> 1 bit normalised by log2(7).
    two = normalized_label_entropy(["happy", "sad"])
    assert two == pytest.approx(1.0 / math.log2(7))


def test_valence_of_lexicon_extremes():
    assert valence_of(_scores("happy")) == pytest.approx(1.0)
    assert valence_of(_scores("sad")) == pytest.approx(-1.0)
    assert valence_of(_scores("neutral")) == pytest.approx(0.0)


def test_aggregate_timeline_flat_has_zero_variability():
    summary = aggregate_timeline(_timeline(["happy"] * 6), window_seconds=2.0, hop_seconds=1.0)
    assert summary.num_faces == 6
    assert summary.dominant_label == "happy"
    assert summary.variability == 0.0
    assert summary.valence_std == 0.0
    assert summary.valence_lability == 0.0
    assert all(w.dominant == "happy" for w in summary.windows)


def test_aggregate_timeline_labile_has_high_variability():
    summary = aggregate_timeline(
        _timeline(["happy", "sad"] * 5), window_seconds=2.0, hop_seconds=1.0
    )
    assert summary.variability == pytest.approx(1.0 / math.log2(7))
    assert summary.valence_lability == pytest.approx(2.0)  # |1 - (-1)| each step
    assert summary.valence_std == pytest.approx(1.0)


def test_aggregate_timeline_no_faces():
    frames = [FrameAffect(time=t * 0.5, scores=None, dominant=None) for t in range(4)]
    timeline = AffectTimeline(frames=frames, sample_fps=2.0, duration_seconds=2.0)
    summary = aggregate_timeline(timeline)
    assert summary.num_faces == 0
    assert summary.dominant_label == "none"
    assert summary.variability == 0.0
    assert summary.windows == []


def test_aggregate_timeline_windows_track_transition():
    # happy for 2s then sad for 2s, sampled at 2 fps.
    labels = ["happy"] * 4 + ["sad"] * 4
    summary = aggregate_timeline(
        _timeline(labels, dt=0.5, duration=4.0), window_seconds=2.0, hop_seconds=2.0
    )
    assert [w.dominant for w in summary.windows] == ["happy", "sad"]


def test_aggregate_timeline_rejects_bad_window():
    with pytest.raises(ValueError):
        aggregate_timeline(_timeline(["happy"]), window_seconds=0.0)


def test_plot_timeline_writes_png(tmp_path):
    timeline = _timeline(["happy", "happy", "sad", "sad"])
    summary = aggregate_timeline(timeline, window_seconds=1.0, hop_seconds=0.5)
    out = tmp_path / "timeline.png"
    plot_timeline(timeline, summary, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_timeline_raises_without_faces(tmp_path):
    frames = [FrameAffect(time=0.0, scores=None, dominant=None)]
    timeline = AffectTimeline(frames=frames, sample_fps=1.0, duration_seconds=1.0)
    summary = aggregate_timeline(timeline)
    with pytest.raises(ValueError):
        plot_timeline(timeline, summary, str(tmp_path / "x.png"))


# --- annotation overlay helpers --------------------------------------------


def test_running_readout_uses_only_recent_window():
    frames = [
        _frame(0.0, "happy"),
        _frame(0.5, "happy"),
        _frame(1.5, "sad"),
        _frame(2.0, "sad"),
    ]
    # window covers [1.0, 2.0] -> only the two 'sad' frames count.
    dom, val = _running_readout(frames, t=2.0, window_seconds=1.0)
    assert dom == "sad"
    assert val == pytest.approx(-1.0)


def test_running_readout_no_recent_faces():
    frames = [FrameAffect(time=0.0, scores=None, dominant=None)]
    assert _running_readout(frames, t=5.0, window_seconds=2.0) == ("none", 0.0)


def test_draw_overlay_modifies_frame():
    import numpy as np

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    _draw_overlay(
        frame,
        box=BoundingBox(20, 20, 40, 40),
        label="happy",
        score=0.9,
        run_dom="happy",
        run_val=0.5,
        t=1.0,
    )
    assert frame.sum() > 0  # something was drawn


def test_draw_overlay_without_box():
    import numpy as np

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    _draw_overlay(
        frame, box=None, label="", score=0.0, run_dom="none", run_val=0.0, t=0.0
    )
    assert frame.sum() > 0  # readout panel still drawn


# --- end-to-end video decode (needs cv2; detection needs mediapipe) --------


def test_analyze_video_rejects_unopenable_path():
    with pytest.raises(ValueError):
        analyze_video("does_not_exist.mp4")


@pytest.mark.integration
def test_analyze_video_on_faceless_clip(tmp_path):
    pytest.importorskip("mediapipe")
    import cv2
    import numpy as np

    path = tmp_path / "gray.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96)
    )
    if not writer.isOpened():
        pytest.skip("no writable mp4v codec available")
    for _ in range(20):
        writer.write(np.full((96, 96, 3), 96, dtype=np.uint8))
    writer.release()

    annotated = tmp_path / "gray.annotated.mp4"
    timeline = analyze_video(str(path), sample_fps=5.0, annotate_path=str(annotated))
    assert len(timeline.frames) > 0
    assert timeline.face_frames == []  # solid gray -> no faces

    # The annotated video was written and is a readable clip with frames.
    assert annotated.exists() and annotated.stat().st_size > 0
    cap = cv2.VideoCapture(str(annotated))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()

    summary = aggregate_timeline(timeline)
    assert summary.dominant_label == "none"
    assert summary.variability == 0.0

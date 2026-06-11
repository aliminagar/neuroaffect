"""Stage 3 — aggregation.

Two layers live here:

1. :func:`aggregate` — collapse per-face predictions from a *single* frame
   into a compact :class:`AffectSummary` (dominant label + mean scores).
2. The **temporal** layer — run detect -> classify across a *video*, build a
   per-frame :class:`AffectTimeline`, then summarise it over rolling time
   windows into a :class:`TemporalAffectSummary` that carries the project's
   differentiator: an **affect variability** score.

Why a rolling window (and not raw per-frame predictions)
--------------------------------------------------------
Frame-level FER predictions are *noisy*. A blink, motion blur, a momentary
head turn or partial occlusion, JPEG/compression artefacts, and the
classifier's own frame-to-frame jitter can all flip the top-1 label between
adjacent frames even when the person's underlying affect is steady. If we
treated every frame as ground truth, that flicker would masquerade as rapid
mood swings — wildly *over*-estimating lability — and the timeline would be a
jagged, uninterpretable mess.

The fix is temporal smoothing. We sample frames at a modest FPS (CPU-friendly)
and aggregate over a sliding window of a few seconds: transient
misclassifications get out-voted by their neighbours, leaving a temporally
coherent estimate. ``window_seconds`` / ``hop_seconds`` are the smoothing
knobs — a longer window is smoother but slower to react to genuine change.

Affect variability — the clinical intuition
-------------------------------------------
In mental-status terms, *affect* has a range/mobility axis:

* **Flat / blunted affect** — little change over time; the distribution of
  expressed emotion is concentrated on one state. -> **low variability.**
* **Labile affect** — rapid, pronounced shifts between emotional states.
  -> **high variability.**

We quantify this two ways, both reported:

* **Categorical** — the normalised **Shannon entropy** of the dominant-label
  distribution over time. Entropy is *the* canonical measure of how
  spread-out a distribution is: all-one-label -> 0; mass spread evenly across
  all labels -> 1. This is the headline ``variability`` score (0 = flat,
  1 = maximally labile), normalised by ``log2(7)`` so clips are comparable.
* **Dimensional** — a continuous **valence** signal (a fixed emotion->valence
  lexicon, see :data:`VALENCE`) whose **standard deviation** and frame-to-frame
  **lability** (mean absolute successive difference) capture the *magnitude*
  and *rate* of swings that a purely categorical view misses.

These are descriptive signal-processing metrics, not a diagnosis.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, pstdev

from neuroaffect.classification import AFFECT_LABELS, AffectPrediction, classify_face
from neuroaffect.detection import detect_faces


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


# ---------------------------------------------------------------------------
# Temporal layer — affect over a video
# ---------------------------------------------------------------------------

# Emotion -> valence lexicon for the continuous (dimensional) variability view.
# Based on the affective circumplex: happy is strongly positive; sad/anger
# strongly negative; fear/disgust moderately negative; surprise mildly positive
# (its valence is genuinely ambiguous, hence the small magnitude); neutral 0.
# Configurable — these weights are a modelling choice, not ground truth.
VALENCE: dict[str, float] = {
    "happy": 1.0,
    "surprise": 0.5,
    "neutral": 0.0,
    "fear": -0.5,
    "disgust": -0.5,
    "angry": -1.0,
    "sad": -1.0,
}

_MAX_ENTROPY_BITS = math.log2(len(AFFECT_LABELS))  # uniform over 7 labels


@dataclass(frozen=True)
class FrameAffect:
    """Affect estimate for one sampled video frame.

    ``scores``/``dominant`` are ``None`` when no face cleared the detector.
    """

    time: float  # seconds from clip start
    scores: dict[str, float] | None
    dominant: str | None
    face_score: float = 0.0


@dataclass
class AffectTimeline:
    """Per-frame affect estimates sampled across a video."""

    frames: list[FrameAffect]
    sample_fps: float
    duration_seconds: float

    @property
    def face_frames(self) -> list[FrameAffect]:
        return [f for f in self.frames if f.scores is not None]


@dataclass
class WindowAffect:
    """Smoothed affect over one rolling time window."""

    start: float
    end: float
    dominant: str
    mean_scores: dict[str, float]
    entropy: float  # normalised [0,1] dominant-label entropy within the window
    mean_valence: float
    valence_std: float
    num_frames: int


@dataclass
class TemporalAffectSummary:
    """Video-level affect summary — the temporal extension of AffectSummary.

    Mirrors :class:`AffectSummary` (``num_faces``, ``dominant_label``,
    ``label_counts``, ``mean_scores``) and adds the temporal/variability
    fields and the per-window timeline.
    """

    num_frames: int  # frames actually sampled
    num_faces: int  # sampled frames that had a detected face
    duration_seconds: float
    sample_fps: float
    dominant_label: str
    label_counts: dict[str, int]
    mean_scores: dict[str, float]
    # variability — the differentiator
    variability: float  # headline: normalised Shannon entropy in [0, 1]
    label_entropy_bits: float  # raw entropy (bits) before normalisation
    mean_valence: float
    valence_std: float
    valence_lability: float  # mean |Δvalence| between consecutive face frames
    window_seconds: float
    hop_seconds: float
    windows: list[WindowAffect] = field(default_factory=list)


def shannon_entropy(probs) -> float:
    """Shannon entropy (in bits) of a probability distribution."""
    # `+ 0.0` collapses the -0.0 that -sum() yields for a one-hot distribution.
    return -sum(p * math.log2(p) for p in probs if p > 0.0) + 0.0


def label_distribution(labels: list[str]) -> dict[str, float]:
    """Empirical distribution of ``labels`` over :data:`AFFECT_LABELS`."""
    counts = Counter(labels)
    n = sum(counts.values())
    if n == 0:
        return {label: 0.0 for label in AFFECT_LABELS}
    return {label: counts.get(label, 0) / n for label in AFFECT_LABELS}


def normalized_label_entropy(labels: list[str]) -> float:
    """Dominant-label entropy normalised to [0, 1] (0 = flat, 1 = uniform)."""
    if not labels:
        return 0.0
    bits = shannon_entropy(label_distribution(labels).values())
    return bits / _MAX_ENTROPY_BITS


def valence_of(scores: dict[str, float]) -> float:
    """Expected valence in [-1, 1] under the :data:`VALENCE` lexicon."""
    return sum(scores.get(label, 0.0) * weight for label, weight in VALENCE.items())


def _mean_scores(frames: list[FrameAffect]) -> dict[str, float]:
    n = len(frames)
    return {
        label: sum(f.scores[label] for f in frames) / n for label in AFFECT_LABELS
    }


def analyze_video(
    path: str,
    *,
    sample_fps: float = 4.0,
    min_score: float = 0.5,
    temperature: float = 1.0,
    max_seconds: float | None = None,
) -> AffectTimeline:
    """Run detect -> classify across a video, sampling at ``sample_fps``.

    Only the most confident face per sampled frame is classified (a
    single-subject assumption appropriate for affect tracking). Sampling well
    below the native frame rate keeps this CPU-friendly; the rolling-window
    aggregation downstream recovers a smooth signal from the sparse samples.
    """
    import cv2

    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 0 or math.isnan(native_fps):
            native_fps = 30.0  # fall back if the container doesn't report FPS
        step = max(1, round(native_fps / sample_fps))

        frames: list[FrameAffect] = []
        idx = 0
        read = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read += 1
            t = idx / native_fps
            if max_seconds is not None and t > max_seconds:
                break
            if idx % step == 0:
                boxes = detect_faces(frame, min_score=min_score)
                if boxes:
                    pred = classify_face(frame, boxes[0], temperature=temperature)
                    frames.append(
                        FrameAffect(
                            time=t,
                            scores=pred.scores,
                            dominant=pred.label,
                            face_score=boxes[0].score,
                        )
                    )
                else:
                    frames.append(FrameAffect(time=t, scores=None, dominant=None))
            idx += 1
    finally:
        cap.release()

    duration = read / native_fps if native_fps else 0.0
    return AffectTimeline(
        frames=frames, sample_fps=sample_fps, duration_seconds=duration
    )


def _window_summary(frames: list[FrameAffect], start: float, end: float) -> WindowAffect:
    labels = [f.dominant for f in frames]
    mean_scores = _mean_scores(frames)
    counts = Counter(labels)
    dominant = max(counts, key=lambda lbl: (counts[lbl], mean_scores.get(lbl, 0.0)))
    valences = [valence_of(f.scores) for f in frames]
    return WindowAffect(
        start=start,
        end=end,
        dominant=dominant,
        mean_scores=mean_scores,
        entropy=normalized_label_entropy(labels),
        mean_valence=mean(valences),
        valence_std=pstdev(valences) if len(valences) > 1 else 0.0,
        num_frames=len(frames),
    )


def _rolling_windows(
    timeline: AffectTimeline, window_seconds: float, hop_seconds: float
) -> list[WindowAffect]:
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    face_frames = timeline.face_frames
    duration = max(timeline.duration_seconds, window_seconds)
    windows: list[WindowAffect] = []
    start = 0.0
    while start < duration:
        end = start + window_seconds
        in_window = [f for f in face_frames if start <= f.time < end]
        if in_window:
            windows.append(_window_summary(in_window, start, end))
        start += hop_seconds
    return windows


def aggregate_timeline(
    timeline: AffectTimeline,
    *,
    window_seconds: float = 3.0,
    hop_seconds: float = 1.0,
) -> TemporalAffectSummary:
    """Summarise an :class:`AffectTimeline` over rolling windows.

    Computes the overall dominant affect and mean scores (same semantics as
    :func:`aggregate`), the headline variability score, the valence-based
    variability signals, and a per-window timeline for plotting.
    """
    face_frames = timeline.face_frames
    base = dict(
        num_frames=len(timeline.frames),
        duration_seconds=timeline.duration_seconds,
        sample_fps=timeline.sample_fps,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )

    if not face_frames:
        return TemporalAffectSummary(
            num_faces=0,
            dominant_label="none",
            label_counts={},
            mean_scores={label: 0.0 for label in AFFECT_LABELS},
            variability=0.0,
            label_entropy_bits=0.0,
            mean_valence=0.0,
            valence_std=0.0,
            valence_lability=0.0,
            windows=[],
            **base,
        )

    labels = [f.dominant for f in face_frames]
    counts = Counter(labels)
    mean_scores = _mean_scores(face_frames)
    dominant = max(counts, key=lambda lbl: (counts[lbl], mean_scores.get(lbl, 0.0)))

    entropy_bits = shannon_entropy(label_distribution(labels).values())
    variability = entropy_bits / _MAX_ENTROPY_BITS

    valences = [valence_of(f.scores) for f in face_frames]
    diffs = [abs(valences[i] - valences[i - 1]) for i in range(1, len(valences))]

    return TemporalAffectSummary(
        num_faces=len(face_frames),
        dominant_label=dominant,
        label_counts=dict(counts),
        mean_scores=mean_scores,
        variability=variability,
        label_entropy_bits=entropy_bits,
        mean_valence=mean(valences),
        valence_std=pstdev(valences) if len(valences) > 1 else 0.0,
        valence_lability=mean(diffs) if diffs else 0.0,
        windows=_rolling_windows(timeline, window_seconds, hop_seconds),
        **base,
    )


def plot_timeline(
    timeline: AffectTimeline,
    summary: TemporalAffectSummary,
    out_path: str,
    *,
    dpi: int = 120,
) -> str:
    """Save a PNG of affect probabilities over time + a dominant-label band."""
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    face_frames = timeline.face_frames
    if not face_frames:
        raise ValueError("no faces detected in the video; nothing to plot")

    times = [f.time for f in face_frames]
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i) for i, label in enumerate(AFFECT_LABELS)}

    fig, (band, ax) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 5.5),
        gridspec_kw={"height_ratios": [1, 6]},
    )

    # Top strip: dominant affect per window (tiled at hop resolution).
    for w in summary.windows:
        band.axvspan(
            w.start, w.start + summary.hop_seconds, color=colors.get(w.dominant, "0.6")
        )
    band.set_yticks([])
    band.set_ylabel("dominant", rotation=0, ha="right", va="center")
    band.set_title(
        f"Affect timeline — variability {summary.variability:.2f} "
        f"(0 = flat, 1 = labile) · dominant: {summary.dominant_label} · "
        f"valence σ {summary.valence_std:.2f}"
    )

    # Main panel: smoothed per-label probabilities.
    for label in AFFECT_LABELS:
        ax.plot(
            times,
            [f.scores[label] for f in face_frames],
            color=colors[label],
            linewidth=1.6,
            label=label,
        )
    ax.set_ylim(0, 1)
    ax.set_xlim(times[0], max(times[-1], summary.duration_seconds))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("affect probability")
    ax.grid(alpha=0.3)
    ax.legend(
        ncol=len(AFFECT_LABELS),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

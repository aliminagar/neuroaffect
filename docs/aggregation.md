# Stage 3 — temporal affect aggregation

Stages 1–2 answer "what affect is on this face *right now*." Stage 3 answers
the temporal question: **how does affect behave over time** — is it steady or
does it swing? That mobility-of-affect axis is this project's differentiator.

## Why not use raw per-frame predictions?

Frame-level FER output is **noisy**. Even with a stable subject, the top-1
label flickers between adjacent frames because of:

- blinks, micro-expressions, brief head turns / partial occlusion;
- motion blur and video compression artefacts;
- lighting changes (auto-exposure, shadows);
- the classifier's own decision-boundary jitter near ambiguous frames.

If we treated each frame as ground truth, that flicker would look like constant
mood swings — **massively over-estimating lability** — and any timeline plot
would be unreadable hash. You can see this directly in the demo clip: during
the "angry" segment the per-frame `angry` probability bounces between ~0.2 and
~0.7 frame to frame, even though the expression is steady.

## The fix: sample + rolling window

1. **Sample** the video at a modest `sample_fps` (default 4) instead of the
   native rate. Detection/classification only run on sampled frames, which is
   what keeps the whole thing CPU-friendly.
2. **Aggregate** over a sliding window of `window_seconds` (default 3) stepped
   by `hop_seconds` (default 1). Within each window we average the probability
   vectors, so a few misclassified frames get out-voted by their neighbours.

`window_seconds` / `hop_seconds` are the smoothing knobs: a longer window is
smoother but slower to react to genuine change; a smaller hop gives finer time
resolution at more compute.

Per sampled frame we classify **only the most confident face** (a
single-subject assumption appropriate for affect tracking); frames with no
detected face are recorded as gaps and skipped in the statistics.

## The variability metric (the differentiator)

In a mental-status exam, affect is described along a range/mobility axis:

| Clinical term | Behaviour over time | Variability |
| --- | --- | --- |
| **Flat / blunted** | little or no change; one emotional state | **low (→ 0)** |
| **Labile** | rapid, pronounced shifts between states | **high (→ 1)** |

We quantify this two complementary ways, and report both:

### 1. Categorical — normalised Shannon entropy (headline `variability`)

Take the distribution of dominant labels over time and compute its Shannon
entropy. Entropy is *the* canonical measure of how concentrated vs. spread a
distribution is:

```
H = -Σ p_i · log2(p_i)            # bits
variability = H / log2(7)         # normalise by max entropy over 7 labels -> [0, 1]
```

- All one label → `H = 0` → **variability 0** (flat).
- Mass spread evenly across all seven labels → `H = log2(7)` →
  **variability 1** (maximally labile).

Normalising by `log2(7)` makes the score comparable across clips regardless of
how many labels actually appear.

### 2. Dimensional — valence variance & lability

Entropy is *categorical*: "happy↔surprise" and "happy↔sad" both count as one
switch, even though the second is a far bigger emotional swing. To capture
*magnitude* and *rate*, we project each frame onto a continuous **valence**
scalar via a fixed lexicon (`VALENCE` in `aggregation.py`; based on the
affective circumplex, and configurable):

```
happy +1.0 · surprise +0.5 · neutral 0.0 · fear/disgust −0.5 · angry/sad −1.0
valence(frame) = Σ p_label · weight_label          # in [-1, 1]
```

From the per-frame valence signal we report:

- **`valence_std`** — overall spread of the valence signal (amplitude of swings);
- **`valence_lability`** — mean absolute change between consecutive frames
  (`mean |v_t − v_{t−1}|`), i.e. the *rate* of swinging.

> These are descriptive signal-processing metrics on a model's outputs, **not a
> clinical diagnosis.** They're meant to make "flat vs. labile" measurable and
> visualisable, nothing more.

## Output

`aggregate_timeline()` returns a `TemporalAffectSummary` — the temporal
extension of `AffectSummary`. It mirrors the single-frame fields
(`num_faces`, `dominant_label`, `label_counts`, `mean_scores`) and adds
`variability`, `label_entropy_bits`, `mean_valence`, `valence_std`,
`valence_lability`, and the per-window timeline (`windows`).

`plot_timeline()` saves a PNG: the smoothed per-label probabilities over time,
with a dominant-affect colour band along the top.

## See it run end-to-end

```powershell
# 1. Make a sample clip. Default "labile" mode cross-fades distinct public-domain
#    faces (happy -> neutral -> sad -> angry -> happy) with per-frame jitter.
#    Falls back to a flat single-portrait animation if offline.
python scripts/make_sample_video.py

# 2. Analyze it -> timeline PNG + JSON summary (with the variability score).
python -m neuroaffect.cli analyze data/sample_clip.mp4 --fps 5 --window 2 --hop 0.5
```

On the labile demo clip this reports **variability ≈ 0.65** with the dominant
band walking happy → neutral → sad → angry → happy and the valence signal
swinging from +0.9 to −0.9. Running the `--mode flat` clip instead reports
**variability ≈ 0** — the flat-affect end of the same scale.

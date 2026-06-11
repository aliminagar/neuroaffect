# neuroaffect

**A CPU-friendly facial-affect analysis pipeline that turns face video into a
temporal emotion signal — and measures how *stable vs. labile* that emotion is
over time.** It detects faces, classifies affect over seven emotions,
aggregates per-frame predictions into a smoothed timeline with a principled
**affect-variability** score, and benchmarks itself honestly against a real
labeled test set. Built end-to-end with off-the-shelf pretrained models, no GPU
required.

⚠️ Not a medical or diagnostic device. This is research / portfolio
software. Facial-expression recognition is noisy and culturally biased, the
bundled model has a known data-contamination issue (see
Results), and "affect variability" here is
a descriptive signal-processing metric — not a clinical assessment. Do not
use it to make decisions about real people. See Limitations
and Disclaimer.

📍 Repository: [github.com/aliminagar/neuroaffect](https://github.com/aliminagar/neuroaffect/)

✨ Highlights

A temporal metric, not just a classifier. Beyond "what emotion is on this
face," it quantifies how affect moves over time — steady (flat) vs. swinging
(labile) — borrowing a concept from the clinical mental-status exam and turning
it into a measurable signal.
Honest evaluation that caught real data leakage. The benchmark harness
surfaced train/test contamination in the pretrained model (an implausible
86% on FER-2013 vs. ~73% state-of-the-art) and reports it transparently
instead of quoting the inflated number — a deliberate demonstration of
evaluation rigor.
Production-shaped engineering. Four stages behind small typed interfaces
(swap any model without touching the others), 53 tests split into fast offline
units + gated integration tests, auto-downloading/caching models, and a clean
staged commit history.
---

## Demo

<p align="center">
  <img src="docs/assets/demo.gif" alt="neuroaffect annotated demo — auto-plays" width="360"><br/>
  <sub>Annotated output: face box, predicted affect + confidence, and a running smoothed affect/valence readout.</sub>
</p>

One pass over a short clip produces that **annotated video** plus a **timeline
plot** with a per-window dominant-affect band. Representative frames across the
clip (the box label is the *per-frame* prediction; the panel's affect/valence
readout is *smoothed* over the rolling window — note the angry frame still reads
SAD while the window catches up):

| happy | sad | angry |
| --- | --- | --- |
| ![happy frame](docs/assets/frame_happy.jpg) | ![sad frame](docs/assets/frame_sad.jpg) | ![angry frame](docs/assets/frame_angry.jpg) |

Affect timeline for the whole clip:

![affect timeline](docs/assets/timeline.png)

This is a deliberately *labile* demo clip (it cross-fades distinct public-domain
faces: happy → neutral → sad → angry → happy). Variability ≈ **0.69** on a 0–1
scale. Note the noisy "angry" stretch where the raw per-frame `angry`
probability bounces between ~0.2 and ~0.7 — exactly the frame-level noise the
rolling window is there to smooth.

```powershell
# Reproduce the demo (no input video needed — it synthesizes one):
python scripts/make_sample_video.py
python -m neuroaffect.cli analyze data/sample_clip.mp4 \
    --fps 10 --window 2 --hop 0.5 \
    --annotate data/sample_clip.annotated.mp4
```

### On a single image

Detect + classify a still with `scripts/draw_faces.py` (green box + affect label):

<p align="center">
  <img src="docs/assets/detection_demo.jpg" alt="detection + classification on a still image" width="320">
</p>

---

## Architecture

![Pipeline architecture](docs/assets/architecture.png)

<sub>Rendered diagram (also available as [SVG](docs/assets/architecture.svg) for slides/PDF). Mermaid source below.</sub>

<details>
<summary>Mermaid source</summary>

```mermaid
flowchart LR
    V[image / video frame] --> D

    subgraph Pipeline
        D["1. detection<br/>MediaPipe BlazeFace<br/>faces to boxes"]
        C["2. classification<br/>ViT FER (HuggingFace)<br/>7-way affect probs"]
        A["3. aggregation<br/>rolling-window timeline<br/>dominant + variability"]
        D --> C --> A
    end

    A --> S["JSON summary + timeline PNG + annotated video"]

    GT["labeled FER-2013 test set"] --> E
    C -. benchmarked by .-> E["4. evaluation<br/>accuracy / macro-F1 / confusion"]
    E --> M["metrics JSON + confusion-matrix PNG"]
```

</details>

Each stage hides behind a small typed interface, so any model can be swapped
without touching the others. Every model and dataset **auto-downloads and
caches** on first use.

## The four stages

| Stage | Module | What it does | Model / data | Why |
| --- | --- | --- | --- | --- |
| 1. Detection | [`detection.py`](src/neuroaffect/detection.py) | faces → bounding boxes | MediaPipe BlazeFace | real-time on CPU, ~10 MB, no Torch/TF · [rationale](docs/detection.md) |
| 2. Classification | [`classification.py`](src/neuroaffect/classification.py) | face → 7-way affect probs | `dima806/...` ViT (Apache-2.0) | exact label match, permissive license · [rationale](docs/classification.md) |
| 3. Aggregation | [`aggregation.py`](src/neuroaffect/aggregation.py) | frames → smoothed timeline + **variability** | — | the differentiator · [method](docs/aggregation.md) |
| 4. Evaluation | [`evaluate.py`](src/neuroaffect/evaluate.py) | scores Stage 2 on real labels | FER-2013 test split | honest metrics · [method + caveat](docs/evaluation.md) |

### The differentiator: affect variability

Stages 1–2 answer "what affect is on this face right now." Stage 3 answers the
temporal question — **is the affect steady or swinging?** — borrowing the
clinical *flat vs. labile* distinction:

- **Flat / blunted affect** → one state, little change → **variability ≈ 0**
- **Labile affect** → rapid, large shifts → **variability ≈ 1**

It's quantified two complementary ways: the **normalized Shannon entropy** of
the dominant-label distribution (categorical spread), and the **standard
deviation + lability** of a continuous valence signal (magnitude and rate of
swings). Raw per-frame predictions are too noisy to use directly, so a rolling
time window smooths them first. Full method and the clinical intuition:
[docs/aggregation.md](docs/aggregation.md).

## Results: honest FER-2013 numbers

Stage 2 evaluated on a 1,000-sample representative subsample of the **FER-2013
test split** (`python -m neuroaffect.cli evaluate --dataset fer2013 --limit 1000`):

| Metric | Value |
| --- | --- |
| Accuracy | **0.862** |
| Macro-F1 | **0.876** |
| Samples | 1,000 of 7,178 |

![confusion matrix](docs/assets/confusion_matrix.png)

### 🚩 Why you should distrust that 86% — and why that's the point

Published FER-2013 accuracy tops out around **73–76%** (state of the art), with
human accuracy ~65%. **An 86% score is a red flag, not a triumph.** The most
likely cause is **train/test contamination**: the pretrained model was almost
certainly trained on FER-2013, so it has *seen these test images*. The
fingerprint is **`disgust` scoring a perfect 1.00 F1** — disgust is FER-2013's
rarest (~1.5%) and hardest class; nobody gets it perfect without having
memorized it.

So the headline number measures **memorization, not generalization**, and is
reported here as an upper bound under probable leakage. Surfacing this — rather
than quoting the inflated figure as a win — is the intended demonstration of
evaluation rigor. A trustworthy estimate needs a test set the model demonstrably
never saw. Details: [docs/evaluation.md](docs/evaluation.md).

## Quickstart

Requires Python **3.11+** (developed on 3.12). First run downloads the models
(~hundreds of MB: a ViT classifier via PyTorch + the BlazeFace detector).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
# source .venv/bin/activate            # macOS / Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                       # installs the `neuroaffect` console script
```

```powershell
# Image → affect summary
python -m neuroaffect.cli analyze data/portrait.jpg

# Video → timeline PNG + JSON summary (with the variability score),
#         and optionally an annotated MP4
python -m neuroaffect.cli analyze data/sample_clip.mp4 \
    --fps 10 --window 2 --hop 0.5 --annotate data/sample_clip.annotated.mp4

# Benchmark the classifier on FER-2013 (auto-downloads; --limit for CPU speed)
python -m neuroaffect.cli evaluate --dataset fer2013 --limit 1000

# Score your own predictions vs. ground truth (two JSON label lists)
python -m neuroaffect.cli evaluate preds.json truth.json
```

Helper scripts to see it working without supplying your own media:

```powershell
python scripts/draw_faces.py            # detect + classify a still → boxes + labels
python scripts/make_sample_video.py     # synthesize a labile sample clip
```

## Tests

```powershell
pytest                          # full suite (integration tests download models)
pytest -m "not integration"     # fast, offline: pure logic only
```

The suite separates **pure-logic unit tests** (entropy, valence, metrics,
box-clamping, label mapping — no models or network) from **integration tests**
that exercise the real models and datasets, so most of it runs offline in
seconds.

## Limitations

- **Model data contamination.** The Stage 2 model was likely trained on
  FER-2013, so the benchmark over-states real-world accuracy (see above).
- **FER is inherently noisy and biased.** Facial-expression → emotion mapping is
  contested; models trial-trained on posed Western datasets generalize poorly
  across cultures, lighting, occlusion, and demographics.
- **"Affect variability" is descriptive, not clinical.** The valence lexicon is
  a fixed modeling choice; the metric quantifies output variation, not a
  person's actual emotional state.
- **Frontal, single-subject bias.** BlazeFace is tuned for near, frontal faces;
  only the most-confident face per frame is tracked. Profiles, small/occluded
  faces, and crowds are out of scope.
- **Probabilities aren't truly calibrated.** Softmax outputs are reported as-is;
  a temperature hook exists but no calibration set is fitted.
- **Synthetic demo clip.** The sample video cross-fades stills, not natural
  motion — convenient and reproducible, but not representative of real video.

## Disclaimer

**neuroaffect is research and portfolio software. It is NOT a medical device and
NOT a diagnostic, screening, or clinical tool.** Facial-expression recognition
cannot reliably infer a person's internal emotional or mental state, and the
clinical terms used here ("flat affect", "labile") are borrowed only as
*intuition* for a signal-processing metric. Do not use this software to make
judgments or decisions about any individual. No warranty of any kind.

## Layout

```
src/neuroaffect/   pipeline stages (importable package)
scripts/           runnable demos (draw_faces, make_sample_video)
tests/             unit + integration tests mirroring src/
docs/              per-stage rationale + assets for this README
data/              inputs/outputs/datasets/model cache (gitignored)
```

## License

Code: **MIT** (portfolio use). Third-party components keep their own licenses —
the Stage 2 model is Apache-2.0, FER-2013 is used under its public research
release terms, and the demo faces are public domain. Model weights and datasets
are downloaded at runtime, not redistributed here.

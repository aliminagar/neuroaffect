# Stage 4 — honest evaluation

This stage runs the **Stage 2 classifier** against a real, labelled FER test
set and reports whatever it actually scores. No numbers here are hand-picked or
synthetic.

## Dataset choice: FER-2013 (not RAF-DB)

| | FER-2013 | RAF-DB |
| --- | --- | --- |
| Access | Public Kaggle competition release; widely mirrored | Free for **academic** use only; requires signing a licence + registering |
| Auto-download | Yes — no login | No — manual request/approval, not redistributable |
| Classes | 7 (matches our labels exactly) | 7 basic (+ compound variants) |

For an auto-downloading portfolio project, **FER-2013 is the right pick**:
RAF-DB's licence forbids unattended download/redistribution, so a reproducible
`pip install`-and-run eval isn't possible with it. We therefore recommend and
implement FER-2013.

### Source & caching

We pull the WebDataset mirror
[`clip-benchmark/wds_fer2013`](https://huggingface.co/datasets/clip-benchmark/wds_fer2013)
from the Hugging Face Hub via `huggingface_hub` (already a dependency through
`transformers`). The **test split** is 7,178 48×48 grayscale aligned face
crops. Files cache in the standard HF cache, so re-runs are offline. The
mirror's class names (`disgusted`, `fearful`, `surprised`) normalise to our
canonical labels via the aliases in `classification.py`.

## Protocol

FER-2013 images are **pre-cropped, aligned faces**, so evaluation classifies
each image directly rather than running the MediaPipe detector first (which is
designed for full scenes, not 48×48 thumbnails). This measures **Stage 2 in
isolation** — the standard FER benchmark protocol.

Metrics reported (in the JSON and on the confusion-matrix PNG):

- overall **accuracy**
- **macro-F1** (unweighted mean of per-class F1 — the fair metric on
  FER-2013's very imbalanced classes, where `disgust` is ~1.5% of the data)
- per-class **precision / recall / F1** and **support**
- a **confusion matrix** (row-normalised) saved as PNG

## CPU feasibility (`--limit`)

A full 7,178-image pass is slow on CPU (a ViT forward pass per image). Use
`--limit N` to evaluate on a **representative even-strided subsample** (kept
deterministic and spread across the whole split so the class mix stays
representative). The reported `samples_used` / `total_available` make the
sample size explicit — we never silently truncate.

```powershell
# Representative 1,000-sample run (writes JSON + confusion PNG under data/)
python -m neuroaffect.cli evaluate --dataset fer2013 --limit 1000

# Full split (slow):
python -m neuroaffect.cli evaluate --dataset fer2013
```

## Results

Measured on a **1,000-sample** representative subsample of the FER-2013 test
split (`--limit 1000`), classifier `dima806/facial_emotions_image_detection`:

| Metric | Value |
| --- | --- |
| **Accuracy** | **0.862** |
| **Macro-F1** | **0.877** |
| Samples used | 1,000 of 7,178 |

Per-class F1: disgust 1.00 · happy 0.93 · surprise 0.91 · angry 0.86 ·
neutral 0.85 · fear 0.79 · sad 0.79. The confusion matrix
([`data/fer2013_confusion.png`](../data/fer2013_confusion.png)) shows a strong
diagonal with the usual FER confusions (fear↔sad, neutral→sad).

(Reproduce with `python -m neuroaffect.cli evaluate --dataset fer2013 --limit 1000`.
Exact figures shift slightly with `--limit`; the full split will differ a little.)

### ⚠️ Read the number critically

Published FER-2013 test accuracy tops out around **~73–76%** (state of the art),
with human accuracy estimated at **~65 ± 5%**. Our **86.2%** sits well above
that ceiling — **that's a red flag, not a triumph.** The most likely
explanation is **train/test contamination**: `dima806/facial_emotions_image_detection`
appears to have been trained on FER-2013, so these test images were probably
seen during training. The give-away is **disgust = 1.00 precision/recall/F1** —
disgust is FER-2013's rarest (~1.5%) and notoriously *hardest* class; a perfect
score on it is the signature of memorisation, not skill. When a model is scored
on data it trained on, the result measures *memorisation*, not *generalisation*.

So treat the headline accuracy as an **upper bound under probable leakage**, not
an estimate of real-world performance. The honest takeaways:

- The harness is correct and the numbers are real (reproducible from the CLI).
- A trustworthy generalisation estimate needs a test set the model demonstrably
  never saw (e.g. AffectNet/RAF-DB under licence, or a freshly collected set) —
  that's the natural next step.

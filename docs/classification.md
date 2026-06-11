# Stage 2 — affect classifier choice

`classify_faces(frame, boxes) -> list[AffectPrediction]` needs a model that
turns a face crop into a distribution over the seven pipeline labels
(`neutral, happy, sad, angry, surprise, fear, disgust`). The brief: **use a
pretrained FER model from Hugging Face** (not train from scratch), keep it
**CPU-friendly, reproducible, and permissively licensed**, with a clean
**upgrade path to fine-tuning** later.

## Why pretrained-from-Hub (not from scratch)

Training a competitive FER model needs AffectNet/FER-2013 data wrangling, GPU
time, and careful regularisation — none of which a reviewer can reproduce in a
`pip install`. A Hub model gives real inference *today*, and because we load it
through `transformers` `Auto*` classes, swapping in a locally fine-tuned
checkpoint later is a one-line change (`NEUROAFFECT_FER_MODEL=./my-checkpoint`).

## Candidates considered

| Model | Arch | Labels | Reported acc. | License | Notes |
| --- | --- | --- | --- | --- | --- |
| **`dima806/facial_emotions_image_detection`** | ViT-base | **exactly our 7** | **~0.91** (own test set) | **Apache-2.0** | Clean `id2label`, widely used, permissive |
| `trpakov/vit-face-expression` | ViT-base | 7 (FER-2013) | ~0.71 (FER-2013) | unspecified | Popular, but license unclear for a public portfolio |
| `motheecreator/vit-Facial-Expression-Recognition` | ViT-base | 7 | ~0.91 | Apache-2.0 | Also strong; comparable alternative |
| ResNet-based FER checkpoints | ResNet | varies | varies | varies | Lighter, but Hub ones are older/less maintained; accuracy generally below ViT |

FER accuracy numbers are **not directly comparable** across rows — each is on
that author's own split, and FER-2013's ceiling is famously low (~65-75% even
for strong models) because of label noise, while AffectNet-style sets report
higher. Treat them as ballpark, not leaderboard.

## Recommendation: `dima806/facial_emotions_image_detection`

- **Labels line up exactly** with the pipeline's seven — no remapping guesswork
  (we still normalise spellings defensively in `classification.py`).
- **Apache-2.0** — safe to show publicly.
- **~91%** reported accuracy on its held-out set; strong per-class precision.
- **ViT-base on CPU** is fine for single-image / frame-by-frame portfolio use
  (one forward pass per face). If you later need many faces per second on CPU,
  a distilled/quantised model is the optimisation path.

Verified locally: on the sample portrait it predicts **happy ≈ 0.88**, a real,
non-uniform, correct result.

## Implementation notes

- Loaded via `AutoImageProcessor` + `AutoModelForImageClassification`; the
  processor handles **resize → 224×224 and normalisation**, so the only manual
  step is **BGR→RGB** (OpenCV → PIL).
- Weights (~330 MB) **auto-download once** and cache in the standard Hugging
  Face cache (`~/.cache/huggingface`, or `$HF_HOME`). Override the model with
  `NEUROAFFECT_FER_MODEL`.
- Logits → probabilities via a **temperature-scaled softmax**
  (`temperature=1.0` default). Probabilities are mapped onto the canonical
  labels; any out-of-set class (e.g. "contempt") is dropped and the rest
  renormalised so the distribution always sums to 1 over our seven.
- **Calibration upgrade path:** fit a single temperature on a validation set
  (Guo et al., 2017) and pass it as `temperature=` — no code change needed.

See it work:

```powershell
python scripts/draw_faces.py            # detect + classify the sample portrait
python scripts/draw_faces.py my.jpg --show
```

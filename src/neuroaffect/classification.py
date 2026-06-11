"""Stage 2 — affect classification.

Given a detected face crop, predict its affect (emotion) label and a
probability distribution over the supported labels.

Model choice
------------
This stage runs a **pretrained Vision Transformer**,
``dima806/facial_emotions_image_detection`` from the Hugging Face Hub, rather
than training from scratch. It was picked for a CPU-friendly, reproducible
portfolio project; see ``docs/classification.md`` for the full comparison.
In short:

* **Architecture** — ViT-base (fine-tuned from ``google/vit-base-patch16-224``).
* **Labels** — exactly the seven this pipeline uses (angry, disgust, fear,
  happy, neutral, sad, surprise), so no label surgery is needed.
* **Accuracy** — ~91% reported on its held-out test set.
* **License** — Apache-2.0 (safe for a public portfolio).
* **CPU** — a single ViT-base forward pass per face is comfortable on CPU.

The weights (~330 MB) are downloaded once and cached by the Hugging Face Hub
(under ``~/.cache/huggingface`` unless ``HF_HOME`` says otherwise). Override
the model with the ``NEUROAFFECT_FER_MODEL`` environment variable — point it at
another Hub id or a local fine-tuned checkpoint when you upgrade.

Calibration
-----------
The model's logits are turned into probabilities with a temperature-scaled
softmax. ``temperature == 1.0`` is the raw softmax; fitting a single
temperature on a validation set (temperature scaling, Guo et al. 2017) is the
clean calibration upgrade path — pass the fitted value as ``temperature``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from neuroaffect.detection import BoundingBox, Frame

# Canonical affect label set (Ekman's basic emotions + neutral).
AFFECT_LABELS: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprise",
    "fear",
    "disgust",
)

# Default Hugging Face model; override with NEUROAFFECT_FER_MODEL.
_DEFAULT_FER_MODEL = "dima806/facial_emotions_image_detection"

# Map the various label spellings models use onto our canonical names.
_LABEL_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "afraid": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}


@dataclass(frozen=True)
class AffectPrediction:
    """Predicted affect for a single face."""

    box: BoundingBox
    label: str
    scores: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.scores.get(self.label, 0.0)


def _model_id() -> str:
    return os.environ.get("NEUROAFFECT_FER_MODEL", _DEFAULT_FER_MODEL)


@lru_cache(maxsize=2)
def _model(model_id: str):
    """Load (and memoise) the image processor + model for ``model_id``."""
    try:
        import torch  # noqa: F401  (imported for side-effect/availability check)
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ImportError(
            "Affect classification requires 'transformers' and 'torch'. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    return processor, model


def _normalize_label(raw: str) -> str | None:
    """Map a model's raw label string onto one of :data:`AFFECT_LABELS`."""
    return _LABEL_ALIASES.get(raw.strip().lower())


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically-stable temperature-scaled softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    x = np.asarray(logits, dtype=np.float64) / temperature
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _scores_from_logits(
    logits: np.ndarray,
    id2label: dict[int, str],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Convert model logits into a calibrated distribution over our labels.

    Probabilities for any model label outside :data:`AFFECT_LABELS` (e.g.
    "contempt") are dropped and the remainder renormalised, so the result
    always sums to 1 over exactly the seven canonical labels.
    """
    probs = _softmax(logits, temperature)
    scores = {label: 0.0 for label in AFFECT_LABELS}
    for idx, prob in enumerate(probs):
        name = _normalize_label(id2label[idx])
        if name in scores:
            scores[name] += float(prob)

    total = sum(scores.values())
    if total > 0:
        scores = {label: value / total for label, value in scores.items()}
    return scores


def classify_face(
    frame: Frame, box: BoundingBox, *, temperature: float = 1.0
) -> AffectPrediction:
    """Classify the affect of the face inside ``box``.

    The crop is converted BGR->RGB and resized/normalised by the model's own
    image processor (224x224 for the default ViT). Returns a calibrated
    distribution over :data:`AFFECT_LABELS` and the arg-max label.
    """
    face = box.crop(frame)
    if face.size == 0:
        raise ValueError("bounding box does not overlap the frame")

    import torch
    from PIL import Image

    processor, model = _model(_model_id())

    rgb = np.ascontiguousarray(face[..., ::-1])  # BGR -> RGB
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()

    scores = _scores_from_logits(logits, model.config.id2label, temperature)
    label = max(scores, key=scores.get)
    return AffectPrediction(box=box, label=label, scores=scores)


def classify_faces(
    frame: Frame, boxes: list[BoundingBox], *, temperature: float = 1.0
) -> list[AffectPrediction]:
    """Classify every detected face in a frame."""
    return [classify_face(frame, box, temperature=temperature) for box in boxes]

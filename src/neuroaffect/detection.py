"""Stage 1 — face detection.

Locate faces in a single image/frame and return their bounding boxes.
The public surface is intentionally small (:func:`detect_faces` +
:class:`BoundingBox`) so the detector can be swapped without touching
downstream stages.

Detector choice
---------------
This stage uses **MediaPipe Face Detection** (the BlazeFace short-range
model, served through MediaPipe's Tasks API). It was picked over MTCNN and
YOLO-face for a CPU-friendly portfolio project; see ``docs/detection.md``
for the full comparison. In short:

* **MediaPipe** — real-time on CPU, ~10 MB wheel, no PyTorch/TensorFlow,
  one tiny (~230 KB) model fetched on first run. Frontal-leaning: weaker on
  extreme profile / heavy occlusion, but the best speed/footprint trade-off.
* **MTCNN** — strong on tiny faces via its image pyramid, but slow on CPU
  and the common pip package drags in TensorFlow (hundreds of MB).
* **YOLO-face** — best accuracy on small/profile/occluded faces, but
  ``ultralytics`` pulls in Torch (~1 GB+) and weights — overkill on CPU.

The MediaPipe model is downloaded once and cached. Override the location
with the ``NEUROAFFECT_MODEL_PATH`` (full file path) or
``NEUROAFFECT_MODEL_DIR`` (directory) environment variables — useful for
offline/air-gapped runs where you pre-place the ``.tflite`` file.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

import numpy as np

# A frame is an (H, W, 3) uint8 BGR image (OpenCV convention).
Frame = np.ndarray

# BlazeFace short-range detector (float16), published by Google for the
# MediaPipe Tasks API. ~230 KB; covers faces within ~2 m of the camera.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_FILENAME = "blaze_face_short_range.tflite"

# Detector-internal confidence floor. Kept low so callers can choose any
# ``min_score`` at call time; the final filtering happens in Python.
_DETECTOR_CONFIDENCE_FLOOR = 0.1


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned face box in pixel coordinates, with a detector score."""

    x: int
    y: int
    width: int
    height: int
    score: float = 1.0

    def crop(self, frame: Frame) -> Frame:
        """Return the sub-image covered by this box."""
        return frame[self.y : self.y + self.height, self.x : self.x + self.width]


def _model_path() -> Path:
    """Resolve where the detector model lives, honouring env overrides."""
    override = os.environ.get("NEUROAFFECT_MODEL_PATH")
    if override:
        return Path(override)
    base = os.environ.get("NEUROAFFECT_MODEL_DIR")
    cache_dir = Path(base) if base else Path.home() / ".cache" / "neuroaffect"
    return cache_dir / _MODEL_FILENAME


def _ensure_model() -> Path:
    """Return the model path, downloading it on first use.

    The download is written to a temp file in the destination directory and
    atomically renamed, so a crashed/partial fetch never leaves a corrupt
    model behind.
    """
    path = _model_path()
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(_MODEL_URL) as response:  # noqa: S310 (trusted Google CDN URL)
        data = response.read()

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return path


@lru_cache(maxsize=4)
def _detector(model_path: str):
    """Build (and memoise) a MediaPipe FaceDetector for ``model_path``."""
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "Face detection requires 'mediapipe'. Install it with "
            "`pip install -r requirements.txt` (or `pip install mediapipe`)."
        ) from exc

    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        min_detection_confidence=_DETECTOR_CONFIDENCE_FLOOR,
    )
    return vision.FaceDetector.create_from_options(options)


def _clamp_box(
    x: float,
    y: float,
    width: float,
    height: float,
    score: float,
    frame_w: int,
    frame_h: int,
) -> BoundingBox | None:
    """Clamp a raw detection to the frame; drop it if nothing remains.

    Detectors can return boxes that spill past the image edge (or, for very
    low-confidence noise, collapse to zero area). This rounds to integer
    pixels, clips to ``[0, frame_*]``, and returns ``None`` for any box with
    no positive area inside the frame.
    """
    x0 = max(0, min(int(round(x)), frame_w))
    y0 = max(0, min(int(round(y)), frame_h))
    x1 = max(0, min(int(round(x + width)), frame_w))
    y1 = max(0, min(int(round(y + height)), frame_h))
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    return BoundingBox(x=x0, y=y0, width=w, height=h, score=float(score))


def detect_faces(frame: Frame, *, min_score: float = 0.5) -> list[BoundingBox]:
    """Detect faces in ``frame``.

    Parameters
    ----------
    frame:
        An (H, W, 3) BGR image (OpenCV convention).
    min_score:
        Discard detections whose confidence is below this threshold.

    Returns
    -------
    A list of :class:`BoundingBox`, one per detected face, ordered by
    descending confidence. Empty if no face clears ``min_score``.

    Notes
    -----
    On the first call the BlazeFace model (~230 KB) is downloaded and cached
    (see module docstring for override env vars). MediaPipe expects RGB input,
    so the BGR frame is converted internally.
    """
    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an (H, W, 3) BGR image array")

    # Local import keeps module import cheap and lets the package load even
    # when mediapipe is absent (e.g. running only the evaluate command).
    import mediapipe as mp

    detector = _detector(str(_ensure_model()))

    frame_h, frame_w = frame.shape[:2]
    rgb = np.ascontiguousarray(frame[..., ::-1], dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    boxes: list[BoundingBox] = []
    for detection in result.detections:
        score = detection.categories[0].score if detection.categories else 0.0
        if score < min_score:
            continue
        bb = detection.bounding_box
        box = _clamp_box(
            bb.origin_x, bb.origin_y, bb.width, bb.height, score, frame_w, frame_h
        )
        if box is not None:
            boxes.append(box)

    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes

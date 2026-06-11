"""Stage 1 — face detection.

Locate faces in a single image/frame and return their bounding boxes.
The public surface is intentionally small so detectors can be swapped
without touching downstream stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A frame is an (H, W, 3) uint8 BGR image (OpenCV convention).
Frame = np.ndarray


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


def detect_faces(frame: Frame, *, min_score: float = 0.5) -> list[BoundingBox]:
    """Detect faces in ``frame``.

    Parameters
    ----------
    frame:
        An (H, W, 3) BGR image.
    min_score:
        Discard detections below this confidence.

    Returns
    -------
    A list of :class:`BoundingBox`, one per detected face.

    Notes
    -----
    Stub implementation. Wire in a real detector here (e.g. OpenCV's
    Haar/DNN face detector, MediaPipe, or RetinaFace) and return its boxes.
    """
    if frame is None or getattr(frame, "ndim", 0) != 3:
        raise ValueError("frame must be an (H, W, 3) image array")

    # TODO: replace with a real detector. Returns no faces for now.
    _ = min_score
    return []

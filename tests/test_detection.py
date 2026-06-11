import numpy as np
import pytest

from neuroaffect.detection import BoundingBox, detect_faces


def test_detect_faces_returns_list_on_valid_frame():
    frame = np.zeros((48, 48, 3), dtype=np.uint8)
    boxes = detect_faces(frame)
    assert isinstance(boxes, list)


def test_detect_faces_rejects_non_image():
    with pytest.raises(ValueError):
        detect_faces(np.zeros((48, 48), dtype=np.uint8))  # 2D, not (H, W, 3)


def test_bounding_box_crop():
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    box = BoundingBox(x=2, y=1, width=3, height=4)
    crop = box.crop(frame)
    assert crop.shape == (4, 3, 3)

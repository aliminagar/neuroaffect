import numpy as np
import pytest

from neuroaffect.detection import BoundingBox, _clamp_box, detect_faces


def test_detect_faces_rejects_non_image():
    # 2D array is not (H, W, 3); rejected before any model is touched.
    with pytest.raises(ValueError):
        detect_faces(np.zeros((48, 48), dtype=np.uint8))


def test_detect_faces_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        detect_faces(np.zeros((48, 48, 4), dtype=np.uint8))


def test_bounding_box_crop():
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    box = BoundingBox(x=2, y=1, width=3, height=4)
    crop = box.crop(frame)
    assert crop.shape == (4, 3, 3)


# --- _clamp_box: pure, no mediapipe / network needed -----------------------


def test_clamp_box_inside_frame_is_unchanged():
    box = _clamp_box(10, 20, 30, 40, 0.9, frame_w=100, frame_h=100)
    assert box == BoundingBox(x=10, y=20, width=30, height=40, score=0.9)


def test_clamp_box_clips_overflow_to_frame_edges():
    box = _clamp_box(90, 90, 50, 50, 0.8, frame_w=100, frame_h=100)
    assert box == BoundingBox(x=90, y=90, width=10, height=10, score=0.8)


def test_clamp_box_clips_negative_origin():
    box = _clamp_box(-10, -5, 30, 30, 0.7, frame_w=100, frame_h=100)
    assert box == BoundingBox(x=0, y=0, width=20, height=25, score=0.7)


def test_clamp_box_rounds_to_integer_pixels():
    box = _clamp_box(10.4, 10.6, 5.5, 5.5, 0.6, frame_w=100, frame_h=100)
    assert (box.x, box.y, box.width, box.height) == (10, 11, 6, 5)


def test_clamp_box_drops_fully_out_of_frame():
    assert _clamp_box(200, 200, 10, 10, 0.9, frame_w=100, frame_h=100) is None


def test_clamp_box_drops_zero_area():
    assert _clamp_box(10, 10, 0, 20, 0.9, frame_w=100, frame_h=100) is None


# --- end-to-end against the real detector (needs mediapipe + first-run net) -


@pytest.mark.integration
def test_detect_faces_on_blank_frame_returns_empty_list():
    pytest.importorskip("mediapipe")
    boxes = detect_faces(np.zeros((64, 64, 3), dtype=np.uint8))
    assert boxes == []


@pytest.mark.integration
def test_detect_faces_finds_a_face_in_sample_portrait():
    pytest.importorskip("mediapipe")
    from pathlib import Path

    import cv2

    sample = Path(__file__).resolve().parent.parent / "data" / "portrait.jpg"
    if not sample.exists():
        pytest.skip("sample portrait not present; run scripts/draw_faces.py once")

    frame = cv2.imread(str(sample))
    assert frame is not None

    boxes = detect_faces(frame, min_score=0.5)
    assert len(boxes) >= 1

    h, w = frame.shape[:2]
    for box in boxes:
        assert 0 <= box.x < box.x + box.width <= w
        assert 0 <= box.y < box.y + box.height <= h
        assert box.score >= 0.5
    # Boxes come back ordered by descending confidence.
    assert boxes == sorted(boxes, key=lambda b: b.score, reverse=True)

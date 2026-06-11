"""Detect faces and draw each one's predicted emotion on an image.

Runs the full detect -> classify pass and annotates every face with its
top affect label and confidence, so you can eyeball both stages at once.

Usage
-----
    # Use the bundled sample portrait (downloaded on first run):
    python scripts/draw_faces.py

    # Or point it at your own image:
    python scripts/draw_faces.py path/to/photo.jpg

    # Choose where the annotated copy is written and the score threshold:
    python scripts/draw_faces.py photo.jpg --out out.jpg --min-score 0.6

    # Draw boxes only, skipping the (heavier) emotion classifier:
    python scripts/draw_faces.py --no-classify

    # Pop up a window instead of (or as well as) writing a file:
    python scripts/draw_faces.py --show

The script prints one line per detected face and saves an annotated copy
next to the input (``<name>.boxes.jpg``) unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

import cv2

# Make the script runnable straight from a checkout (without `pip install -e .`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuroaffect.classification import classify_faces  # noqa: E402
from neuroaffect.detection import detect_faces  # noqa: E402

# A CC-licensed portrait Google ships with the MediaPipe samples; handy as a
# zero-setup default so `python scripts/draw_faces.py` just works.
_SAMPLE_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"
_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "portrait.jpg"

_BOX_COLOR = (0, 200, 0)  # BGR — green
_TEXT_COLOR = (0, 0, 0)


def _ensure_sample() -> Path:
    """Download the bundled sample portrait on first use, then reuse it."""
    if _SAMPLE_PATH.exists():
        return _SAMPLE_PATH
    print(f"Fetching sample image -> {_SAMPLE_PATH}")
    _SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(_SAMPLE_URL) as response:  # noqa: S310 (trusted CDN)
        _SAMPLE_PATH.write_bytes(response.read())
    return _SAMPLE_PATH


def draw_annotations(image, annotations: list[tuple]):
    """Draw each (box, caption) onto a copy of ``image``."""
    annotated = image.copy()
    for box, caption in annotations:
        cv2.rectangle(
            annotated,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            _BOX_COLOR,
            2,
        )
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(box.y, th + 4)
        cv2.rectangle(
            annotated, (box.x, ty - th - 4), (box.x + tw + 4, ty), _BOX_COLOR, -1
        )
        cv2.putText(
            annotated,
            caption,
            (box.x + 2, ty - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "image",
        nargs="?",
        help="path to an input image (defaults to the bundled sample portrait)",
    )
    parser.add_argument("--out", help="path for the annotated output image")
    parser.add_argument(
        "--min-score", type=float, default=0.5, help="detector confidence threshold"
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="draw detection boxes only, skip the emotion classifier",
    )
    parser.add_argument(
        "--show", action="store_true", help="display the result in a window"
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image) if args.image else _ensure_sample()
    if not image_path.exists():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"error: could not read image: {image_path}", file=sys.stderr)
        return 1

    boxes = detect_faces(frame, min_score=args.min_score)
    print(f"Detected {len(boxes)} face(s) in {image_path.name}:")

    if args.no_classify:
        annotations = [(box, f"{box.score:.2f}") for box in boxes]
        for i, box in enumerate(boxes, 1):
            print(
                f"  [{i}] x={box.x} y={box.y} w={box.width} h={box.height} "
                f"score={box.score:.3f}"
            )
    else:
        preds = classify_faces(frame, boxes)
        annotations = [
            (p.box, f"{p.label} {p.confidence:.2f}") for p in preds
        ]
        for i, p in enumerate(preds, 1):
            box = p.box
            print(
                f"  [{i}] x={box.x} y={box.y} w={box.width} h={box.height} "
                f"face={box.score:.3f}  affect={p.label} ({p.confidence:.3f})"
            )

    annotated = draw_annotations(frame, annotations)

    out_path = Path(args.out) if args.out else image_path.with_suffix(".boxes.jpg")
    cv2.imwrite(str(out_path), annotated)
    print(f"Wrote annotated image -> {out_path}")

    if args.show:
        cv2.imshow("neuroaffect — faces + affect", annotated)
        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

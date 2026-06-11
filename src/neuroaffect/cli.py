"""Command-line entry point.

Thin orchestration only — the real logic lives in the stage modules so it
stays unit-testable.

    python -m neuroaffect.cli analyze data/sample.jpg
    python -m neuroaffect.cli evaluate data/preds.json data/truth.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2

from neuroaffect.aggregation import (
    aggregate,
    aggregate_timeline,
    analyze_video,
    plot_timeline,
)
from neuroaffect.classification import classify_faces
from neuroaffect.detection import detect_faces
from neuroaffect.evaluate import evaluate

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _round_floats(obj, ndigits: int = 4):
    """Recursively round floats so the printed JSON stays readable."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _cmd_analyze(args: argparse.Namespace) -> int:
    if Path(args.image).suffix.lower() in VIDEO_EXTS:
        return _analyze_video(args)
    return _analyze_image(args)


def _analyze_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"error: could not read image: {args.image}", file=sys.stderr)
        return 1

    boxes = detect_faces(frame, min_score=args.min_score)
    predictions = classify_faces(frame, boxes)
    summary = aggregate(predictions)

    print(json.dumps(summary.__dict__, indent=2))
    return 0


def _analyze_video(args: argparse.Namespace) -> int:
    try:
        timeline = analyze_video(
            args.image, sample_fps=args.fps, min_score=args.min_score
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = aggregate_timeline(
        timeline, window_seconds=args.window, hop_seconds=args.hop
    )

    if summary.num_faces > 0:
        plot_path = args.plot or str(Path(args.image).with_suffix(".timeline.png"))
        plot_timeline(timeline, summary, plot_path)
        print(f"Wrote timeline plot -> {plot_path}", file=sys.stderr)
    else:
        print("warning: no faces detected; skipping timeline plot", file=sys.stderr)

    print(json.dumps(_round_floats(dataclasses.asdict(summary)), indent=2))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    with open(args.preds, encoding="utf-8") as f:
        y_pred = json.load(f)
    with open(args.truth, encoding="utf-8") as f:
        y_true = json.load(f)

    result = evaluate(y_true, y_pred)
    print(json.dumps(result.__dict__, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroaffect", description="Facial affect analysis pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser(
        "analyze",
        help="analyze an image (detect+classify+aggregate) or a video "
        "(temporal affect timeline + variability)",
    )
    p_analyze.add_argument("image", help="path to an input image or video")
    p_analyze.add_argument(
        "--min-score", type=float, default=0.5, help="detector confidence threshold"
    )
    # Video-only options (ignored for images).
    p_analyze.add_argument(
        "--fps", type=float, default=4.0, help="[video] frames sampled per second"
    )
    p_analyze.add_argument(
        "--window", type=float, default=3.0, help="[video] rolling window seconds"
    )
    p_analyze.add_argument(
        "--hop", type=float, default=1.0, help="[video] window hop seconds"
    )
    p_analyze.add_argument(
        "--plot", help="[video] output path for the timeline PNG"
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    p_eval = sub.add_parser("evaluate", help="score predictions vs. ground truth")
    p_eval.add_argument("preds", help="JSON file: list of predicted labels")
    p_eval.add_argument("truth", help="JSON file: list of ground-truth labels")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point.

Thin orchestration only — the real logic lives in the stage modules so it
stays unit-testable.

    python -m neuroaffect.cli analyze data/sample.jpg
    python -m neuroaffect.cli evaluate data/preds.json data/truth.json
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from neuroaffect.aggregation import aggregate
from neuroaffect.classification import classify_faces
from neuroaffect.detection import detect_faces
from neuroaffect.evaluate import evaluate


def _cmd_analyze(args: argparse.Namespace) -> int:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"error: could not read image: {args.image}", file=sys.stderr)
        return 1

    boxes = detect_faces(frame, min_score=args.min_score)
    predictions = classify_faces(frame, boxes)
    summary = aggregate(predictions)

    print(json.dumps(summary.__dict__, indent=2))
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

    p_analyze = sub.add_parser("analyze", help="detect + classify + aggregate an image")
    p_analyze.add_argument("image", help="path to an input image")
    p_analyze.add_argument(
        "--min-score", type=float, default=0.5, help="detector confidence threshold"
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

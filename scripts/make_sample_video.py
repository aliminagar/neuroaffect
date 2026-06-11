"""Synthesize a short test clip so the temporal pipeline can run end-to-end.

There's no sample video in the repo. Fetching a single Creative-Commons *video*
with a clear face is fragile (URLs rot, codecs vary), so instead we build one
from still faces:

* **labile mode (default)** — cross-fade between several *public-domain* faces
  with genuinely different expressions (happy/neutral/sad/angry), with mild
  per-frame brightness/gamma jitter on top. This produces a real, *labile*
  affect signal AND realistic frame-to-frame noise — so the rolling-window
  smoothing has something to do. Faces are cached under ``data/faces/`` so
  re-runs work offline.
* **flat mode / offline fallback** — if the faces can't be fetched, animate the
  single bundled portrait through brightness/gamma/zoom. The expression stays
  constant, so this demonstrates the *flat* (low-variability) end of the scale.

    python scripts/make_sample_video.py                  # -> data/sample_clip.mp4
    python scripts/make_sample_video.py --mode flat
    python scripts/make_sample_video.py --seconds 10 --fps 12

Sources (all public domain): the bundled MediaPipe portrait; Lincoln (1863);
Dorothea Lange's "Migrant Mother" (1936, FSA); a Beethoven portrait.
"""

from __future__ import annotations

import argparse
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PORTRAIT = DATA / "portrait.jpg"
FACE_CACHE = DATA / "faces"
DEFAULT_OUT = DATA / "sample_clip.mp4"

PORTRAIT_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"

# (slug, expected affect, Wikimedia Commons filename) — all public domain.
FACE_SOURCES = [
    ("happy", "happy", None),  # the bundled portrait
    ("neutral", "neutral", "Abraham Lincoln O-77 matte collodion print.jpg"),
    ("sad", "sad", "Lange-MigrantMother02.jpg"),
    ("angry", "angry", "Ludwig van Beethoven.jpg"),
]

_UA = {"User-Agent": "neuroaffect-demo/0.1 (portfolio; research use)"}
_CANVAS = 512


def _download(url: str, dest: Path) -> Path | None:
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        return dest
    except Exception as exc:  # noqa: BLE001 - any network/HTTP failure -> fallback
        print(f"  could not fetch {url}: {exc}", file=sys.stderr)
        return None


def _filepath_url(name: str, width: int = 600) -> str:
    quoted = urllib.parse.quote(name)
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quoted}?width={width}"


def _ensure_portrait() -> np.ndarray | None:
    if not PORTRAIT.exists() and _download(PORTRAIT_URL, PORTRAIT) is None:
        return None
    return cv2.imread(str(PORTRAIT))


def _load_face(slug: str, filename: str | None) -> np.ndarray | None:
    """Load a source face (from cache, the bundled portrait, or Wikimedia)."""
    if filename is None:
        return _ensure_portrait()
    cached = FACE_CACHE / f"{slug}.jpg"
    if cached.exists():
        return cv2.imread(str(cached))
    if _download(_filepath_url(filename), cached) is None:
        return None
    return cv2.imread(str(cached))


def _fit_to_canvas(img: np.ndarray, size: int = _CANVAS) -> np.ndarray:
    """Letterbox ``img`` onto a square gray canvas, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 96, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _gamma_lut(gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-3)
    return np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)


def _jitter(frame: np.ndarray, t: float) -> np.ndarray:
    """Mild per-frame photometric jitter -> realistic frame-level FER noise."""
    phase = 2 * math.pi * t
    alpha = 1.0 + 0.18 * math.sin(phase * 1.7)
    beta = 12.0 * math.sin(phase * 1.1)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    gamma = 1.0 + 0.22 * math.sin(phase * 0.9 + 0.5)
    return cv2.LUT(frame, _gamma_lut(gamma))


def _open_writer(out: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, out
    writer.release()
    avi = out.with_suffix(".avi")
    print(f"mp4v unavailable; falling back to MJPG -> {avi}", file=sys.stderr)
    return cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, size), avi


def build_labile(out: Path, fps: float, hold: float, trans: float) -> Path | None:
    """Cross-fade across distinct public-domain faces. Returns None if offline."""
    print("Fetching public-domain faces (cached under data/faces/) ...")
    keyframes = []
    for slug, _affect, filename in FACE_SOURCES:
        img = _load_face(slug, filename)
        if img is not None:
            keyframes.append(_fit_to_canvas(img))
    if len(keyframes) < 2:
        print("  too few faces reachable; cannot build labile clip", file=sys.stderr)
        return None
    keyframes.append(keyframes[0])  # loop back to the first expression

    hold_n, trans_n = int(round(hold * fps)), int(round(trans * fps))
    writer, actual = _open_writer(out, fps, (_CANVAS, _CANVAS))
    if not writer.isOpened():
        raise SystemExit("could not open any VideoWriter codec")

    count = 0
    try:
        for i in range(len(keyframes) - 1):
            a, b = keyframes[i], keyframes[i + 1]
            for _ in range(hold_n):
                writer.write(_jitter(a, count / fps))
                count += 1
            for k in range(trans_n):
                w = (k + 1) / (trans_n + 1)
                blended = cv2.addWeighted(a, 1 - w, b, w, 0)
                writer.write(_jitter(blended, count / fps))
                count += 1
        for _ in range(hold_n):  # final hold
            writer.write(_jitter(keyframes[-1], count / fps))
            count += 1
    finally:
        writer.release()
    print(f"Wrote labile clip: {count} frames ({count / fps:.1f}s) -> {actual}")
    return actual


def build_flat(out: Path, seconds: float, fps: float, period: float) -> Path:
    """Animate the single portrait (constant expression -> flat affect)."""
    img = _ensure_portrait()
    if img is None:
        raise SystemExit("could not obtain the sample portrait")
    h, w = img.shape[:2]
    writer, actual = _open_writer(out, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("could not open any VideoWriter codec")
    n = int(round(seconds * fps))
    try:
        for i in range(n):
            t = i / fps
            phase = 2 * math.pi * t / period
            zoom = 0.85 + 0.10 * math.sin(phase)
            cw, ch = int(w * zoom), int(h * zoom)
            dx = int((0.5 + 0.5 * math.sin(phase * 0.7)) * (w - cw))
            dy = int((0.5 + 0.5 * math.cos(phase * 0.5)) * (h - ch))
            frame = cv2.resize(img[dy : dy + ch, dx : dx + cw], (w, h))
            writer.write(_jitter(frame, t))
    finally:
        writer.release()
    print(f"Wrote flat clip: {n} frames ({seconds:.1f}s) -> {actual}")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output video path")
    parser.add_argument(
        "--mode",
        choices=("labile", "flat"),
        default="labile",
        help="labile = cross-fade distinct faces; flat = animate one portrait",
    )
    parser.add_argument("--seconds", type=float, default=8.0, help="flat-mode length")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--hold", type=float, default=1.0, help="labile: secs/face")
    parser.add_argument("--trans", type=float, default=0.6, help="labile: cross-fade")
    parser.add_argument("--period", type=float, default=4.0, help="flat: cycle secs")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if args.mode == "labile":
        result = build_labile(out, args.fps, args.hold, args.trans)
        if result is None:
            print("Falling back to flat single-portrait synthesis.", file=sys.stderr)
            build_flat(out, args.seconds, args.fps, args.period)
    else:
        build_flat(out, args.seconds, args.fps, args.period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

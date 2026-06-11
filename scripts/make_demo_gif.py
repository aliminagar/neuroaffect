"""Convert an annotated demo video into a small, looping, optimized GIF.

No ffmpeg required — reads frames with OpenCV and writes an optimized GIF with
Pillow (downscaled, reduced frame rate, quantized palette) so it auto-plays on
GitHub at a reasonable file size.

    python scripts/make_demo_gif.py \
        --in data/sample_clip.annotated.mp4 --out docs/assets/demo.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def make_gif(
    src: Path, out: Path, *, width: int, fps: float, colors: int, max_seconds: float
) -> Path:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {src}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    stride = max(1, round(src_fps / fps))
    max_frames = int(max_seconds * src_fps)

    frames: list[Image.Image] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx > max_frames:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            new_h = round(h * width / w)
            small = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            # Quantize to a compact palette to keep the GIF small.
            pil = Image.fromarray(rgb).quantize(colors=colors, method=Image.FASTOCTREE)
            frames.append(pil)
        idx += 1
    cap.release()

    if not frames:
        raise SystemExit("no frames read from video")

    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,  # loop forever
        optimize=True,
        disposal=2,
    )
    kb = out.stat().st_size / 1024
    print(f"Wrote {len(frames)} frames @ {width}px, {fps:g} fps -> {out} ({kb:.0f} KB)")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="src", default="data/sample_clip.annotated.mp4")
    p.add_argument("--out", default="docs/assets/demo.gif")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--colors", type=int, default=128)
    p.add_argument("--max-seconds", type=float, default=8.0)
    a = p.parse_args(argv)
    make_gif(
        Path(a.src), Path(a.out),
        width=a.width, fps=a.fps, colors=a.colors, max_seconds=a.max_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

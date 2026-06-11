# Stage 1 — face detector choice

`detect_faces(frame) -> list[BoundingBox]` needs a detector. The three
obvious candidates for an open-source affect pipeline are **MediaPipe Face
Detection**, **MTCNN**, and a **YOLO-face** model. This project targets a
**CPU-only, portfolio-friendly** setup: cheap to install, fast enough to feel
interactive, no GPU assumed. Here's how they compare and why MediaPipe won.

## Comparison

| Criterion | MediaPipe (BlazeFace) | MTCNN | YOLO-face (YOLOv8-face) |
| --- | --- | --- | --- |
| **CPU speed** | Excellent — real-time (designed for mobile); single forward pass | Poor — 3-stage cascade over an image pyramid; a few FPS on CPU for large images | Moderate — small `n` model is OK on CPU, but heavier than BlazeFace |
| **Small faces** | Fair — short-range model is tuned for near faces (≤~2 m) | Strong — the image pyramid is its whole point | Strong — trained on WIDER FACE, good at tiny faces |
| **Profile / rotated** | Weak — frontal-leaning | Fair | Strong |
| **Occlusion** | Weak–fair | Fair | Strong |
| **Dependency weight** | Light — one ~10 MB wheel, **no Torch/TF**; ~230 KB model fetched once | Heavy — common `mtcnn` pip pkg pulls **TensorFlow** (hundreds of MB); `facenet-pytorch` pulls **Torch** | Heavy — `ultralytics` pulls **Torch + torchvision** (~1 GB+) plus weights |
| **Setup friction** | Low — `pip install mediapipe`, model auto-downloads | Medium — TF/Torch install pain on some platforms | Medium — large download; weights not always on PyPI |
| **Landmarks** | 6 keypoints | 5 keypoints | varies by model |
| **License** | Apache-2.0 | MIT (impl) | model-dependent (some YOLOv5/v8 forks are AGPL/GPL) — watch this for a public portfolio |

## Recommendation: MediaPipe

For this project's constraints MediaPipe is the clear pick:

- **It's genuinely CPU-real-time.** The detector never becomes the bottleneck
  in a `detect → classify → aggregate` loop on a laptop.
- **It's light and self-contained.** No Torch/TensorFlow, a 10 MB wheel, and a
  ~230 KB model that downloads itself on first run. A reviewer can
  `pip install -r requirements.txt` and have it working in seconds.
- **Permissive licensing** (Apache-2.0) — safe to show off publicly, unlike
  some AGPL-licensed YOLO forks.

The honest trade-off is accuracy on **hard** faces: extreme profiles, heavy
occlusion, and very small/distant faces. For typical portrait/webcam affect
input that's fine. If a future use case is crowd analysis or surveillance-style
small faces, **YOLO-face is the upgrade path** (best hard-case accuracy, at the
cost of a Torch dependency); MTCNN isn't recommended — YOLO beats it on both
accuracy and CPU speed.

> Honorable mention: OpenCV's bundled **YuNet** DNN detector (`cv2.FaceDetectorYN`)
> is another excellent CPU-friendly option with *zero* extra dependency beyond
> `opencv-python`. MediaPipe was chosen for its slightly richer ecosystem and
> landmarks, but YuNet would be a reasonable swap behind the same interface.

## Implementation notes

- Uses MediaPipe's **Tasks API** (`mediapipe.tasks.python.vision.FaceDetector`).
  The legacy `mp.solutions.face_detection` API was removed in MediaPipe 0.10.x.
- The model is downloaded once to `~/.cache/neuroaffect/` (override with
  `NEUROAFFECT_MODEL_PATH` / `NEUROAFFECT_MODEL_DIR` for offline runs).
- MediaPipe wants RGB; OpenCV frames are BGR, so `detect_faces` converts.
- The detector runs with a low internal confidence floor; the caller's
  `min_score` does the real filtering, and boxes are clamped to the frame and
  returned sorted by descending confidence.

See it work:

```powershell
python scripts/draw_faces.py            # annotates the bundled sample portrait
python scripts/draw_faces.py my.jpg --show
```

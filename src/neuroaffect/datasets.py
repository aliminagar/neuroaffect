"""Labelled datasets for offline evaluation (Stage 4).

Currently provides the **FER-2013** test split. We recommend FER-2013 over
RAF-DB for an auto-downloading portfolio project: RAF-DB is free only for
*academic* use and requires signing a licence agreement and registering to get
the data, so it cannot be fetched unattended or redistributed. FER-2013 was
released for a public Kaggle competition, is widely mirrored, and downloads
without registration — which is what makes a reproducible ``pip install`` eval
possible.

Source
------
We pull the WebDataset mirror ``clip-benchmark/wds_fer2013`` from the Hugging
Face Hub (the test split is ~7,178 48x48 grayscale face crops). Files are
fetched with ``huggingface_hub`` (already a dependency via ``transformers``)
and cached in the standard HF cache, so re-runs are offline.

FER-2013's seven classes map onto this project's canonical labels via the
aliases in :mod:`neuroaffect.classification` (``disgusted->disgust`` etc.).
"""

from __future__ import annotations

import tarfile

import cv2
import numpy as np

from neuroaffect.classification import _normalize_label

FER2013_REPO = "clip-benchmark/wds_fer2013"
# Class order used by the mirror's integer ``.cls`` labels.
_FER2013_CLASSES = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)


def _subsample(items: list, limit: int | None) -> list:
    """Deterministically pick ``limit`` items evenly spread across ``items``.

    Even striding (rather than taking the first N) keeps the class mix roughly
    representative of the full split, so a ``--limit`` run isn't biased toward
    whatever classes happen to come first.
    """
    n = len(items)
    if limit is None or limit >= n:
        return items
    step = n / limit
    return [items[int(i * step)] for i in range(limit)]


def load_fer2013_test(limit: int | None = None) -> tuple[list[tuple[np.ndarray, str]], int]:
    """Load the FER-2013 test split as ``(image_bgr, true_label)`` pairs.

    Parameters
    ----------
    limit:
        If given, evaluate on a representative even-strided subsample of this
        many images (CPU-friendly). ``None`` loads the full test split.

    Returns
    -------
    ``(samples, total_available)`` where ``samples`` is the (possibly
    subsampled) list and ``total_available`` is the full split size.
    """
    from huggingface_hub import hf_hub_download

    def _grab(name: str) -> str:
        return hf_hub_download(FER2013_REPO, name, repo_type="dataset")

    n_shards = int(open(_grab("test/nshards.txt")).read().strip())

    # Pass 1: enumerate every sample + its label (cheap; no image decode).
    index: list[tuple[str, str, str]] = []  # (tar_path, jpg_member, label)
    for shard in range(n_shards):
        tar_path = _grab(f"test/{shard}.tar")
        with tarfile.open(tar_path) as tf:
            for member in tf.getnames():
                if not member.endswith(".cls"):
                    continue
                base = member[:-4]
                class_idx = int(tf.extractfile(member).read().decode().strip())
                label = _normalize_label(_FER2013_CLASSES[class_idx])
                index.append((tar_path, f"{base}.jpg", label))

    total = len(index)
    selected = _subsample(index, limit)

    # Pass 2: decode only the selected images, opening each tar at most once.
    by_tar: dict[str, list[tuple[str, str]]] = {}
    for tar_path, jpg_member, label in selected:
        by_tar.setdefault(tar_path, []).append((jpg_member, label))

    samples: list[tuple[np.ndarray, str]] = []
    for tar_path, members in by_tar.items():
        with tarfile.open(tar_path) as tf:
            for jpg_member, label in members:
                data = tf.extractfile(jpg_member).read()
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    samples.append((image, label))
    return samples, total

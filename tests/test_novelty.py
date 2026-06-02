"""Synthetic smoke test: novelty + peak detection on hand-crafted embeddings.

Construct 3 "scenes" of T=30 frames each, each scene being a tight cluster
around a different random unit vector. Boundaries should fire at frames
~30 and ~60.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make `src` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.novelty import compute_novelty, detect_peaks, peaks_to_segments  # noqa: E402
from src.keyframes import select_keyframes  # noqa: E402


def _make_scene(center: np.ndarray, n: int, jitter: float, rng: np.random.Generator) -> np.ndarray:
    pts = center + jitter * rng.standard_normal((n, center.shape[0]))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def test_three_scene_detection():
    rng = np.random.default_rng(0)
    D = 64
    centers = rng.standard_normal((3, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    s1 = _make_scene(centers[0], 30, 0.05, rng)
    s2 = _make_scene(centers[1], 30, 0.05, rng)
    s3 = _make_scene(centers[2], 30, 0.05, rng)
    embeddings = np.vstack([s1, s2, s3]).astype(np.float32)

    scores = compute_novelty(embeddings, memory=8, warmup=4)
    nv = detect_peaks(scores, min_gap=5, prominence_k=2.0, smoothing=3)

    print(f"threshold={nv.threshold:.4f} peaks={nv.peak_idxs.tolist()}")
    assert len(nv.peak_idxs) >= 2, f"expected >=2 boundaries, got {len(nv.peak_idxs)}"

    # Boundaries should be close to frame 30 and 60.
    near_30 = any(abs(p - 30) <= 3 for p in nv.peak_idxs)
    near_60 = any(abs(p - 60) <= 3 for p in nv.peak_idxs)
    assert near_30 and near_60, f"peaks {nv.peak_idxs.tolist()} not near 30 and 60"

    # Keyframe selection should produce 3 segments.
    segs = peaks_to_segments(nv.peak_idxs, n_frames=embeddings.shape[0])
    pts_s = np.arange(embeddings.shape[0]) / 3.0
    scenes = select_keyframes(segs, embeddings, pts_s, nv.scores)
    assert len(scenes) >= 3, f"expected >=3 scenes, got {len(scenes)}"
    print(f"scenes={len(scenes)} keyframes={[s.keyframe_idx for s in scenes]}")


if __name__ == "__main__":
    test_three_scene_detection()
    print("OK")

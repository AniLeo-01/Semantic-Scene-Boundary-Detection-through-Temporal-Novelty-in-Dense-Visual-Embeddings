"""Pick a representative keyframe per segment using centroid-closest selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Scene:
    scene_idx: int
    start_idx: int
    end_idx: int       # inclusive
    start_s: float
    end_s: float
    keyframe_idx: int  # index into the sampled stream
    novelty_peak: float


def select_keyframes(
    segments: List[Tuple[int, int]],
    embeddings: np.ndarray,
    pts_s: np.ndarray,
    scores: np.ndarray,
    method: str = "centroid",  # or "peak"
) -> List[Scene]:
    scenes: List[Scene] = []
    for i, (s, e) in enumerate(segments):
        seg = embeddings[s : e + 1]                # (L, D)
        if method == "peak":
            local = scores[s : e + 1]
            kf_local = int(np.argmax(local))
        else:
            centroid = seg.mean(axis=0, keepdims=True)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            sims = seg @ centroid.T                 # (L, 1)
            kf_local = int(np.argmax(sims))
        kf_global = s + kf_local
        scenes.append(
            Scene(
                scene_idx=i,
                start_idx=s,
                end_idx=e,
                start_s=float(pts_s[s]),
                end_s=float(pts_s[e]),
                keyframe_idx=int(kf_global),
                novelty_peak=float(scores[s : e + 1].max()),
            )
        )
    return scenes

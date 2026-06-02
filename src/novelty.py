"""Temporal memory bank + novelty scoring + adaptive peak detection.

Novelty(t) = 1 - max cosine similarity between embed_t and the K most recent
embeddings in the memory bank. Higher = more novel.

Why max instead of mean: prevents a long stretch of similar frames from
"averaging out" and falsely flagging the next normal frame as novel.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from scipy.signal import find_peaks


@dataclass
class NoveltyResult:
    scores: np.ndarray          # (T,) novelty per sampled frame
    peak_idxs: np.ndarray       # indices into scores (sampled-frame indices)
    threshold: float            # effective threshold used (median + k*MAD)


def compute_novelty(embeddings: np.ndarray, memory: int = 16, warmup: int = 4) -> np.ndarray:
    """embeddings: (T, D) L2-normalized. Returns (T,) novelty in [0, 2]."""
    T = embeddings.shape[0]
    scores = np.zeros(T, dtype=np.float32)
    if T == 0:
        return scores

    buf: deque = deque(maxlen=memory)
    for t in range(T):
        if len(buf) < warmup:
            scores[t] = 0.0
        else:
            M = np.stack(buf, axis=0)              # (k, D)
            sims = M @ embeddings[t]                # (k,)
            scores[t] = float(1.0 - sims.max())
        buf.append(embeddings[t])
    return scores


def smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(window, dtype=x.dtype) / window
    return np.convolve(xp, ker, mode="valid")


def detect_peaks(
    scores: np.ndarray,
    min_gap: int = 8,
    prominence_k: float = 2.0,
    smoothing: int = 3,
) -> NoveltyResult:
    """Adaptive peak detection on the novelty signal.

    Threshold = median(scores) + prominence_k * MAD. Robust to outliers.
    """
    y = smooth(scores, smoothing)
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)) + 1e-8)
    thresh = med + prominence_k * 1.4826 * mad  # 1.4826 makes MAD a std estimator

    peaks, _ = find_peaks(y, height=thresh, distance=max(min_gap, 1))
    return NoveltyResult(scores=y.astype(np.float32), peak_idxs=peaks.astype(np.int64), threshold=float(thresh))


def peaks_to_segments(peak_idxs: Iterable[int], n_frames: int) -> List[tuple]:
    """Convert peak indices to [(start_idx, end_idx_inclusive)] segments."""
    peaks = sorted(set(int(p) for p in peak_idxs))
    starts = [0] + peaks
    ends = [p - 1 for p in peaks] + [n_frames - 1]
    return [(s, e) for s, e in zip(starts, ends) if e >= s]

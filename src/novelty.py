"""Temporal memory bank + novelty scoring + adaptive peak detection.

Novelty(t) = 1 - max cosine similarity between embed_t and the K most recent
embeddings in the memory bank. Higher = more novel.

Why max instead of mean: prevents a long stretch of similar frames from
"averaging out" and falsely flagging the next normal frame as novel.

Peak detection uses BOTH:
  * a loose absolute height floor (so noise spikes near baseline don't count), and
  * a prominence requirement (so a peak must stand out from its local
    neighbourhood, regardless of absolute value).

The prominence test catches real but modest peaks that sit only slightly
above a noisy baseline — peaks the old height-only rule missed.
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
    threshold: float            # effective height floor used in find_peaks
    prominence: float           # effective prominence requirement used


def compute_novelty(embeddings: np.ndarray, memory: int = 16, warmup: int = 4) -> np.ndarray:
    """Pooled-vector novelty.

    embeddings: (T, D) L2-normalized. Returns (T,) novelty in [0, 2].
    Per-frame score = 1 - max cosine similarity to any frame in the
    K-frame memory bank.
    """
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


def compute_patch_novelty(
    patches: np.ndarray,
    memory: int = 16,
    warmup: int = 4,
    agg: str = "mean",
) -> np.ndarray:
    """Per-patch (Chamfer) novelty.

    Each frame holds N patch tokens (e.g. 196 for a 224x224 ViT/16).
    For each query patch we find its single best-matching patch in the
    K-frame memory bank, take that similarity, then aggregate across
    patches:

      * ``mean`` (default, recommended) — Chamfer-style:
        novelty(t) = 1 - mean_p  max_{m, k}  cos( p_t,p ,  m_k,m )
        Robust; captures the *fraction* of patches that look new.

      * ``topk`` — average of the N/4 lowest per-patch similarities:
        emphasises localised changes that mean-pooling washes out.

      * ``min`` — single most-novel patch dominates:
        novelty(t) = 1 - min_p  best_sim(p)
        High-variance; one outlier patch can fire the detector.

    Inputs:
      patches  -- (T, N, D), already L2-normalised per row
      memory   -- ring-buffer length in frames
      warmup   -- frames at the start whose novelty is forced to 0

    Returns (T,) novelty signal.

    Cost is O(T * N * K * N). For T=200, N=196, K=16 that's ~125M ops;
    fine on CPU, fast on GPU but we keep it numpy for portability.
    """
    if patches.ndim != 3:
        raise ValueError(f"patches must be (T,N,D); got {patches.shape}")
    T, N, _D = patches.shape
    scores = np.zeros(T, dtype=np.float32)
    if T == 0:
        return scores
    if agg not in ("mean", "topk", "min"):
        raise ValueError(f"unknown agg={agg!r}; expected mean|topk|min")

    buf: deque = deque(maxlen=memory)
    topk = max(1, N // 4)

    for t in range(T):
        if len(buf) < warmup:
            scores[t] = 0.0
        else:
            # (K*N, D)  — concatenate every memory frame's patches
            M = np.concatenate(list(buf), axis=0)
            # (N, K*N)  — every query patch vs every memory patch
            sims = patches[t] @ M.T
            best = sims.max(axis=1)            # (N,) best match per query
            if agg == "mean":
                score = 1.0 - float(best.mean())
            elif agg == "topk":
                # average of the smallest topk best-matches => most-novel patches
                lowest = np.partition(best, topk)[:topk]
                score = 1.0 - float(lowest.mean())
            else:  # min
                score = 1.0 - float(best.min())
            scores[t] = score
        buf.append(patches[t])
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
    height_floor_ratio: float = 0.6,
) -> NoveltyResult:
    """Adaptive peak detection on the novelty signal.

    Two adaptive thresholds derived from the data:
      * ``thresh``      = median + prominence_k * 1.4826 * MAD
        Treated as a loose absolute *height floor* (scaled by
        ``height_floor_ratio``) so that obvious noise near baseline
        is excluded.
      * ``prominence``  = prominence_k * 1.4826 * MAD
        A peak must rise by at least this much above the surrounding
        signal to count. This catches real peaks that are only
        modestly above the global baseline but clearly local maxima.

    Tuning notes:
      * Too many peaks  -> raise ``prominence_k`` (e.g. 2.5 - 3.0) or
        raise ``min_gap``.
      * Missed peaks     -> lower ``prominence_k`` (e.g. 1.5 - 1.8) or
        lower ``height_floor_ratio`` (e.g. 0.4).
      * Clusters of peaks on one transition -> raise ``min_gap`` and/or
        ``smoothing``; ``prominence_k`` should usually stay put.
    """
    y = smooth(scores, smoothing)
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)) + 1e-8)

    # Adaptive thresholds. 1.4826 makes MAD a Gaussian-std estimator.
    thresh = med + prominence_k * 1.4826 * mad
    prominence = prominence_k * 1.4826 * mad

    peaks, _ = find_peaks(
        y,
        height=thresh * height_floor_ratio,  # loose absolute floor
        prominence=prominence,               # must stand out locally
        distance=max(min_gap, 1),
    )
    return NoveltyResult(
        scores=y.astype(np.float32),
        peak_idxs=peaks.astype(np.int64),
        threshold=float(thresh),
        prominence=float(prominence),
    )


def peaks_to_segments(peak_idxs: Iterable[int], n_frames: int) -> List[tuple]:
    """Convert peak indices to [(start_idx, end_idx_inclusive)] segments."""
    peaks = sorted(set(int(p) for p in peak_idxs))
    starts = [0] + peaks
    ends = [p - 1 for p in peaks] + [n_frames - 1]
    return [(s, e) for s, e in zip(starts, ends) if e >= s]

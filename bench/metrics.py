"""Dataset-agnostic F1@rel_dis scoring for boundary detection.

A predicted boundary is a true positive iff it lies within
``rel_dis * video_duration`` seconds of an unmatched ground-truth
boundary. Greedy nearest-neighbour matching, 1-to-1 (no double-counting).

This is the standard tolerance metric used in GEBD, BBC Planet Earth,
and most other boundary-detection benchmarks. The only thing that varies
between datasets is the *shape* of the ground truth — single list per
video, or multiple annotators per video. We support both: pass a list
of lists per video and we average per-annotator before per-video.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def match_greedy(pred: Sequence[float], gt: Sequence[float], tol_s: float):
    """Greedy 1-to-1 matching between predicted and GT boundary timestamps.

    A prediction is a TP if within ``tol_s`` seconds of an unmatched GT.
    Greedy by nearest unmatched GT. Returns ``(tp, fp, fn)``.
    """
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    gt_used = [False] * len(gt)
    tp = 0
    for p in sorted(pred):
        best_i, best_d = -1, float("inf")
        for i, g in enumerate(gt):
            if gt_used[i]:
                continue
            d = abs(p - g)
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= tol_s:
            gt_used[best_i] = True
            tp += 1
    fp = len(pred) - tp
    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn


def _normalize_gt(gt) -> List[List[float]]:
    """Accept a flat list (single annotator) or a list of lists (multiple)."""
    if not gt:
        return [[]]
    if isinstance(gt[0], (int, float)):
        return [list(gt)]
    return [list(one) for one in gt]


def f1_at(
    pred_per_video: Dict[str, List[float]],
    gt_per_video: Dict[str, list],
    dur_per_video: Dict[str, float],
    rel_dis: float,
) -> dict:
    """Mean over annotators, then mean over videos."""
    per_vid_f1, per_vid_p, per_vid_r = [], [], []
    for vid in sorted(set(pred_per_video) & set(gt_per_video)):
        dur = dur_per_video.get(vid, 0.0)
        if dur <= 0:
            continue
        tol = rel_dis * dur
        anns = _normalize_gt(gt_per_video[vid])
        per_ann_f1, per_ann_p, per_ann_r = [], [], []
        for gt in anns:
            tp, fp, fn = match_greedy(pred_per_video[vid], gt, tol)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_ann_f1.append(f1); per_ann_p.append(prec); per_ann_r.append(rec)
        if per_ann_f1:
            per_vid_f1.append(float(np.mean(per_ann_f1)))
            per_vid_p.append(float(np.mean(per_ann_p)))
            per_vid_r.append(float(np.mean(per_ann_r)))
    return {
        "rel_dis": rel_dis,
        "n_videos": len(per_vid_f1),
        "precision": float(np.mean(per_vid_p)) if per_vid_p else 0.0,
        "recall": float(np.mean(per_vid_r)) if per_vid_r else 0.0,
        "f1": float(np.mean(per_vid_f1)) if per_vid_f1 else 0.0,
    }


def f1_grid(
    pred_per_video: Dict[str, List[float]],
    gt_per_video: Dict[str, list],
    dur_per_video: Dict[str, float],
    rel_dis_grid: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> List[dict]:
    return [f1_at(pred_per_video, gt_per_video, dur_per_video, rd) for rd in rel_dis_grid]


def print_table(results: List[dict]) -> None:
    print(f"{'rel_dis':>8}  {'n_videos':>8}  {'P':>6}  {'R':>6}  {'F1':>6}")
    for r in results:
        print(f"{r['rel_dis']:>8.2f}  {r['n_videos']:>8d}  "
              f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}")

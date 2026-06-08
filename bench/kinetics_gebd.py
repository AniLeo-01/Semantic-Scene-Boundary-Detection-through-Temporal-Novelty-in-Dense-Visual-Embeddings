"""Kinetics-GEBD validation harness.

Implements the official F1@rel_dis metric and a runner that wraps our
pipeline. Use this to quantify scene-boundary quality against the
canonical GEBD benchmark.

Usage:
    # 1) download labels + a sample of videos (see bench/fetch_kinetics_gebd.py)
    # 2) run predictions and score:
    python -m bench.kinetics_gebd \
        --labels  data/gebd/k400_mr345_val_min_change_duration0.3.pkl \
        --videos  data/gebd/val_videos \
        --out     outputs/gebd_run1 \
        --fps 3 --memory 16 --peak-prom 2.0 --min-gap 8 \
        --max-videos 200

    # 3) re-score cached predictions only (no re-extraction):
    python -m bench.kinetics_gebd \
        --labels data/gebd/k400_mr345_val_min_change_duration0.3.pkl \
        --predictions outputs/gebd_run1/predictions.json \
        --eval-only

Reference paper / data / official eval:
    Shou et al., "Generic Event Boundary Detection: A Benchmark
        for Event Segmentation," ICCV 2021.
    Repo: https://github.com/StanLei52/GEBD
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_gebd_labels(labels_path: str) -> Dict[str, dict]:
    """Load the official GEBD pickle/json.

    The official file is a pickle keyed by video_id, each value containing:
        {
          'fps': float,
          'num_frames': int,
          'path_video': str,
          'substages_timestamps': List[List[float]],  # one list per annotator
          'video_duration': float,
        }
    Older JSON exports use the same fields. We accept either format.
    """
    p = Path(labels_path)
    if p.suffix == ".json":
        with open(p) as f:
            labels = json.load(f)
    else:
        with open(p, "rb") as f:
            head = f.read(8)
            f.seek(0)
            if head.startswith(b"<") or head.startswith(b"\n<") or head.startswith(b"version "):
                raise RuntimeError(
                    f"{p} is not a pickle. First bytes: {head!r}\n"
                    "Your download saved an HTML page or a Git-LFS pointer, not the\n"
                    "actual labels. The official GEBD labels live on Google Drive:\n"
                    "  https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo\n"
                    "  pip install gdown && gdown --folder <that URL> -O data/gebd"
                )
            labels = pickle.load(f)
    if not isinstance(labels, dict):
        raise ValueError(f"unexpected label format in {labels_path}")
    return labels


def gt_boundaries(entry: dict) -> List[List[float]]:
    """Return per-annotator boundary lists in seconds.

    Handles BOTH the raw annotation format (``substages_timestamps`` as a
    list of per-annotator entries, each an "A: B"-labelled list) and the
    processed format (``substages_myframeidx`` as frame indices).
    """
    fps = float(entry.get("fps", 30.0))
    dur = float(entry.get("video_duration", 1e9))

    # Processed format: frame indices, one list per annotator
    if "substages_myframeidx" in entry and entry["substages_myframeidx"]:
        ann = entry["substages_myframeidx"]
        out = []
        for one in ann:
            seq = sorted(float(x) / fps if fps > 0 else 0.0 for x in one)
            out.append(seq)
        return out

    # Raw format: each annotator's entry is a list of dicts with
    # start_time / end_time / label. Range entries (start < end) describe
    # a gradual transition; the conventional boundary point is the midpoint.
    ann = entry.get("substages_timestamps") or []
    out: List[List[float]] = []
    for one in ann:
        seq: List[float] = []
        for item in one:
            if isinstance(item, dict):
                if "start_time" in item and "end_time" in item:
                    s = float(item["start_time"]); e = float(item["end_time"])
                    seq.append((s + e) / 2.0)
                    continue
                v = item.get("timestamp", item.get("time"))
                if v is not None:
                    try:
                        seq.append(float(v))
                    except (TypeError, ValueError):
                        pass
                continue
            if isinstance(item, (list, tuple)) and item:
                # legacy: bare timestamp pair
                if len(item) >= 2:
                    try:
                        seq.append((float(item[0]) + float(item[1])) / 2.0)
                        continue
                    except (TypeError, ValueError):
                        pass
                item = item[0]
            try:
                seq.append(float(item))
            except (TypeError, ValueError):
                continue
        if seq and max(seq) > dur * 2 and fps > 0:
            seq = [x / fps for x in seq]
        out.append(sorted(seq))
    return out


def video_duration(entry: dict) -> float:
    if "video_duration" in entry:
        return float(entry["video_duration"])
    fps = float(entry.get("fps", 30.0))
    n = int(entry.get("num_frames", 0))
    return n / fps if fps > 0 else 0.0


# ---------------------------------------------------------------------------
# Metric: F1 @ rel_dis
# ---------------------------------------------------------------------------

def _match_greedy(pred: Sequence[float], gt: Sequence[float],
                  tol_s: float) -> Tuple[int, int, int]:
    """Greedy 1-to-1 matching between predicted and GT boundary timestamps.

    A prediction is a TP if within ``tol_s`` seconds of an unmatched GT.
    Returns (tp, fp, fn). Greedy by nearest unmatched GT — matches the
    behaviour of the official GEBD eval script.
    """
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0

    pred_sorted = sorted(pred)
    gt_used = [False] * len(gt)
    tp = 0
    for p in pred_sorted:
        # find nearest unmatched gt
        best_i = -1
        best_d = float("inf")
        for i, g in enumerate(gt):
            if gt_used[i]:
                continue
            d = abs(p - g)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d <= tol_s:
            gt_used[best_i] = True
            tp += 1
    fp = len(pred) - tp
    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn


def f1_at(pred_per_video: Dict[str, List[float]],
          gt_per_video: Dict[str, List[List[float]]],
          dur_per_video: Dict[str, float],
          rel_dis: float) -> dict:
    """Mean over videos, mean over annotators — the official GEBD protocol."""
    per_vid_f1: List[float] = []
    per_vid_p: List[float] = []
    per_vid_r: List[float] = []
    common = sorted(set(pred_per_video) & set(gt_per_video))
    for vid in common:
        dur = dur_per_video.get(vid, 0.0)
        if dur <= 0:
            continue
        tol = rel_dis * dur
        per_ann_f1, per_ann_p, per_ann_r = [], [], []
        for gt in gt_per_video[vid]:
            tp, fp, fn = _match_greedy(pred_per_video[vid], gt, tol)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_ann_f1.append(f1)
            per_ann_p.append(prec)
            per_ann_r.append(rec)
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


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_one(video_path: str, *, fps: float, memory: int,
                peak_prom: float, min_gap: int, smoothing: int,
                use_patches: bool, model: str | None,
                batch_size: int, extractor) -> Tuple[List[float], float]:
    """Run the full pipeline on a single video and return (boundary_seconds,
    duration_s).

    ``extractor`` is a preloaded DinoFeatureExtractor (don't reload per video).
    """
    # Lazy imports keep the eval-only path free of torch / av deps.
    from src.sampling import sample_frames, video_meta
    from src.novelty import compute_novelty, detect_peaks
    import numpy as _np

    dur, _src_fps, _n = video_meta(video_path)

    images, idxs, pts = [], [], []
    for sf in sample_frames(video_path, target_fps=fps):
        images.append(sf.image); idxs.append(sf.idx); pts.append(sf.pts_s)
    if not images:
        return [], dur

    # batched embed
    all_vec = []
    for i in range(0, len(images), batch_size):
        embs = extractor.embed_batch(
            images[i:i + batch_size],
            idxs[i:i + batch_size],
            pts[i:i + batch_size],
        )
        for e in embs:
            all_vec.append(e.combined if use_patches else e.cls)
    E = _np.stack(all_vec, axis=0).astype(_np.float32)

    scores = compute_novelty(E, memory=memory)
    nv = detect_peaks(scores, min_gap=min_gap, prominence_k=peak_prom, smoothing=smoothing)
    boundary_s = [float(pts[i]) for i in nv.peak_idxs.tolist()]
    return boundary_s, dur


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_eval(labels_path: str, videos_dir: str | None, out_dir: str,
             *, predictions_path: str | None = None,
             fps: float = 3.0, memory: int = 16, peak_prom: float = 2.0,
             min_gap: int = 8, smoothing: int = 3, use_patches: bool = True,
             model: str | None = None, batch_size: int = 16,
             max_videos: int | None = None,
             rel_dis_grid: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
             ) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    labels = load_gebd_labels(labels_path)
    print(f"[gebd] {len(labels)} videos in label file")

    # ----- predictions -----
    pred_per_video: Dict[str, List[float]] = {}
    if predictions_path and Path(predictions_path).exists():
        with open(predictions_path) as f:
            pred_per_video = json.load(f)
        print(f"[gebd] loaded cached predictions for {len(pred_per_video)} videos")
    else:
        from src.features import DinoFeatureExtractor
        extractor = DinoFeatureExtractor(model_name=model)
        print(f"[gebd] model: {extractor.model_name} on {extractor.device}")

        vids = sorted(labels.keys())
        if max_videos:
            vids = vids[:max_videos]

        t0 = time.time()
        for i, vid in enumerate(vids):
            entry = labels[vid]
            # Resolve video path: <videos_dir>/<vid>.mp4 by default.
            cand = []
            if videos_dir:
                for ext in (".mp4", ".webm", ".mkv"):
                    cand.append(Path(videos_dir) / f"{vid}{ext}")
            if "path_video" in entry:
                cand.insert(0, Path(entry["path_video"]))
            vpath = next((p for p in cand if p.exists()), None)
            if vpath is None:
                continue

            try:
                bd, _ = predict_one(
                    str(vpath), fps=fps, memory=memory,
                    peak_prom=peak_prom, min_gap=min_gap, smoothing=smoothing,
                    use_patches=use_patches, model=model,
                    batch_size=batch_size, extractor=extractor,
                )
                pred_per_video[vid] = bd
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {vid}: {e}", file=sys.stderr)
                traceback.print_exc(limit=1, file=sys.stderr)
            if (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  [{i+1}/{len(vids)}] elapsed={dt:.1f}s "
                      f"({(i+1)/dt:.2f} vid/s) cached={len(pred_per_video)}")

        with open(out / "predictions.json", "w") as f:
            json.dump(pred_per_video, f)

    # ----- evaluation -----
    gt_per_video = {vid: gt_boundaries(labels[vid]) for vid in pred_per_video}
    dur_per_video = {vid: video_duration(labels[vid]) for vid in pred_per_video}

    results: List[dict] = []
    print(f"\n{'rel_dis':>8}  {'n_videos':>8}  {'P':>6}  {'R':>6}  {'F1':>6}")
    for rd in rel_dis_grid:
        r = f1_at(pred_per_video, gt_per_video, dur_per_video, rd)
        results.append(r)
        print(f"{rd:>8.2f}  {r['n_videos']:>8d}  "
              f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}")

    summary = {
        "config": {
            "fps": fps, "memory": memory, "peak_prom": peak_prom,
            "min_gap": min_gap, "smoothing": smoothing,
            "use_patches": use_patches, "model": model, "batch_size": batch_size,
        },
        "n_predicted": len(pred_per_video),
        "n_evaluated": results[0]["n_videos"] if results else 0,
        "metrics": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True,
                   help="path to k400_mr345_val_min_change_duration0.3.pkl (or .json)")
    p.add_argument("--videos", default=None, help="directory of .mp4 files keyed by GEBD video_id")
    p.add_argument("--out", required=True)
    p.add_argument("--predictions", default=None,
                   help="path to predictions.json to re-score without re-extracting")
    p.add_argument("--eval-only", action="store_true", help="alias for: skip prediction if --predictions exists")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--memory", type=int, default=16)
    p.add_argument("--peak-prom", type=float, default=2.0)
    p.add_argument("--min-gap", type=int, default=8)
    p.add_argument("--smoothing", type=int, default=3)
    p.add_argument("--no-patches", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-videos", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        labels_path=args.labels,
        videos_dir=args.videos,
        out_dir=args.out,
        predictions_path=args.predictions,
        fps=args.fps,
        memory=args.memory,
        peak_prom=args.peak_prom,
        min_gap=args.min_gap,
        smoothing=args.smoothing,
        use_patches=not args.no_patches,
        model=args.model,
        batch_size=args.batch_size,
        max_videos=args.max_videos,
    )

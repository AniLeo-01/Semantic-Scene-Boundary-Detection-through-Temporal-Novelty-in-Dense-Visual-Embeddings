"""Cheap baselines to compare DINO+novelty against.

We need these to argue our pipeline isn't just rediscovering pixel
statistics. Two baselines implemented:

  * **frame_diff** — mean absolute intensity difference between
    consecutive sampled frames, MAD-thresholded peak detection. The
    textbook "shot detector for poor people."
  * **uniform** — N predictions placed at evenly spaced fractions of the
    clip duration. ``N`` matches either the model's prediction count
    (``match-pred``) or the GT count (``match-gt``).

Both baselines reuse the metric module so the output table looks
identical to the DINO bench.

CLI examples
------------

::

    # frame-diff baseline on Charades
    python -m bench.baselines \\
        --dataset charades \\
        --annotations data/charades/Charades/Charades_v1_test.csv \\
        --videos      data/charades/Charades_v1_480 \\
        --out         outputs/charades_frame_diff \\
        --baseline    frame_diff \\
        --fps 5 --peak-prom 1.8 --min-gap 3 --max-videos 200

    # uniform-spacing baseline matched to the model's prediction count
    python -m bench.baselines \\
        --dataset charades \\
        --annotations data/charades/Charades/Charades_v1_test.csv \\
        --predictions outputs/charades_run1/predictions.json \\
        --out         outputs/charades_uniform_matchpred \\
        --baseline    uniform --n-strategy match-pred

    # frame-diff on hand-annotated custom videos
    python -m bench.baselines \\
        --dataset custom \\
        --labels  data/custom/labels.json \\
        --videos  data/custom/videos \\
        --out     outputs/custom_frame_diff \\
        --baseline frame_diff --fps 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .metrics import f1_grid, print_table


# ---------------------------------------------------------------------------
# Frame-difference baseline
# ---------------------------------------------------------------------------

def _grayscale_resized(frame_rgb: np.ndarray, size=(160, 120)) -> np.ndarray:
    """RGB uint8 HxWx3 -> small grayscale int16 array. No cv2 dep."""
    from PIL import Image
    img = Image.fromarray(frame_rgb).convert("L").resize(size)
    return np.asarray(img, dtype=np.int16)


def frame_diff_signal(video_path: str, fps: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample frames at ``fps`` and return (diffs, pts_s, duration_s).

    ``diffs[t]`` is the mean abs grayscale diff between sampled frames
    ``t-1`` and ``t``. ``diffs[0]`` is 0.
    """
    from src.sampling import sample_frames, video_meta
    dur, _, _ = video_meta(video_path)
    diffs: List[float] = []
    pts: List[float] = []
    prev: np.ndarray | None = None
    for sf in sample_frames(video_path, target_fps=fps):
        g = _grayscale_resized(sf.image)
        if prev is None:
            diffs.append(0.0)
        else:
            diffs.append(float(np.mean(np.abs(g - prev))))
        pts.append(sf.pts_s)
        prev = g
    return np.asarray(diffs, dtype=np.float32), np.asarray(pts, dtype=np.float64), float(dur)


def predict_frame_diff(video_path: str, *, fps: float, peak_prom: float,
                       min_gap: int, smoothing: int) -> Tuple[List[float], float]:
    from src.novelty import detect_peaks
    diffs, pts, dur = frame_diff_signal(video_path, fps=fps)
    if len(diffs) < 3:
        return [], dur
    nv = detect_peaks(diffs, min_gap=min_gap, prominence_k=peak_prom, smoothing=smoothing)
    return [float(pts[i]) for i in nv.peak_idxs.tolist()], dur


# ---------------------------------------------------------------------------
# Uniform-spacing baseline
# ---------------------------------------------------------------------------

def predict_uniform(duration_s: float, n: int) -> List[float]:
    """Place ``n`` boundaries evenly across [0, duration_s], avoiding endpoints."""
    if n <= 0 or duration_s <= 0:
        return []
    return [duration_s * (i + 1) / (n + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

def _load_charades(annotations_csv: str) -> Tuple[Dict[str, dict], str]:
    from .charades import load_charades_labels
    labels = load_charades_labels(annotations_csv)
    return labels, "id"


def _load_custom(labels_path: str, videos_dir: str | None) -> Tuple[Dict[str, dict], str]:
    from .custom import load_custom_labels
    labels = load_custom_labels(labels_path, videos_dir)
    return labels, "id"


def _video_path_for(dataset: str, videos_dir: str, vid: str, entry: dict | None = None) -> Path | None:
    if dataset == "charades":
        from .charades import video_path_for as _v
        return _v(videos_dir, vid)
    # custom
    from .charades import video_path_for as _v
    fname = (entry or {}).get("_filename")
    return _v(videos_dir, fname or vid) if fname else _v(videos_dir, vid)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baseline(
    dataset: str,
    out_dir: str,
    *,
    baseline: str,
    annotations: str | None = None,
    labels: str | None = None,
    videos_dir: str | None = None,
    predictions_path: str | None = None,
    fps: float = 5.0, peak_prom: float = 1.8, min_gap: int = 3, smoothing: int = 1,
    n_strategy: str = "match-gt",  # "match-pred" | "match-gt" | "fixed:<N>"
    max_videos: int | None = None,
    rel_dis_grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load GT
    if dataset == "charades":
        if not annotations:
            sys.exit("error: --annotations required for dataset=charades")
        labels_map, _ = _load_charades(annotations)
    elif dataset == "custom":
        if not labels:
            sys.exit("error: --labels required for dataset=custom")
        labels_map, _ = _load_custom(labels, videos_dir)
    else:
        sys.exit(f"unknown dataset {dataset!r}; expected charades|custom")
    print(f"[baseline:{baseline}] {len(labels_map)} videos in label set")

    # Possibly load existing model predictions for the "match-pred" strategy.
    model_preds: Dict[str, List[float]] = {}
    if predictions_path and Path(predictions_path).exists():
        with open(predictions_path) as f:
            model_preds = json.load(f)
        print(f"[baseline] loaded model predictions for {len(model_preds)} videos")

    pred_per_video: Dict[str, List[float]] = {}
    t0 = time.time()

    if baseline == "uniform":
        for vid in sorted(labels_map.keys()):
            entry = labels_map[vid]
            dur = entry.get("duration_s", 0.0)
            if dur <= 0:
                continue
            gt = entry.get("boundaries") or []
            gt_count = len(gt[0]) if (gt and isinstance(gt[0], list)) else len(gt)
            if n_strategy == "match-pred":
                n = len(model_preds.get(vid, []))
            elif n_strategy == "match-gt":
                n = gt_count
            elif n_strategy.startswith("fixed:"):
                n = int(n_strategy.split(":")[1])
            else:
                sys.exit(f"unknown --n-strategy {n_strategy!r}")
            pred_per_video[vid] = predict_uniform(dur, n)

    elif baseline == "frame_diff":
        if not videos_dir:
            sys.exit("error: --videos required for baseline=frame_diff")
        vids = sorted(labels_map.keys())
        if max_videos:
            vids = vids[:max_videos]
        for i, vid in enumerate(vids, 1):
            entry = labels_map[vid]
            vpath = _video_path_for(dataset, videos_dir, vid, entry)
            if vpath is None:
                continue
            try:
                bd, dur = predict_frame_diff(
                    str(vpath), fps=fps,
                    peak_prom=peak_prom, min_gap=min_gap, smoothing=smoothing,
                )
                pred_per_video[vid] = bd
                if entry.get("duration_s", 0.0) <= 0:
                    entry["duration_s"] = dur
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {vid}: {e}", file=sys.stderr)
            if i % 50 == 0:
                dt = time.time() - t0
                print(f"  [{i}/{len(vids)}] elapsed={dt:.1f}s ({i/dt:.1f} vid/s)")

        with open(out / "predictions.json", "w") as f:
            json.dump(pred_per_video, f)
    else:
        sys.exit(f"unknown baseline {baseline!r}; expected uniform|frame_diff")

    # Score
    nonempty = [v for v in pred_per_video if labels_map[v].get("boundaries")]
    print(f"[baseline] {len(nonempty)}/{len(pred_per_video)} videos have ≥1 GT boundary")
    gt_per_video = {v: labels_map[v]["boundaries"] for v in nonempty}
    dur_per_video = {v: labels_map[v].get("duration_s", 0.0) for v in nonempty}
    results = f1_grid({v: pred_per_video[v] for v in nonempty}, gt_per_video, dur_per_video,
                      rel_dis_grid)
    print()
    print_table(results)

    summary = {
        "dataset": dataset,
        "baseline": baseline,
        "config": {
            "fps": fps, "peak_prom": peak_prom, "min_gap": min_gap,
            "smoothing": smoothing, "n_strategy": n_strategy,
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
    p.add_argument("--dataset", required=True, choices=["charades", "custom"])
    p.add_argument("--baseline", required=True, choices=["frame_diff", "uniform"])
    p.add_argument("--out", required=True)
    p.add_argument("--annotations", default=None, help="charades CSV (charades dataset)")
    p.add_argument("--labels", default=None, help="JSON/CSV/TXT (custom dataset)")
    p.add_argument("--videos", default=None, help="dir of source videos (required for frame_diff)")
    p.add_argument("--predictions", default=None,
                   help="model predictions.json — only used by uniform --n-strategy match-pred")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--peak-prom", type=float, default=1.8)
    p.add_argument("--min-gap", type=int, default=3)
    p.add_argument("--smoothing", type=int, default=1)
    p.add_argument("--n-strategy", default="match-gt",
                   help="how many boundaries to place (uniform only): match-pred | match-gt | fixed:N")
    p.add_argument("--max-videos", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        dataset=args.dataset,
        out_dir=args.out,
        baseline=args.baseline,
        annotations=args.annotations,
        labels=args.labels,
        videos_dir=args.videos,
        predictions_path=args.predictions,
        fps=args.fps,
        peak_prom=args.peak_prom,
        min_gap=args.min_gap,
        smoothing=args.smoothing,
        n_strategy=args.n_strategy,
        max_videos=args.max_videos,
    )

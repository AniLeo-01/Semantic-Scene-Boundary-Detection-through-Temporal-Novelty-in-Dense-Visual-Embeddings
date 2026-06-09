"""Charades scene-boundary benchmark.

Treats Charades **action segments** as scene units: each segment is
"do this activity from t1 to t2". A boundary is a moment where the
active-action set changes — operationally, the union of all segment
start_times and end_times in the video, deduplicated within a small
tolerance.

Annotation file shape (Charades_v1_train.csv / Charades_v1_test.csv)
-------------------------------------------------------------------
CSV with columns: id, subject, scene, quality, relevance, verified,
script, objects, descriptions, actions, length.

``actions`` is a semicolon-separated list of triples
``c<class> <start_sec> <end_sec>``. Example::

    "c092 11.10 21.50;c147 23.10 31.50"

We ignore the action class and only use the timestamps.

Usage::

    python -m bench.charades \\
        --annotations data/charades/Charades/Charades_v1_test.csv \\
        --videos      data/charades/Charades_v1_480 \\
        --out         outputs/charades_run1 \\
        --model       facebook/dinov3-vits16-pretrain-lvd1689m \\
        --fps 5 --memory 12 --peak-prom 1.8 --min-gap 3 --batch-size 64 \\
        --max-videos  500

Re-score cached predictions only::

    python -m bench.charades --annotations ... --predictions ... --out ... --eval-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .metrics import f1_grid, print_table


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def _parse_actions(actions_field: str) -> List[Tuple[float, float]]:
    """``"c092 11.10 21.50;c147 23.10 31.50"`` -> [(11.10, 21.50), (23.10, 31.50)]."""
    out: List[Tuple[float, float]] = []
    if not actions_field:
        return out
    for tok in actions_field.split(";"):
        parts = tok.strip().split()
        if len(parts) < 3:
            continue
        try:
            s = float(parts[-2]); e = float(parts[-1])
            if e >= s:
                out.append((s, e))
        except ValueError:
            continue
    return out


def segments_to_boundaries(
    segments: List[Tuple[float, float]],
    duration: float,
    dedup_tol_s: float = 0.5,
) -> List[float]:
    """Union of all start/end times, deduplicated by ``dedup_tol_s``.

    Boundaries at 0 or at the very end of the clip are dropped — those
    aren't transitions, they're the video's start/stop.
    """
    if not segments:
        return []
    pts = sorted({round(s, 3) for s, _ in segments} | {round(e, 3) for _, e in segments})
    out: List[float] = []
    last = -1e9
    for p in pts:
        if p < 0.05:
            continue
        if duration > 0 and p > duration - 0.05:
            continue
        if p - last < dedup_tol_s:
            continue
        out.append(p)
        last = p
    return out


def load_charades_labels(
    csv_path: str,
    dedup_tol_s: float = 0.5,
) -> Dict[str, dict]:
    """Return ``{video_id: {duration_s, boundaries}}`` from a Charades CSV."""
    out: Dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("id")
            if not vid:
                continue
            try:
                duration = float(row.get("length") or 0.0)
            except ValueError:
                duration = 0.0
            segs = _parse_actions(row.get("actions") or "")
            boundaries = segments_to_boundaries(segs, duration, dedup_tol_s)
            out[vid] = {
                "duration_s": duration,
                "boundaries": boundaries,
                "n_actions": len(segs),
            }
    return out


def video_path_for(videos_dir: str, vid: str) -> Path | None:
    base = Path(videos_dir)
    for ext in (".mp4", ".webm", ".mkv"):
        cand = base / f"{vid}{ext}"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# Prediction (one video)
# ---------------------------------------------------------------------------

def predict_one(video_path: str, *, fps: float, memory: int,
                peak_prom: float, min_gap: int, smoothing: int,
                use_patches: bool, batch_size: int, extractor,
                patch_novelty: bool = False,
                patch_agg: str = "mean") -> Tuple[List[float], float]:
    from src.sampling import sample_frames, video_meta
    from src.novelty import compute_novelty, compute_patch_novelty, detect_peaks

    dur, _src_fps, _n = video_meta(video_path)

    images, idxs, pts = [], [], []
    for sf in sample_frames(video_path, target_fps=fps):
        images.append(sf.image); idxs.append(sf.idx); pts.append(sf.pts_s)
    if not images:
        return [], dur

    all_vec: list = []
    all_patches: list = []
    for i in range(0, len(images), batch_size):
        embs = extractor.embed_batch(
            images[i:i + batch_size],
            idxs[i:i + batch_size],
            pts[i:i + batch_size],
        )
        for e in embs:
            all_vec.append(e.combined if use_patches else e.cls)
            if patch_novelty:
                all_patches.append(e.patches)
    E = np.stack(all_vec, axis=0).astype(np.float32)

    if patch_novelty:
        P = np.stack(all_patches, axis=0).astype(np.float32)  # (T, N, D)
        scores = compute_patch_novelty(P, memory=memory, agg=patch_agg)
    else:
        scores = compute_novelty(E, memory=memory)
    nv = detect_peaks(scores, min_gap=min_gap, prominence_k=peak_prom, smoothing=smoothing)
    return [float(pts[i]) for i in nv.peak_idxs.tolist()], dur


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(
    annotations_csv: str, videos_dir: str | None, out_dir: str,
    *,
    predictions_path: str | None = None,
    dedup_tol_s: float = 0.5,
    fps: float = 5.0, memory: int = 12, peak_prom: float = 1.8,
    min_gap: int = 3, smoothing: int = 1, use_patches: bool = True,
    model: str | None = None, batch_size: int = 32,
    max_videos: int | None = None,
    patch_novelty: bool = False, patch_agg: str = "mean",
    rel_dis_grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = load_charades_labels(annotations_csv, dedup_tol_s=dedup_tol_s)
    print(f"[charades] {len(labels)} videos in CSV")

    pred_per_video: Dict[str, List[float]] = {}
    if predictions_path and Path(predictions_path).exists():
        with open(predictions_path) as f:
            pred_per_video = json.load(f)
        print(f"[charades] loaded cached predictions for {len(pred_per_video)} videos")
    else:
        if not videos_dir:
            sys.exit("error: --videos required when no cached predictions exist")

        from src.features import DinoFeatureExtractor
        extractor = DinoFeatureExtractor(model_name=model)
        print(f"[charades] model: {extractor.model_name} on {extractor.device}")

        vids = sorted(labels.keys())
        if max_videos:
            vids = vids[:max_videos]

        t0 = time.time()
        for i, vid in enumerate(vids, 1):
            vpath = video_path_for(videos_dir, vid)
            if vpath is None:
                continue
            try:
                bd, dur = predict_one(
                    str(vpath), fps=fps, memory=memory,
                    peak_prom=peak_prom, min_gap=min_gap, smoothing=smoothing,
                    use_patches=use_patches, batch_size=batch_size,
                    extractor=extractor,
                    patch_novelty=patch_novelty, patch_agg=patch_agg,
                )
                pred_per_video[vid] = bd
                if labels[vid].get("duration_s", 0.0) <= 0:
                    labels[vid]["duration_s"] = dur
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {vid}: {e}", file=sys.stderr)
                traceback.print_exc(limit=1, file=sys.stderr)

            if i % 50 == 0:
                dt = time.time() - t0
                print(f"  [{i}/{len(vids)}] elapsed={dt:.1f}s "
                      f"({i/dt:.1f} vid/s) cached={len(pred_per_video)}")

        with open(out / "predictions.json", "w") as f:
            json.dump(pred_per_video, f)

    # ----- eval -----
    gt_per_video = {vid: labels[vid]["boundaries"] for vid in pred_per_video}
    dur_per_video = {vid: labels[vid].get("duration_s", 0.0) for vid in pred_per_video}

    # Charades videos sometimes have 0 actions in the test split — skip those.
    nonempty = [vid for vid in pred_per_video if gt_per_video.get(vid)]
    print(f"[charades] {len(nonempty)}/{len(pred_per_video)} videos have ≥1 GT boundary")

    results = f1_grid(
        {vid: pred_per_video[vid] for vid in nonempty},
        {vid: gt_per_video[vid] for vid in nonempty},
        {vid: dur_per_video[vid] for vid in nonempty},
        rel_dis_grid,
    )
    print()
    print_table(results)

    summary = {
        "dataset": "Charades",
        "config": {
            "fps": fps, "memory": memory, "peak_prom": peak_prom,
            "min_gap": min_gap, "smoothing": smoothing, "dedup_tol_s": dedup_tol_s,
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
    p.add_argument("--annotations", required=True, help="Charades_v1_test.csv (or _train.csv)")
    p.add_argument("--videos", default=None, help="directory of Charades videos (id.mp4)")
    p.add_argument("--out", required=True)
    p.add_argument("--predictions", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--memory", type=int, default=12)
    p.add_argument("--peak-prom", type=float, default=1.8)
    p.add_argument("--min-gap", type=int, default=3)
    p.add_argument("--smoothing", type=int, default=1)
    p.add_argument("--dedup-tol-s", type=float, default=0.5,
                   help="merge GT boundaries within this many seconds (default 0.5)")
    p.add_argument("--no-patches", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--patch-novelty", action="store_true",
                   help="use per-patch Chamfer novelty over raw patch tokens")
    p.add_argument("--patch-agg", choices=["mean", "topk", "min"], default="mean",
                   help="patch-novelty aggregation rule (default: mean)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        annotations_csv=args.annotations,
        videos_dir=args.videos,
        out_dir=args.out,
        predictions_path=args.predictions,
        dedup_tol_s=args.dedup_tol_s,
        fps=args.fps,
        memory=args.memory,
        peak_prom=args.peak_prom,
        min_gap=args.min_gap,
        smoothing=args.smoothing,
        use_patches=not args.no_patches,
        model=args.model,
        batch_size=args.batch_size,
        max_videos=args.max_videos,
        patch_novelty=args.patch_novelty,
        patch_agg=args.patch_agg,
    )

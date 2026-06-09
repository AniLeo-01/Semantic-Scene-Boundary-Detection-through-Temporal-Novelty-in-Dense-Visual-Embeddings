"""BBC Planet Earth scene-boundary benchmark.

The BBC Planet Earth dataset is the canonical small benchmark for
*narrative* scene segmentation — 11 ~50-minute documentary episodes
with human-annotated scene boundaries. Multiple annotators exist for
most episodes; we average F1 per annotator and then per episode, same
as Kinetics-GEBD.

Label-file format
-----------------
We accept three shapes — pick whichever your annotation source uses:

(a) JSON, single annotator per episode::

    {
      "EP01": {"fps": 25, "duration_s": 3010.0,
               "boundaries": [12.4, 81.0, 154.6, ...]},
      "EP02": {...}
    }

(b) JSON, multiple annotators per episode::

    {
      "EP01": {"fps": 25, "duration_s": 3010.0,
               "boundaries": [[12.4, 81.0, ...], [13.0, 80.5, ...]]},
      "EP02": {...}
    }

(c) CSV with one row per scene boundary::

    episode,boundary_s
    EP01,12.4
    EP01,81.0
    EP02,9.8
    ...

  When using CSV, also supply ``--durations  episode_durations.json``
  with ``{episode: duration_seconds}``.

Usage
-----

    python -m bench.bbc \\
        --labels  data/bbc/bbc_labels.json \\
        --videos  data/bbc/episodes \\
        --out     outputs/bbc_run1 \\
        --fps 3 --memory 24 --peak-prom 2.0 --min-gap 30 \\
        --model facebook/dinov3-vits16-pretrain-lvd1689m \\
        --batch-size 64

Re-score cached predictions only::

    python -m bench.bbc --labels ... --predictions outputs/bbc_run1/predictions.json \\
        --out outputs/bbc_rescored --eval-only
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

def _load_csv(path: Path, durations_path: Path | None) -> Dict[str, dict]:
    if durations_path is None or not durations_path.exists():
        raise RuntimeError(
            f"CSV labels need a companion duration file. Pass --durations "
            f"<json> mapping episode -> seconds."
        )
    durations = json.loads(durations_path.read_text())
    by_ep: Dict[str, List[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row.get("episode") or row.get("video") or row.get("video_id")
            t = float(row.get("boundary_s") or row.get("time") or row.get("seconds"))
            by_ep.setdefault(ep, []).append(t)
    return {
        ep: {"duration_s": float(durations.get(ep, 0.0)),
             "boundaries": sorted(bs)}
        for ep, bs in by_ep.items()
    }


def load_bbc_labels(labels_path: str, durations_path: str | None = None) -> Dict[str, dict]:
    """Return ``{episode_id: {duration_s, boundaries}}``.

    ``boundaries`` is a list (single annotator) or list of lists (multiple).
    """
    p = Path(labels_path)
    if p.suffix.lower() == ".csv":
        return _load_csv(p, Path(durations_path) if durations_path else None)

    with open(p) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"unexpected label JSON shape in {p}")

    out: Dict[str, dict] = {}
    for ep, entry in raw.items():
        if isinstance(entry, list):
            # bare list of boundaries — duration unknown
            out[ep] = {"duration_s": 0.0, "boundaries": entry}
        else:
            out[ep] = {
                "duration_s": float(entry.get("duration_s") or entry.get("video_duration", 0.0)),
                "boundaries": entry.get("boundaries", []),
            }
    return out


def episode_to_video_path(videos_dir: str, episode_id: str) -> Path | None:
    base = Path(videos_dir)
    for ext in (".mp4", ".mkv", ".webm", ".avi", ".m4v"):
        cand = base / f"{episode_id}{ext}"
        if cand.exists():
            return cand
    # case-insensitive fallback
    for cand in base.iterdir():
        if cand.stem.lower() == episode_id.lower() and cand.suffix.lower() in (".mp4", ".mkv", ".webm", ".avi", ".m4v"):
            return cand
    return None


# ---------------------------------------------------------------------------
# Prediction (one episode)
# ---------------------------------------------------------------------------

def predict_one(video_path: str, *, fps: float, memory: int,
                peak_prom: float, min_gap: int, smoothing: int,
                use_patches: bool, batch_size: int, extractor) -> Tuple[List[float], float]:
    from src.sampling import sample_frames, video_meta
    from src.novelty import compute_novelty, detect_peaks

    dur, _src_fps, _n = video_meta(video_path)

    images, idxs, pts = [], [], []
    for sf in sample_frames(video_path, target_fps=fps):
        images.append(sf.image); idxs.append(sf.idx); pts.append(sf.pts_s)
    if not images:
        return [], dur

    all_vec = []
    for i in range(0, len(images), batch_size):
        embs = extractor.embed_batch(
            images[i:i + batch_size],
            idxs[i:i + batch_size],
            pts[i:i + batch_size],
        )
        for e in embs:
            all_vec.append(e.combined if use_patches else e.cls)
    E = np.stack(all_vec, axis=0).astype(np.float32)

    scores = compute_novelty(E, memory=memory)
    nv = detect_peaks(scores, min_gap=min_gap, prominence_k=peak_prom, smoothing=smoothing)
    return [float(pts[i]) for i in nv.peak_idxs.tolist()], dur


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_eval(
    labels_path: str, videos_dir: str | None, out_dir: str,
    *,
    durations_path: str | None = None,
    predictions_path: str | None = None,
    fps: float = 3.0, memory: int = 24, peak_prom: float = 2.0,
    min_gap: int = 30, smoothing: int = 3, use_patches: bool = True,
    model: str | None = None, batch_size: int = 16,
    rel_dis_grid=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
) -> dict:
    """Run prediction + score.

    Note: BBC episodes are ~3000s, far longer than GEBD 10s clips, so
    the default ``rel_dis_grid`` is tighter (0.5–20% of duration ⇒
    15s–600s tolerance). Adjust per dataset convention.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    labels = load_bbc_labels(labels_path, durations_path)
    print(f"[bbc] {len(labels)} episodes in label file")

    pred_per_video: Dict[str, List[float]] = {}
    if predictions_path and Path(predictions_path).exists():
        with open(predictions_path) as f:
            pred_per_video = json.load(f)
        print(f"[bbc] loaded cached predictions for {len(pred_per_video)} episodes")
    else:
        if not videos_dir:
            sys.exit("error: --videos required when no cached predictions exist")
        from src.features import DinoFeatureExtractor
        extractor = DinoFeatureExtractor(model_name=model)
        print(f"[bbc] model: {extractor.model_name} on {extractor.device}")

        t0 = time.time()
        for i, ep in enumerate(sorted(labels.keys()), 1):
            vpath = episode_to_video_path(videos_dir, ep)
            if vpath is None:
                print(f"  [skip] {ep}: no video file in {videos_dir}", file=sys.stderr)
                continue
            try:
                bd, dur = predict_one(
                    str(vpath), fps=fps, memory=memory,
                    peak_prom=peak_prom, min_gap=min_gap, smoothing=smoothing,
                    use_patches=use_patches, batch_size=batch_size,
                    extractor=extractor,
                )
                pred_per_video[ep] = bd
                # back-fill duration if labels lacked it
                if labels[ep].get("duration_s", 0.0) == 0.0:
                    labels[ep]["duration_s"] = dur
                print(f"  [{i}/{len(labels)}] {ep}  predicted={len(bd)}  "
                      f"elapsed={time.time()-t0:.1f}s")
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {ep}: {e}", file=sys.stderr)
                traceback.print_exc(limit=1, file=sys.stderr)

        with open(out / "predictions.json", "w") as f:
            json.dump(pred_per_video, f)

    # Eval
    gt_per_video = {ep: labels[ep]["boundaries"] for ep in pred_per_video}
    dur_per_video = {ep: labels[ep].get("duration_s", 0.0) for ep in pred_per_video}
    if any(d <= 0 for d in dur_per_video.values()):
        missing = [ep for ep, d in dur_per_video.items() if d <= 0]
        sys.exit(f"error: no duration available for episodes: {missing}. "
                 f"Add `duration_s` in the JSON, or pass --durations.")

    results = f1_grid(pred_per_video, gt_per_video, dur_per_video, rel_dis_grid)
    print()
    print_table(results)

    summary = {
        "dataset": "BBC Planet Earth",
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
    p.add_argument("--labels", required=True, help="JSON or CSV label file")
    p.add_argument("--durations", default=None, help="JSON {episode: duration_s} (CSV labels only)")
    p.add_argument("--videos", default=None, help="directory of episode videos (filename == episode_id + ext)")
    p.add_argument("--out", required=True)
    p.add_argument("--predictions", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--memory", type=int, default=24)
    p.add_argument("--peak-prom", type=float, default=2.0)
    p.add_argument("--min-gap", type=int, default=30)
    p.add_argument("--smoothing", type=int, default=3)
    p.add_argument("--no-patches", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        labels_path=args.labels,
        durations_path=args.durations,
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
    )

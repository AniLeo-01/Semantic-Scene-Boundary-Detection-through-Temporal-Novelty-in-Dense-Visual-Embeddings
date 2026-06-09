"""Run + evaluate the pipeline on hand-annotated custom videos.

You bring the videos and your boundary timestamps; this module runs the
DINOv3+novelty pipeline and scores F1@rel_dis with the same metric used
on Charades. Useful for:

  * One-off domain tests (surveillance, gameplay, manufacturing).
  * Quick sanity checks before committing to a large benchmark run.
  * Building a tiny private benchmark when no public dataset fits.

Accepted label formats
----------------------

(1) **JSON** (recommended)::

    {
      "clip_01": {"duration_s": 142.3, "boundaries": [12.5, 89.3, 110.0]},
      "clip_02": {"duration_s":  73.8, "boundaries": [22.1, 41.0, 55.4, 65.0]}
    }

    Multi-annotator (optional) — "boundaries" can also be a list of lists::

    {
      "clip_01": {"duration_s": 142.3,
                  "boundaries": [[12.5, 89.3], [12.7, 88.0]]}
    }

(2) **CSV** with one boundary per row, plus a duration column on each row
    (any row sets the video's duration; the loader uses the first
    non-zero value per video)::

    video,boundary_s,duration_s
    clip_01,12.5,142.3
    clip_01,89.3,142.3
    clip_02,22.1,73.8
    clip_02,41.0,73.8
    clip_02,55.4,73.8

(3) **Plain text** with one line per video, durations auto-detected from
    the video files via PyAV::

        # comments are ignored
        clip_01.mp4    12.5, 89.3, 110.0
        clip_02.mp4    22.1  41.0  55.4 65.0

    (Tabs/spaces/commas all work as separators; the leading token is the
    video filename relative to ``--videos``.)

Annotation tips
---------------

Easiest workflow with no extra tooling:

  1. Open the video in **VLC**. Press ``Ctrl-T`` to show the current
     timestamp.
  2. Scrub to each scene boundary, jot the timestamp down (HH:MM:SS.mmm
     is fine — the loader parses these).
  3. Fill the JSON/CSV/text file and feed it to this module.

A bit more comfortable: **MPV** with ``mpv --osd-fractions`` shows
fractional seconds in the OSD. Same workflow.

Usage
-----

::

    python -m bench.custom \\
        --labels  data/custom/labels.json \\
        --videos  data/custom/videos \\
        --out     outputs/custom_run1 \\
        --model   facebook/dinov3-vits16-pretrain-lvd1689m \\
        --fps 3 --memory 16 --peak-prom 2.0 --min-gap 6 --batch-size 64

Re-score cached predictions (sweep ``rel_dis`` grids cheaply)::

    python -m bench.custom \\
        --labels      data/custom/labels.json \\
        --predictions outputs/custom_run1/predictions.json \\
        --out         outputs/custom_rescored \\
        --eval-only
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .charades import predict_one, video_path_for
from .metrics import f1_grid, print_table


_HMS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def _parse_time(s: str) -> float | None:
    """Accept ``12.5``, ``00:12.500``, ``00:00:12.500``, or raw seconds."""
    s = s.strip()
    if not s:
        return None
    if _HMS_RE.match(s):
        h, m, sec = _HMS_RE.match(s).groups()
        return (int(h) if h else 0) * 3600 + int(m) * 60 + float(sec)
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, dict]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"unexpected JSON shape in {path}")
    out: Dict[str, dict] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            out[k] = {"duration_s": 0.0, "boundaries": v}
        elif isinstance(v, dict):
            out[k] = {
                "duration_s": float(v.get("duration_s") or v.get("duration") or 0.0),
                "boundaries": v.get("boundaries", []),
            }
    return out


def _load_csv(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video") or row.get("id") or row.get("video_id")
            if not vid:
                continue
            t = _parse_time(row.get("boundary_s") or row.get("time") or row.get("seconds") or "")
            d = _parse_time(row.get("duration_s") or row.get("duration") or "") or 0.0
            entry = out.setdefault(vid, {"duration_s": 0.0, "boundaries": []})
            if t is not None:
                entry["boundaries"].append(t)
            if d > 0 and entry["duration_s"] == 0.0:
                entry["duration_s"] = d
    for entry in out.values():
        entry["boundaries"] = sorted(entry["boundaries"])
    return out


def _load_text(path: Path, videos_dir: str | None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    fps_cache: Dict[str, Tuple[float, float]] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"[\s,]+", line)
        vid_file = parts[0]
        vid = Path(vid_file).stem
        ts = [t for t in (_parse_time(p) for p in parts[1:]) if t is not None]
        out[vid] = {"duration_s": 0.0, "boundaries": sorted(ts), "_filename": vid_file}
        if videos_dir:
            from src.sampling import video_meta
            candidate = Path(videos_dir) / vid_file
            if not candidate.exists():
                hit = video_path_for(videos_dir, vid)
                candidate = hit if hit else candidate
            if candidate.exists():
                try:
                    dur, _src_fps, _n = video_meta(str(candidate))
                    out[vid]["duration_s"] = float(dur)
                except Exception:  # noqa: BLE001
                    pass
    return out


def load_custom_labels(labels_path: str, videos_dir: str | None) -> Dict[str, dict]:
    p = Path(labels_path)
    suf = p.suffix.lower()
    if suf == ".json":
        return _load_json(p)
    if suf == ".csv":
        return _load_csv(p)
    if suf in (".txt", ".tsv"):
        return _load_text(p, videos_dir)
    sys.exit(f"error: label file extension {suf} not recognized (use .json, .csv, .txt)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(
    labels_path: str, videos_dir: str | None, out_dir: str,
    *,
    predictions_path: str | None = None,
    fps: float = 3.0, memory: int = 16, peak_prom: float = 2.0,
    min_gap: int = 6, smoothing: int = 3, use_patches: bool = True,
    model: str | None = None, batch_size: int = 16,
    rel_dis_grid=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    labels = load_custom_labels(labels_path, videos_dir)
    print(f"[custom] {len(labels)} videos in label file")

    pred_per_video: Dict[str, List[float]] = {}
    if predictions_path and Path(predictions_path).exists():
        with open(predictions_path) as f:
            pred_per_video = json.load(f)
        print(f"[custom] loaded cached predictions for {len(pred_per_video)} videos")
    else:
        if not videos_dir:
            sys.exit("error: --videos required when no cached predictions exist")
        from src.features import DinoFeatureExtractor
        extractor = DinoFeatureExtractor(model_name=model)
        print(f"[custom] model: {extractor.model_name} on {extractor.device}")

        t0 = time.time()
        for i, vid in enumerate(sorted(labels.keys()), 1):
            entry = labels[vid]
            vpath = video_path_for(videos_dir, entry.get("_filename") or vid) \
                if entry.get("_filename") else video_path_for(videos_dir, vid)
            if vpath is None:
                print(f"  [skip] {vid}: no video file in {videos_dir}", file=sys.stderr)
                continue
            try:
                bd, dur = predict_one(
                    str(vpath), fps=fps, memory=memory,
                    peak_prom=peak_prom, min_gap=min_gap, smoothing=smoothing,
                    use_patches=use_patches, batch_size=batch_size,
                    extractor=extractor,
                )
                pred_per_video[vid] = bd
                if labels[vid].get("duration_s", 0.0) <= 0:
                    labels[vid]["duration_s"] = dur
                print(f"  [{i}/{len(labels)}] {vid}  predicted={len(bd)}  "
                      f"gt={len(entry.get('boundaries', []))}  "
                      f"elapsed={time.time()-t0:.1f}s")
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {vid}: {e}", file=sys.stderr)
                traceback.print_exc(limit=1, file=sys.stderr)

        with open(out / "predictions.json", "w") as f:
            json.dump(pred_per_video, f)

    # ----- eval -----
    gt_per_video = {vid: labels[vid]["boundaries"] for vid in pred_per_video}
    dur_per_video = {vid: labels[vid].get("duration_s", 0.0) for vid in pred_per_video}
    missing_dur = [v for v, d in dur_per_video.items() if d <= 0]
    if missing_dur:
        sys.exit(
            f"error: missing duration for videos {missing_dur}.\n"
            "Add `duration_s` in the JSON/CSV, or include videos so the loader "
            "can read it from the file metadata."
        )

    results = f1_grid(pred_per_video, gt_per_video, dur_per_video, rel_dis_grid)
    print()
    print_table(results)

    summary = {
        "dataset": "custom",
        "labels_path": str(labels_path),
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
    p.add_argument("--labels", required=True, help="JSON, CSV, or TXT labels (see module docstring)")
    p.add_argument("--videos", default=None, help="directory containing the video files")
    p.add_argument("--out", required=True)
    p.add_argument("--predictions", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--memory", type=int, default=16)
    p.add_argument("--peak-prom", type=float, default=2.0)
    p.add_argument("--min-gap", type=int, default=6)
    p.add_argument("--smoothing", type=int, default=3)
    p.add_argument("--no-patches", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=16)
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
    )

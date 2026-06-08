"""Download a sample of Kinetics-GEBD val videos.

Kinetics videos live on YouTube and are not centrally hosted; the
GEBD release ships *labels* only. This helper fetches each labelled
video via yt-dlp into a local directory.

Many YouTube videos go offline over time — expect a 20–40% failure
rate on the val split. That's normal and matches what every
Kinetics benchmark deals with.

Usage:
    pip install yt-dlp
    python -m bench.fetch_kinetics_gebd \
        --labels data/gebd/k400_mr345_val_min_change_duration0.3.pkl \
        --out    data/gebd/val_videos \
        --max    200
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path


def load_labels(path: str) -> dict:
    p = Path(path)
    if p.suffix == ".json":
        return json.load(open(p))
    return pickle.load(open(p, "rb"))


def youtube_url(video_id: str) -> str:
    # Kinetics IDs are the YouTube watch IDs, sometimes with a trailing
    # "_<start>_<end>" segment marker. Strip if present.
    yt_id = video_id.split("_")[0] if len(video_id.split("_")[0]) == 11 else video_id
    return f"https://www.youtube.com/watch?v={yt_id}"


def fetch_one(video_id: str, out_dir: Path) -> bool:
    dst = out_dir / f"{video_id}.mp4"
    if dst.exists():
        return True
    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=480]+ba/b[height<=480]",
        "--merge-output-format", "mp4",
        "-o", str(dst),
        "--no-warnings",
        "--quiet",
        youtube_url(video_id),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return dst.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    if not shutil.which("yt-dlp"):
        sys.exit("yt-dlp not on PATH. pip install yt-dlp.")

    labels = load_labels(args.labels)
    vids = sorted(labels.keys())
    if args.max:
        vids = vids[:args.max]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ok = miss = 0
    for i, vid in enumerate(vids, 1):
        success = fetch_one(vid, out)
        ok += int(success); miss += int(not success)
        print(f"[{i}/{len(vids)}] {vid}  ok={ok}  miss={miss}", end="\r")
    print()
    print(f"done: {ok}/{len(vids)} downloaded, {miss} missing")


if __name__ == "__main__":
    main()

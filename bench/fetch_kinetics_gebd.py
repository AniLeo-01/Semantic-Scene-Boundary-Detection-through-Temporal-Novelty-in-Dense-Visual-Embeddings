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
        return pickle.load(f)


def parse_video_id(video_id: str) -> tuple[str, float | None, float | None]:
    """Parse a Kinetics video_id into (youtube_id, start_s, end_s).

    Kinetics IDs are typically the YouTube watch ID (11 chars). Some
    exports glue on ``_<start>_<end>`` segment markers like
    ``abcdefgABCD_000010_000020`` — interpret those as a 6-digit
    zero-padded second-offset pair when present.
    """
    parts = video_id.split("_")
    if len(parts) >= 3 and len(parts[0]) == 11 and parts[-1].isdigit() and parts[-2].isdigit():
        return parts[0], float(parts[-2]), float(parts[-1])
    if len(parts[0]) == 11:
        return parts[0], None, None
    return video_id, None, None


def youtube_url(yt_id: str) -> str:
    return f"https://www.youtube.com/watch?v={yt_id}"


def fetch_one(video_id: str, out_dir: Path,
              start_s: float | None = None, end_s: float | None = None) -> bool:
    """Download a single clip. If ``start_s``/``end_s`` are provided,
    cut to that range so the local file's timeline matches GEBD's GT.
    """
    dst = out_dir / f"{video_id}.mp4"
    if dst.exists():
        return True
    yt_id, ps, pe = parse_video_id(video_id)
    start_s = start_s if start_s is not None else ps
    end_s = end_s if end_s is not None else pe

    cmd = [
        "yt-dlp",
        "-f", "bv*[height<=480]+ba/b[height<=480]",
        "--merge-output-format", "mp4",
        "-o", str(dst),
        "--no-warnings",
        "--quiet",
    ]
    if start_s is not None and end_s is not None and end_s > start_s:
        # Range cut — yt-dlp will keyframe-align; close enough for GEBD.
        cmd += ["--download-sections", f"*{start_s:.2f}-{end_s:.2f}",
                "--force-keyframes-at-cuts"]
    cmd.append(youtube_url(yt_id))

    try:
        subprocess.run(cmd, check=True, timeout=180,
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
        # Prefer start/end from the label entry — most reliable source.
        entry = labels[vid] if isinstance(labels[vid], dict) else {}
        start_s = entry.get("start_time") or entry.get("start_s")
        end_s = entry.get("end_time") or entry.get("end_s")
        if start_s is None and "video_duration" in entry:
            # Fall back to parsing from the id; many K400 ids encode it.
            _, ps, pe = parse_video_id(vid)
            start_s, end_s = ps, pe
        success = fetch_one(vid, out, start_s=start_s, end_s=end_s)
        ok += int(success); miss += int(not success)
        print(f"[{i}/{len(vids)}] {vid}  ok={ok}  miss={miss}", end="\r")
    print()
    print(f"done: {ok}/{len(vids)} downloaded, {miss} missing")


if __name__ == "__main__":
    main()

"""Download the Charades dataset (annotations + videos).

Charades:
  Sigurdsson et al., "Hollywood in Homes", ECCV 2016.
  Project page: https://prior.allenai.org/projects/charades
  License: non-commercial use; see project page for terms.

Mirrors (AI2 public S3):
  * Annotations (~3 MB):      Charades.zip
  * Videos 480p (~13 GB zip): Charades_v1_480.zip   (recommended)
  * Videos original (~55 GB): Charades_v1.zip       (full quality)

The 480p subset is plenty for our pipeline (the DINOv3 vit-S/16
processor downscales to 224x224 anyway), so we default to it.

Usage:
    python -m bench.fetch_charades --out data/charades                # everything
    python -m bench.fetch_charades --out data/charades --annotations  # labels only
    python -m bench.fetch_charades --out data/charades --hi-res       # full quality
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

S3 = "https://ai2-public-datasets.s3.amazonaws.com/charades"

ANNOTATIONS_URL = f"{S3}/Charades.zip"
VIDEOS_480_URL  = f"{S3}/Charades_v1_480.zip"
VIDEOS_HD_URL   = f"{S3}/Charades_v1.zip"


def download(url: str, dst: Path) -> bool:
    """Stream a single URL to disk with a progress bar."""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [cached] {dst.name}")
        return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=600) as r:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            last_pct = -1
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = int(100 * done / total)
                        if pct != last_pct and pct % 2 == 0:
                            mb = done / 1e6
                            tot = total / 1e6
                            sys.stderr.write(f"\r  {dst.name}: {pct:3d}% ({mb:.0f}/{tot:.0f} MB)")
                            sys.stderr.flush()
                            last_pct = pct
        sys.stderr.write("\n")
        tmp.rename(dst)
        return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"\n  FAILED {url}: {e}\n")
        if tmp.exists():
            tmp.unlink()
        return False


def extract(zip_path: Path, dst_dir: Path) -> bool:
    print(f"  extracting {zip_path.name} -> {dst_dir}")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dst_dir)
        return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"\n  FAILED to extract {zip_path}: {e}\n")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotations", action="store_true",
                    help="only download annotations CSV (no videos)")
    ap.add_argument("--videos-only", action="store_true",
                    help="only download the videos zip (no annotations)")
    ap.add_argument("--hi-res", action="store_true",
                    help="download full-quality videos (~55 GB) instead of 480p (~13 GB)")
    ap.add_argument("--keep-zips", action="store_true",
                    help="don't delete the .zip files after extraction")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    do_ann = not args.videos_only
    do_vid = not args.annotations

    ok = miss = 0

    if do_ann:
        print(f"[charades] downloading annotations -> {out}", file=sys.stderr)
        ann_zip = out / "Charades.zip"
        if download(ANNOTATIONS_URL, ann_zip) and extract(ann_zip, out):
            ok += 1
            if not args.keep_zips:
                ann_zip.unlink()
        else:
            miss += 1

    if do_vid:
        url = VIDEOS_HD_URL if args.hi_res else VIDEOS_480_URL
        zip_name = "Charades_v1.zip" if args.hi_res else "Charades_v1_480.zip"
        print(f"[charades] downloading videos -> {out}", file=sys.stderr)
        vid_zip = out / zip_name
        if download(url, vid_zip) and extract(vid_zip, out):
            ok += 1
            if not args.keep_zips:
                vid_zip.unlink()
        else:
            miss += 1

    print(f"\ndone: {ok} step(s) ok, {miss} failed")
    if do_vid:
        sub = "Charades_v1" if args.hi_res else "Charades_v1_480"
        print(f"videos:      {out / sub}")
    if do_ann:
        print(f"annotations: {out / 'Charades'}")
    print()
    print("Next: run the bench")
    print("  python -m bench.charades \\")
    print(f"    --annotations {out}/Charades/Charades_v1_test.csv \\")
    print(f"    --videos      {out}/{'Charades_v1' if args.hi_res else 'Charades_v1_480'} \\")
    print("    --out         outputs/charades_run1 \\")
    print("    --model       facebook/dinov3-vits16-pretrain-lvd1689m \\")
    print("    --fps 5 --memory 12 --peak-prom 1.8 --min-gap 3 --batch-size 64")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Sweep memory / peak_prom / patch-novelty without re-embedding videos.

The expensive part of a bench run is feature extraction. ``--save-embeddings``
on the underlying runners now writes ``embeddings.npz`` per run; this
script reuses those cached embeddings to re-run novelty + peak detection
across a grid of configurations in seconds.

Workflow
--------

1. Once: run the full bench with ``--save-cache``::

       python -m bench.sweep cache \\
           --dataset     charades \\
           --annotations data/charades/Charades/Charades_v1_test.csv \\
           --videos      data/charades/Charades_v1_480 \\
           --cache       outputs/charades_cache \\
           --model       facebook/dinov2-small \\
           --fps 5 --batch-size 64 --max-videos 500

   This writes one ``<video_id>.npz`` per video containing
   ``{cls, combined, patches, pts_s, duration_s}``.

2. Then: sweep any post-embedding knob cheaply::

       python -m bench.sweep run \\
           --dataset     charades \\
           --annotations data/charades/Charades/Charades_v1_test.csv \\
           --cache       outputs/charades_cache \\
           --out         outputs/charades_sweep \\
           --memory     6 12 24 48 \\
           --peak-prom  1.5 1.8 2.0 2.5 \\
           --score-modes pooled patch_mean patch_topk
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .metrics import f1_grid, print_table


# ---------------------------------------------------------------------------
# Step 1 — cache embeddings
# ---------------------------------------------------------------------------

def cache_embeddings(
    dataset: str,
    *,
    annotations: str | None = None,
    labels: str | None = None,
    videos_dir: str,
    cache_dir: str,
    fps: float = 5.0, model: str | None = None, batch_size: int = 64,
    max_videos: int | None = None,
) -> None:
    """Run feature extraction once and dump per-video npz files."""
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)

    if dataset == "charades":
        from .charades import load_charades_labels, video_path_for as _vp
        labels_map = load_charades_labels(annotations)
        path_for = lambda vid, entry: _vp(videos_dir, vid)
    elif dataset == "custom":
        from .custom import load_custom_labels
        from .charades import video_path_for as _vp
        labels_map = load_custom_labels(labels, videos_dir)
        def path_for(vid, entry):
            fname = entry.get("_filename")
            return _vp(videos_dir, fname or vid)
    else:
        sys.exit(f"unknown dataset {dataset!r}")

    from src.features import DinoFeatureExtractor
    from src.sampling import sample_frames, video_meta
    extractor = DinoFeatureExtractor(model_name=model)
    print(f"[cache] model: {extractor.model_name} on {extractor.device}")

    vids = sorted(labels_map.keys())
    if max_videos:
        vids = vids[:max_videos]

    t0 = time.time()
    for i, vid in enumerate(vids, 1):
        out_path = cache / f"{vid}.npz"
        if out_path.exists():
            continue
        entry = labels_map[vid]
        vpath = path_for(vid, entry)
        if vpath is None or not Path(vpath).exists():
            continue
        try:
            dur, _src_fps, _n = video_meta(str(vpath))
            images, idxs, pts = [], [], []
            for sf in sample_frames(str(vpath), target_fps=fps):
                images.append(sf.image); idxs.append(sf.idx); pts.append(sf.pts_s)
            if not images:
                continue
            cls_list, comb_list, patch_list = [], [], []
            for j in range(0, len(images), batch_size):
                embs = extractor.embed_batch(
                    images[j:j + batch_size],
                    idxs[j:j + batch_size],
                    pts[j:j + batch_size],
                )
                for e in embs:
                    cls_list.append(e.cls); comb_list.append(e.combined); patch_list.append(e.patches)
            np.savez_compressed(
                out_path,
                cls=np.stack(cls_list).astype(np.float32),
                combined=np.stack(comb_list).astype(np.float32),
                patches=np.stack(patch_list).astype(np.float32),
                pts_s=np.asarray(pts, dtype=np.float64),
                duration_s=np.float64(dur),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {vid}: {e}", file=sys.stderr)
        if i % 25 == 0:
            dt = time.time() - t0
            print(f"  [{i}/{len(vids)}] elapsed={dt:.1f}s ({i/dt:.2f} vid/s)")
    print(f"[cache] done -> {cache}")


# ---------------------------------------------------------------------------
# Step 2 — sweep
# ---------------------------------------------------------------------------

SCORE_MODES = ("pooled", "cls", "patch_mean", "patch_topk", "patch_min")


def _score(npz: dict, mode: str, memory: int) -> np.ndarray:
    from src.novelty import compute_novelty, compute_patch_novelty
    if mode == "pooled":
        E = npz["combined"]
    elif mode == "cls":
        E = npz["cls"]
    else:
        agg = mode.split("_", 1)[1]  # "mean" | "topk" | "min"
        return compute_patch_novelty(npz["patches"], memory=memory, agg=agg)
    return compute_novelty(E, memory=memory)


def sweep(
    dataset: str,
    *,
    cache_dir: str,
    out_dir: str,
    annotations: str | None = None,
    labels: str | None = None,
    memory_list: Sequence[int] = (12,),
    peak_prom_list: Sequence[float] = (1.8,),
    min_gap: int = 3,
    smoothing: int = 1,
    score_modes: Sequence[str] = ("pooled",),
    rel_dis_grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict:
    from src.novelty import detect_peaks

    cache = Path(cache_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    if dataset == "charades":
        from .charades import load_charades_labels
        labels_map = load_charades_labels(annotations)
    elif dataset == "custom":
        from .custom import load_custom_labels
        labels_map = load_custom_labels(labels, None)
    else:
        sys.exit(f"unknown dataset {dataset!r}")

    # Load every cache file once into memory.
    cached: Dict[str, dict] = {}
    for npz_file in sorted(cache.glob("*.npz")):
        vid = npz_file.stem
        if vid not in labels_map:
            continue
        with np.load(npz_file) as data:
            cached[vid] = {k: data[k] for k in data.files}
    print(f"[sweep] loaded cache for {len(cached)} videos")

    all_results: List[dict] = []
    for mode, memory, peak_prom in product(score_modes, memory_list, peak_prom_list):
        if mode not in SCORE_MODES:
            sys.exit(f"unknown score mode {mode!r}; pick from {SCORE_MODES}")
        pred_per_video: Dict[str, List[float]] = {}
        for vid, npz in cached.items():
            try:
                scores = _score(npz, mode, memory=memory)
                nv = detect_peaks(scores, min_gap=min_gap,
                                  prominence_k=peak_prom, smoothing=smoothing)
                pts_s = npz["pts_s"]
                pred_per_video[vid] = [float(pts_s[i]) for i in nv.peak_idxs.tolist()]
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {vid}: {e}", file=sys.stderr)

        nonempty = [v for v in pred_per_video if labels_map[v].get("boundaries")]
        gt = {v: labels_map[v]["boundaries"] for v in nonempty}
        dur = {v: float(cached[v].get("duration_s", labels_map[v].get("duration_s", 0.0)))
               for v in nonempty}
        results = f1_grid({v: pred_per_video[v] for v in nonempty}, gt, dur, rel_dis_grid)
        f1_05 = results[0]["f1"]
        f1_10 = results[1]["f1"] if len(results) > 1 else float("nan")
        print(f"  mode={mode:10s}  memory={memory:>3d}  k={peak_prom:>4.2f}  "
              f"F1@0.05={f1_05:.4f}  F1@0.10={f1_10:.4f}  N={len(nonempty)}")
        all_results.append({
            "mode": mode, "memory": memory, "peak_prom": peak_prom,
            "min_gap": min_gap, "smoothing": smoothing,
            "metrics": results,
        })

    out_path = out / "sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[sweep] {len(all_results)} configs written -> {out_path}")

    # Pretty-print best per mode
    print()
    print("Best F1@0.05 per score mode:")
    for mode in score_modes:
        sub = [r for r in all_results if r["mode"] == mode]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["metrics"][0]["f1"])
        m = best["metrics"][0]
        print(f"  {mode:10s}  memory={best['memory']:>3d}  k={best['peak_prom']:.2f}  "
              f"F1@0.05={m['f1']:.4f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    return {"runs": all_results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("cache", help="extract features once and write per-video npz")
    pc.add_argument("--dataset", required=True, choices=["charades", "custom"])
    pc.add_argument("--annotations", default=None)
    pc.add_argument("--labels", default=None)
    pc.add_argument("--videos", required=True)
    pc.add_argument("--cache", required=True)
    pc.add_argument("--fps", type=float, default=5.0)
    pc.add_argument("--model", default=None)
    pc.add_argument("--batch-size", type=int, default=64)
    pc.add_argument("--max-videos", type=int, default=None)

    pr = sub.add_parser("run", help="sweep memory / prominence / score mode using a cache")
    pr.add_argument("--dataset", required=True, choices=["charades", "custom"])
    pr.add_argument("--annotations", default=None)
    pr.add_argument("--labels", default=None)
    pr.add_argument("--cache", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--memory", type=int, nargs="+", default=[12])
    pr.add_argument("--peak-prom", type=float, nargs="+", default=[1.8])
    pr.add_argument("--min-gap", type=int, default=3)
    pr.add_argument("--smoothing", type=int, default=1)
    pr.add_argument("--score-modes", nargs="+", default=["pooled"],
                    choices=list(SCORE_MODES))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cmd == "cache":
        cache_embeddings(
            dataset=args.dataset,
            annotations=args.annotations, labels=args.labels,
            videos_dir=args.videos, cache_dir=args.cache,
            fps=args.fps, model=args.model, batch_size=args.batch_size,
            max_videos=args.max_videos,
        )
    else:
        sweep(
            dataset=args.dataset,
            cache_dir=args.cache, out_dir=args.out,
            annotations=args.annotations, labels=args.labels,
            memory_list=args.memory, peak_prom_list=args.peak_prom,
            min_gap=args.min_gap, smoothing=args.smoothing,
            score_modes=args.score_modes,
        )

"""End-to-end pipeline: video -> scene boundaries + keyframes + novelty plot.

Usage:
    python -m src.main --video sample_videos/clip.mp4 --out outputs/run1 --fps 3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .features import DinoFeatureExtractor
from .keyframes import select_keyframes
from .novelty import compute_novelty, detect_peaks, peaks_to_segments
from .sampling import sample_frames, video_meta
from .viewer import write_viewer
from .viz import plot_novelty


def run(
    video_path: str,
    out_dir: str,
    fps: float = 3.0,
    model: str | None = None,
    batch_size: int = 8,
    memory: int = 16,
    peak_prom: float = 2.0,
    min_gap: int = 8,
    use_patches: bool = True,
    keyframe_method: str = "centroid",
    smoothing: int = 3,
    save_embeddings: bool = False,
    copy_video: bool = False,
    video_url: str | None = None,
) -> dict:
    out = Path(out_dir)
    (out / "keyframes").mkdir(parents=True, exist_ok=True)

    dur, src_fps, n = video_meta(video_path)
    print(f"[video] duration={dur:.1f}s src_fps={src_fps:.2f} frames={n}")

    extractor = DinoFeatureExtractor(model_name=model)
    print(f"[model] {extractor.model_name} on {extractor.device}")

    # 1) sample + 2) embed (streamed, batched)
    images_buf: list = []
    idxs_buf: list = []
    pts_buf: list = []
    raw_frames: list = []   # keep small thumbnails for keyframe output

    all_embeds: list = []
    all_pts: list = []
    all_idxs: list = []

    def flush():
        if not images_buf:
            return
        embs = extractor.embed_batch(images_buf, idxs_buf, pts_buf)
        for e in embs:
            vec = e.combined if use_patches else e.cls
            all_embeds.append(vec)
            all_pts.append(e.pts_s)
            all_idxs.append(e.idx)
        images_buf.clear()
        idxs_buf.clear()
        pts_buf.clear()

    pbar = tqdm(desc="frames", unit="f")
    for sf in sample_frames(video_path, target_fps=fps):
        images_buf.append(sf.image)
        idxs_buf.append(sf.idx)
        pts_buf.append(sf.pts_s)
        raw_frames.append(sf.image)
        if len(images_buf) >= batch_size:
            flush()
        pbar.update(1)
    flush()
    pbar.close()

    if not all_embeds:
        raise RuntimeError("No frames decoded from video.")

    embeddings = np.stack(all_embeds, axis=0).astype(np.float32)  # (T, D)
    pts_s = np.asarray(all_pts, dtype=np.float64)
    print(f"[embed] T={embeddings.shape[0]} D={embeddings.shape[1]}")

    if save_embeddings:
        np.savez(out / "embeddings.npz", embeddings=embeddings, pts_s=pts_s)

    # 3) novelty + 4) peaks
    scores = compute_novelty(embeddings, memory=memory)
    nv = detect_peaks(scores, min_gap=min_gap, prominence_k=peak_prom, smoothing=smoothing)
    print(
        f"[novelty] height_floor={nv.threshold * 0.6:.4f} "
        f"prominence={nv.prominence:.4f} peaks={len(nv.peak_idxs)}"
    )

    # 5) segments + keyframes
    segs = peaks_to_segments(nv.peak_idxs, n_frames=len(scores))
    scenes = select_keyframes(segs, embeddings, pts_s, nv.scores, method=keyframe_method)

    # 6) save keyframes + JSON
    for s in scenes:
        img = raw_frames[s.keyframe_idx]
        Image.fromarray(img).save(out / "keyframes" / f"scene_{s.scene_idx:03d}.jpg", quality=88)

    boundaries = [asdict(s) for s in scenes]
    summary = {
        "video": str(video_path),
        "model": extractor.model_name,
        "fps_sampled": fps,
        "n_frames_sampled": int(len(scores)),
        "memory": memory,
        "peak_prom": peak_prom,
        "min_gap": min_gap,
        "use_patches": use_patches,
        "threshold": nv.threshold,
        "prominence": nv.prominence,
        "n_scenes": len(scenes),
        "scenes": boundaries,
    }
    with open(out / "boundaries.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 7) plot
    plot_novelty(
        nv.scores,
        nv.peak_idxs,
        nv.threshold,
        pts_s,
        str(out / "novelty.png"),
        height_floor=nv.threshold * 0.6,
        prominence=nv.prominence,
    )

    # 8) interactive viewer (self-contained HTML)
    viewer_path = write_viewer(
        out_dir=out,
        video_path=video_path,
        novelty=nv.scores.tolist(),
        pts_s=pts_s.tolist(),
        peak_idxs=nv.peak_idxs.tolist(),
        threshold=nv.threshold,
        prominence=nv.prominence,
        scenes=boundaries,
        duration_s=dur,
        model_name=extractor.model_name,
        fps_sampled=fps,
        copy_video=copy_video,
        video_url=video_url,
    )
    print(f"[viewer] open in a browser: {viewer_path}")

    print(f"[done] {len(scenes)} scenes -> {out}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--model", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--memory", type=int, default=16)
    p.add_argument("--peak-prom", type=float, default=2.0)
    p.add_argument("--min-gap", type=int, default=8)
    p.add_argument("--no-patches", action="store_true", help="use CLS-only embedding")
    p.add_argument("--keyframe-method", choices=["centroid", "peak"], default="centroid")
    p.add_argument("--smoothing", type=int, default=3)
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument(
        "--copy-video",
        action="store_true",
        help="copy source video into output dir so viewer.html references it by filename",
    )
    p.add_argument(
        "--video-url",
        default=None,
        help="explicit src for the <video> tag (overrides --copy-video and the default relative path)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        video_path=args.video,
        out_dir=args.out,
        fps=args.fps,
        model=args.model,
        batch_size=args.batch_size,
        memory=args.memory,
        peak_prom=args.peak_prom,
        min_gap=args.min_gap,
        use_patches=not args.no_patches,
        keyframe_method=args.keyframe_method,
        smoothing=args.smoothing,
        save_embeddings=args.save_embeddings,
        copy_video=args.copy_video,
        video_url=args.video_url,
    )

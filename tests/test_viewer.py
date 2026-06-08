"""Smoke test: viewer HTML generation with synthetic data."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.viewer import build_viewer_html, write_viewer  # noqa: E402


def _fake_run():
    rng = np.random.default_rng(0)
    T = 100
    pts = (np.arange(T) / 3.0).tolist()
    nov = (0.05 + 0.02 * rng.standard_normal(T)).clip(min=0).tolist()
    for p in (33, 66):
        nov[p] = 0.4
    peaks = [33, 66]
    scenes = [
        {"scene_idx": 0, "start_idx": 0,  "end_idx": 32, "start_s": 0.0,    "end_s": 10.7,
         "keyframe_idx": 10, "novelty_peak": 0.08},
        {"scene_idx": 1, "start_idx": 33, "end_idx": 65, "start_s": 11.0,   "end_s": 21.7,
         "keyframe_idx": 50, "novelty_peak": 0.40},
        {"scene_idx": 2, "start_idx": 66, "end_idx": 99, "start_s": 22.0,   "end_s": 33.0,
         "keyframe_idx": 80, "novelty_peak": 0.40},
    ]
    return pts, nov, peaks, scenes


def test_build_viewer_html_well_formed():
    pts, nov, peaks, scenes = _fake_run()
    kfs = [f"keyframes/scene_{s['scene_idx']:03d}.jpg" for s in scenes]
    html = build_viewer_html(
        video_path="/tmp/foo/clip.mp4",
        video_relpath="../clip.mp4",
        novelty=nov, pts_s=pts, peak_idxs=peaks,
        threshold=0.12, prominence=0.06,
        scenes=scenes, keyframe_relpaths=kfs,
        duration_s=33.0,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        fps_sampled=3.0,
    )

    # Structural checks
    assert "<!doctype html>" in html.lower()
    assert "<canvas id=\"chart\">" in html
    assert "<video id=\"vid\"" in html
    assert "../clip.mp4" in html
    assert "scene 000" in html and "scene 002" in html
    # No unsubstituted template tokens
    for token in ["__VIDEO_NAME__", "__VIDEO_REL__", "__META_LINE__",
                  "__N_SCENES__", "__KEYFRAME_CARDS__", "__DATA_JSON__"]:
        assert token not in html, f"unsubstituted token: {token}"

    # Embedded DATA JSON parses
    m = re.search(r"const DATA = (\{.*?\});\s*\nconst vid", html, flags=re.S)
    assert m, "could not find embedded DATA literal"
    data = json.loads(m.group(1))
    assert len(data["novelty"]) == len(nov)
    assert data["peak_idxs"] == peaks
    assert data["height_floor"] == 0.12 * 0.6
    assert len(data["scenes"]) == 3
    print(f"HTML length: {len(html)} chars; DATA novelty len: {len(data['novelty'])}")


def test_write_viewer_creates_file():
    pts, nov, peaks, scenes = _fake_run()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run1"
        (out / "keyframes").mkdir(parents=True)
        path = write_viewer(
            out_dir=out,
            video_path=str(Path(td) / "clip.mp4"),
            novelty=nov, pts_s=pts, peak_idxs=peaks,
            threshold=0.12, prominence=0.06,
            scenes=scenes,
            duration_s=33.0,
            model_name="facebook/dinov2-small",
            fps_sampled=3.0,
        )
        assert path.exists(), path
        content = path.read_text()
        assert "clip.mp4" in content
        print(f"viewer.html written: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    test_build_viewer_html_well_formed()
    test_write_viewer_creates_file()
    print("OK")

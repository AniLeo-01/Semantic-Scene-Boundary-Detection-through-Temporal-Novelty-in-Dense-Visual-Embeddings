# Semantic Scene Boundary Detection

DINOv3-based scene boundary detector. Self-supervised, training-free,
single-pass.

For the research narrative — motivation, hypothesis, theory, glossary,
limitations — see **[THESIS.md](THESIS.md)**. This file is the technical
reference: install, run, configure, extend.

---

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [CLI reference](#cli-reference)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)
- [Module reference](#module-reference)
- [Programmatic use](#programmatic-use)
- [Tests](#tests)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Extending the pipeline](#extending-the-pipeline)

---

## Requirements

- Python 3.10+
- ~2 GB disk for model weights (DINOv3 ViT-S/16 ≈ 85 MB; falls back to
  DINOv2-small if v3 weights are gated)
- CPU works; GPU (CUDA) recommended for videos > 1 minute
- `ffmpeg` available on PATH (used by PyAV for video decode)

Python deps are pinned in `requirements.txt`.

## Install

```bash
git clone <your-fork-url> scene-boundary
cd scene-boundary
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads weights from HuggingFace Hub. If the DINOv3 repo is
gated for your account, the loader silently falls back to DINOv2-small —
the same interface, no code change.

To force a specific backbone:

```bash
python -m src.main --video clip.mp4 --out outputs/run1 \
  --model facebook/dinov2-base
```

## Run

```bash
python -m src.main --video sample.mp4 --out outputs/run1 --fps 3
```

That's it. The command prints the model name, the number of sampled
frames, the adaptive threshold, the peak count, and writes everything to
`outputs/run1/`.

## CLI reference

`python -m src.main [flags]`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--video` | path | **required** | Input video. Any FFmpeg-decodable format. |
| `--out` | path | **required** | Output directory. Created if absent. |
| `--fps` | float | `3.0` | Target sampling rate. Lower = faster, may miss short scenes. |
| `--model` | str | DINOv3 ViT-S/16 | Any HF vision model exposing `last_hidden_state`. |
| `--batch-size` | int | `8` | Embedding batch size. Raise on GPU; lower on CPU. |
| `--memory` | int | `16` | Memory bank length, in sampled frames. |
| `--peak-prom` | float | `2.0` | Sensitivity `k`. Sets the prominence requirement `k · 1.4826 · MAD` and the absolute height floor `0.6 · (median + k · 1.4826 · MAD)`. Lower = more peaks. |
| `--min-gap` | int | `8` | Minimum sampled-frame distance between detected boundaries. |
| `--no-patches` | flag | off | Use CLS-only embedding (skip patch-mean). |
| `--keyframe-method` | str | `centroid` | `centroid` or `peak`. |
| `--smoothing` | int | `3` | Length of moving-average filter on novelty signal. |
| `--save-embeddings` | flag | off | Cache `(T, D)` embeddings to `outputs/<run>/embeddings.npz`. |

## Outputs

A run writes:

```
outputs/run1/
├── boundaries.json          # full result + config
├── novelty.png              # static signal plot with peaks
├── viewer.html              # interactive viewer (open in any browser)
├── keyframes/
│   ├── scene_000.jpg
│   ├── scene_001.jpg
│   └── ...
└── embeddings.npz           # only with --save-embeddings
```

### `viewer.html` — interactive visualization

A self-contained, dependency-free HTML page that ties the three outputs
together against the source video:

- the video plays on the left;
- the novelty signal is rendered live underneath it (canvas), with the
  height floor drawn, detected peaks shown as red verticals, and a
  yellow **playhead cursor** that tracks `video.currentTime`;
- the right column is a scrollable list of keyframe thumbnails — the
  card for the currently-playing scene is highlighted and auto-scrolls
  into view.

Everything is anchored to the video timeline:

- click anywhere on the chart → seeks the video to that timestamp;
- click any keyframe card → jumps the video to the start of that scene;
- playing the video → the chart cursor and active keyframe update in
  real time.

The page inlines all of the data it needs (novelty samples, peak indices,
scene metadata, thresholds), so it works directly from `file://` without
a web server. The video is referenced by a relative path from the output
directory and the keyframes by relative paths into `keyframes/`. To view
the file on a different machine, copy the entire `outputs/run1/` folder
**plus the source video** while preserving the relative path.

### `boundaries.json` schema

```json
{
  "video": "sample.mp4",
  "model": "facebook/dinov3-vits16-pretrain-lvd1689m",
  "fps_sampled": 3.0,
  "n_frames_sampled": 540,
  "memory": 16,
  "peak_prom": 2.0,
  "min_gap": 8,
  "use_patches": true,
  "threshold": 0.0843,
  "prominence": 0.0421,
  "n_scenes": 7,
  "scenes": [
    {
      "scene_idx": 0,
      "start_idx": 0,
      "end_idx": 87,
      "start_s": 0.0,
      "end_s": 29.0,
      "keyframe_idx": 42,
      "novelty_peak": 0.061
    }
  ]
}
```

`start_idx` / `end_idx` are indices into the **sampled** frame stream
(not the original). `start_s` / `end_s` are seconds in the source video's
timeline.

## Repository layout

```
.
├── README.md          ← this file (technical doc)
├── THESIS.md          ← research narrative
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── sampling.py    ← PyAV frame sampling
│   ├── features.py    ← DINOv3 feature extractor
│   ├── novelty.py     ← memory bank + scoring + peak detection
│   ├── keyframes.py   ← centroid/peak keyframe selection
│   ├── viz.py         ← static novelty plot
│   ├── viewer.py      ← interactive viewer.html generator
│   └── main.py        ← CLI + end-to-end orchestrator
└── tests/
    ├── test_novelty.py
    └── test_viewer.py
```

## Module reference

### `src.sampling`

```python
sample_frames(video_path: str, target_fps: float) -> Iterator[SampledFrame]
video_meta(video_path: str) -> (duration_s, src_fps, total_frames)
```

`SampledFrame(idx, pts_s, image)` — `image` is `HxWx3 uint8` RGB.

Decodes every frame sequentially and drops in between to hit the target
FPS. Reliable for short videos. For very long videos, switch to keyframe
decoding by setting `stream.codec_context.skip_frame = "NONKEY"` (one
line change).

### `src.features`

```python
DinoFeatureExtractor(model_name=None, device=None, dtype=torch.float32)
  .embed_batch(images, idxs, pts) -> List[FrameEmbedding]
```

`FrameEmbedding(idx, pts_s, cls, patch_mean, combined)`:

- `cls` — L2-normalized [CLS] token output, `(D,)`
- `patch_mean` — L2-normalized mean of patch tokens, `(D,)`
- `combined` — L2-normalized concat `[cls, patch_mean]`, `(2D,)`

Model name defaults to DINOv3 with DINOv2 fallback. Any HF model whose
forward exposes `last_hidden_state` of shape `(B, 1+N, D)` works.

### `src.novelty`

```python
compute_novelty(embeddings: ndarray[T,D], memory=16, warmup=4) -> ndarray[T]
detect_peaks(scores, min_gap=8, prominence_k=2.0, smoothing=3,
             height_floor_ratio=0.6) -> NoveltyResult
peaks_to_segments(peak_idxs, n_frames) -> List[(start, end)]
```

`NoveltyResult(scores, peak_idxs, threshold, prominence)`. Embeddings must be
L2-normalized.

- Novelty: `1 − max_k cos(e_t, m_k)`.
- A candidate peak is accepted iff **both**:
  1. its value clears the absolute floor `height_floor_ratio · (median + k · 1.4826 · MAD)`, and
  2. it has at least `k · 1.4826 · MAD` *prominence* — i.e. it stands out by that much
     from its local surroundings (SciPy's definition of prominence).
- The prominence test catches real but modest peaks that the height-only rule misses
  when the global baseline is noisy. The loose height floor still rejects noise spikes
  near zero so the prominence rule isn't applied to garbage.

### `src.keyframes`

```python
select_keyframes(segments, embeddings, pts_s, scores, method="centroid")
  -> List[Scene]
```

`method="peak"` picks the highest-novelty frame in the segment instead of
the centroid-closest.

### `src.viz`

```python
plot_novelty(scores, peak_idxs, threshold, pts_s, out_path,
             height_floor=None, prominence=None) -> None
```

Writes a single PNG. When `height_floor` is supplied, it is drawn as the
dashed reference line (since the detector's effective absolute cut is the
height floor, not the raw threshold); `prominence` is reported in the
legend so the detector state is readable from the plot.

### `src.viewer`

```python
build_viewer_html(*, video_path, video_relpath, novelty, pts_s,
                  peak_idxs, threshold, prominence, scenes,
                  keyframe_relpaths, duration_s, model_name,
                  fps_sampled) -> str
write_viewer(out_dir, video_path, novelty, pts_s, peak_idxs,
             threshold, prominence, scenes, duration_s, model_name,
             fps_sampled) -> Path
```

Builds the interactive `viewer.html`. `build_viewer_html` returns the
HTML string; `write_viewer` writes it into `out_dir/viewer.html` and
returns the path. All numeric data (novelty samples, peak indices, scene
metadata, thresholds) is inlined as a JSON literal, so the page works
over `file://` with no fetch / CORS issues.

### `src.main`

```python
run(video_path, out_dir, fps=3.0, model=None, batch_size=8, memory=16,
    peak_prom=2.0, min_gap=8, use_patches=True,
    keyframe_method="centroid", smoothing=3, save_embeddings=False) -> dict
```

End-to-end pipeline. Returns the summary dict that is also written to
`boundaries.json`.

## Programmatic use

```python
from src.main import run

summary = run(
    video_path="sample.mp4",
    out_dir="outputs/run1",
    fps=3.0,
    memory=16,
    peak_prom=2.0,
)
for scene in summary["scenes"]:
    print(scene["start_s"], scene["end_s"], scene["keyframe_idx"])
```

Or wire the stages together yourself if you want a custom pipeline:

```python
import numpy as np
from src.sampling import sample_frames
from src.features import DinoFeatureExtractor
from src.novelty import compute_novelty, detect_peaks, peaks_to_segments
from src.keyframes import select_keyframes

ext = DinoFeatureExtractor()
frames = list(sample_frames("sample.mp4", target_fps=3.0))
images = [f.image for f in frames]
idxs = [f.idx for f in frames]
pts = [f.pts_s for f in frames]

embs = ext.embed_batch(images, idxs, pts)
E = np.stack([e.combined for e in embs])

scores = compute_novelty(E, memory=16)
nv = detect_peaks(scores, min_gap=8, prominence_k=2.0)
segments = peaks_to_segments(nv.peak_idxs, n_frames=len(scores))
scenes = select_keyframes(segments, E, np.array(pts), nv.scores)
```

## Tests

```bash
python tests/test_novelty.py
```

The test fabricates three 30-frame "scenes" as tight Gaussian clusters
around three random unit vectors and asserts that:

- the detector finds ≥ 2 boundaries,
- the boundaries land within ±3 frames of the true 30 and 60,
- segmentation yields ≥ 3 scenes,
- keyframes land inside their respective segments.

It exercises the algorithmic core (`novelty.py`, `keyframes.py`) without
requiring `torch`, `transformers`, or `av`. Useful as a fast regression
gate when iterating on the scoring / threshold logic.

## Performance notes

| Setting | Effect |
|---|---|
| `--fps 2` | ~50% faster than 3; risk of missing short scenes |
| `--fps 5` | ~70% slower than 3; helps on fast content (sports, gameplay) |
| `--batch-size 16+` | Useful on a 12GB+ GPU |
| ViT-S/16 → ViT-B/16 | ~3× slower, marginally better embeddings |
| `--no-patches` | ~2× faster on the post-embedding stages (smaller vectors); ablation also tells you whether patches help |
| `--save-embeddings` | Caches per-run embeddings; rerun threshold sweeps without re-extracting features |

Rough numbers on a single RTX 4090 with DINOv3 ViT-S/16 at 3 FPS, ViT
input 224×224, batch 8: **~120 sampled FPS** for feature extraction. A
10-minute video processes in well under a minute, dominated by decode.

## Troubleshooting

**`PyAV` can't open the video.** Check that `ffmpeg` is installed and on
PATH. PyAV bundles its own FFmpeg in the wheel, but rebuilds from source
when the system one is mismatched.

**`AutoModel.from_pretrained` 401 / gated.** DINOv3 may be gated for your
account. Either `huggingface-cli login` and accept the model terms, or
let the fallback to DINOv2-small happen. Pass `--model facebook/dinov2-small`
to skip the v3 attempt entirely.

**No peaks detected, or modest real peaks are being missed.** Lower
`--peak-prom` (try `1.5` or `1.0`). This loosens both the absolute height
floor and the prominence requirement together. If the whole video is one
scene, MAD collapses and no peak clears either bar — that's the correct
behaviour, not a bug.

**Too many peaks.** Raise `--peak-prom` (try `2.5` or `3.0`). This
tightens both the height floor *and* the prominence requirement, so it
rejects small peaks regardless of whether they're absolutely high or just
locally prominent.

**Clusters of peaks bunched around a single transition.** Don't touch
`--peak-prom` — that hides small *real* peaks elsewhere. Instead raise
`--min-gap` (the minimum spacing between accepted peaks) and/or
`--smoothing` (which collapses multi-bump transitions into one peak).
Rule of thumb: `min_gap ≈ minimum_scene_length_s × fps`.

**Out of GPU memory.** Lower `--batch-size`. ViT-S/16 at batch 1 fits in
~2 GB VRAM.

**Keyframes look like motion blur.** Switch keyframe selection from
`centroid` to `peak` — but be aware the "peak" frame is the transition
moment, not a typical view of the scene.

## Extending the pipeline

Each stage is a single short file with no hidden coupling. To swap a
stage, write a drop-in replacement and call it from `src/main.py`.

**Replace the backbone** (e.g., SigLIP, CLIP, V-JEPA 2):
Implement a class with the same `embed_batch` signature as
`DinoFeatureExtractor`. The downstream code only needs L2-normalized
`(T, D)` vectors.

**Replace the scoring rule** (e.g., per-patch Sinkhorn distance):
Replace `compute_novelty`. The contract is `(T, D) → (T,)` float.

**Replace the peak rule** (e.g., learned changepoint detector):
Replace `detect_peaks`. The contract is `(T,) → NoveltyResult` with peak
indices, an absolute height floor (`threshold`), and a `prominence` value
to report — both are surfaced in the JSON output and the novelty plot.

**Replace keyframe selection** (e.g., max-information frame):
Replace `select_keyframes`. The contract is
`(segments, embeddings, pts_s, scores) → List[Scene]`.

See `THESIS.md` §13 for the planned ablation list.

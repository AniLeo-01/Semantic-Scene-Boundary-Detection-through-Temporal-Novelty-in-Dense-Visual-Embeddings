# Semantic Scene Boundary Detection via Embedding Novelty

A self-supervised system that segments videos into **semantically distinct scenes**
by measuring **novelty in dense visual embeddings** extracted from a vision
foundation model (DINOv3). It detects when "what is happening" changes — not
just when pixels change — without object detection, tracking, or labels.

This README is intentionally long. It is the design document: every term is
defined, every choice has an explanation, and every parameter is justified.
Skip to [Quick start](#quick-start) if you only want to run it.

---

## Table of contents

1. [Motivation: what problem are we solving?](#1-motivation)
2. [Why existing methods fall short](#2-why-existing-methods-fall-short)
3. [Core hypothesis](#3-core-hypothesis)
4. [Background: terms you need to know](#4-background-terms-you-need-to-know)
5. [Pipeline overview](#5-pipeline-overview)
6. [Detailed methodology, step by step](#6-detailed-methodology-step-by-step)
7. [Why this design? — choices, alternatives, trade-offs](#7-why-this-design)
8. [Honest limitations](#8-honest-limitations)
9. [Quick start](#9-quick-start)
10. [Configuration reference](#10-configuration-reference)
11. [Output format](#11-output-format)
12. [Roadmap and experiments](#12-roadmap-and-experiments)

---

## 1. Motivation

A "scene" in video is a stretch of time over which **what is happening stays
roughly the same**. A scene *boundary* is the moment that changes. For a human,
boundaries are obvious:

- a traffic light flips red → green, and the intersection's whole behavior shifts
- a manufacturing cell finishes a unit and starts the next
- a video-game player stops exploring and combat begins

In each case the **camera does not cut**. The pixels barely move. Yet the
*meaning* of the scene changes completely. We want a system that catches these
transitions automatically so we can:

- summarize long videos with one keyframe per scene
- index surveillance footage by activity, not by hour
- benchmark a production line by state, not by time
- search "find the moment the game became combat" without keywords

Crucially, we want this to work on **arbitrary video** — videos containing
objects we've never labeled, activities we've never named, environments we've
never seen.

## 2. Why existing methods fall short

Most off-the-shelf scene/shot detectors look for one of these signals:

- **Frame differencing** — subtract consecutive frames; threshold the residual.
  Catches camera cuts, fails on slow semantic transitions because the pixel
  difference between "exploring" and "combat begins" is small *per frame*.
- **Histogram comparison** — compare color distributions. Same failure mode,
  plus brittleness to lighting.
- **Optical flow** — measure motion. Triggers on *any* motion, not specifically
  semantically meaningful motion. A windy tree generates more flow than a
  traffic light changing.
- **Shot boundary detection** (e.g. TransNet V2) — purpose-built for film
  *cuts*. Excellent for that. Mostly silent on continuous-shot scene changes,
  which is what we care about.
- **Object detection + tracking** — works if you have a closed vocabulary and
  the right detector, and if "scene" is defined by which objects appear. Breaks
  the moment the change is about *what objects are doing*, not which exist.

What we want is a method that **understands the scene as a whole** and
notices when that whole has shifted, without enumerating its parts.

## 3. Core hypothesis

> **A scene can be represented by a high-dimensional semantic embedding.
> When the current frame's embedding cannot be well explained by recently
> observed embeddings, a semantic scene transition has occurred.**

This is a *novelty detection* framing of scene boundary detection. Instead of
"compare frame to frame," we ask: **does the most recent slice of history
already contain something that looks like this frame?** If yes, no boundary.
If no, boundary.

The hypothesis rests on two empirical bets:

1. **Foundation-model embeddings carry semantic content** beyond raw appearance.
   This is well-supported by the DINO/DINOv2/DINOv3 line of work: nearest
   neighbors in DINOv3 feature space are objects/scenes of the same *kind*, not
   just images of the same color palette.
2. **Semantic change is rare on short timescales.** Most consecutive frames are
   variations on the same situation, so a novelty score has a clear background
   level. Boundaries are the *peaks* above that background.

Both bets are testable. Bet #1 is the one most likely to break for "state"
transitions where appearance is nearly unchanged — see [§8](#8-honest-limitations).

## 4. Background: terms you need to know

This section defines vocabulary used throughout. Skip if familiar.

**Frame.** A single still image extracted from a video. A 30-FPS video has
30 frames per second.

**Frame rate (FPS).** Frames per second. Sampling at lower FPS means fewer
frames to process; we typically sample at **2–5 FPS** because semantic scenes
last several seconds.

**Embedding.** A vector of numbers that represents an image. Two images with
similar content end up with similar vectors. Modern image embeddings have a few
hundred to a few thousand dimensions.

**L2 normalization.** Scaling a vector so its length equals 1. After
normalization, comparing two embeddings by **cosine similarity** reduces to a
dot product, which is fast and numerically stable.

**Cosine similarity.** The cosine of the angle between two vectors. Equals 1
when they point the same direction (very similar), 0 when orthogonal (unrelated),
and −1 when opposite. For L2-normalized embeddings, cosine similarity is just
their dot product.

**DINOv3.** A self-supervised vision transformer released by Meta (Aug 2025).
"Self-supervised" means it was trained without labels — it learned image
representations by predicting transformations of its own inputs. The result is
a general-purpose feature extractor whose embeddings capture object identity,
scene structure, and spatial layout. We use the **ViT-S/16** variant: a Vision
Transformer (ViT) with patch size 16 and the "small" parameter count.

**Vision Transformer (ViT).** A neural network that chops an image into
fixed-size patches (here, 16×16 pixels), embeds each patch, and processes them
like tokens in a language model. The first token is a special **[CLS] token**
whose output represents the whole image; the remaining tokens represent local
patches.

**CLS embedding.** The output vector at the [CLS] token position. A single
holistic image-level vector, ~384-dim for ViT-S. Think of it as the model's
global summary of the frame.

**Patch (or dense) embedding.** One output vector per image patch. A 224×224
image with patch size 16 yields 14×14 = 196 patches, each with its own
embedding. These capture *local* semantics — what's in this corner of the
image.

**Patch-mean embedding.** The average of all patch embeddings, then
L2-normalized. A pooled summary of local content. Why average and not, say,
attention-pool? Averaging is parameter-free and works surprisingly well for
distinguishing "is the scene composition different overall?" without learning
a new pooling layer. We treat it as a complement to CLS, not a replacement.

**Combined embedding.** The concatenation `[CLS, patch_mean]` followed by L2
normalization. This is the per-frame vector we actually use. It's sensitive to
both the **global gestalt** (CLS) and the **average local content**
(patch_mean), which is useful because some scene transitions show up in one
signal but not the other (a global lighting change vs. a single hand entering
the frame).

**Memory bank.** A short queue (a "ring buffer") of recent embeddings. We
compare each new frame against this queue. Length is configurable; default 16
frames ≈ 5 seconds at 3 FPS.

**Novelty score.** A number per frame in `[0, 2]` measuring how unlike the
memory bank the current frame is. Formally: `novelty(t) = 1 − max_k cos(e_t, m_k)`
where `m_k` ranges over memory entries. Higher = more novel. We use **max**
similarity (i.e. compare to the most-similar memory entry) so that a long
stretch of repetitive footage doesn't average itself into looking "novel"
when something normal happens next.

**Peak detection.** Finding local maxima in a 1D signal that exceed a
threshold and are separated by a minimum gap. We use SciPy's `find_peaks` on
the novelty signal.

**Median / MAD.** The median is the middle value of a list; **MAD** is the
"median absolute deviation," i.e. the median of `|x − median|`. MAD is a
robust (outlier-resistant) estimator of spread, analogous to standard
deviation. Threshold = `median + k · 1.4826 · MAD` is a classic robust outlier
rule; the constant 1.4826 makes MAD comparable to a Gaussian standard
deviation.

**Scene segment.** A contiguous range of sampled frame indices `[start, end]`
between two consecutive boundaries (or between the start/end of the video and
the nearest boundary).

**Keyframe.** A single frame chosen to represent a scene. We pick the frame
whose embedding is closest (highest cosine similarity) to the segment's
embedding **centroid** — i.e. the most "average" or "typical" frame for that
scene. Alternative: pick the highest-novelty frame inside the segment (most
"defining" moment). Both are supported.

**Adaptive threshold.** A threshold computed from the data itself rather than
hand-set. Ours is `median + k·MAD`, which adapts to whatever baseline novelty
the video happens to have.

## 5. Pipeline overview

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
│  video  │ → │ sample   │ → │ DINOv3  │ → │ memory │ → │ novelty  │ → │  peak   │ → │ keyframe │
│  file   │   │ @ N FPS  │   │ embed   │   │  bank  │   │  score   │   │ detect  │   │  pick    │
└─────────┘   └──────────┘   └─────────┘   └────────┘   └──────────┘   └─────────┘   └──────────┘
                                                                            │
                                                                            ▼
                                                              ┌──────────────────────────┐
                                                              │ JSON boundaries +        │
                                                              │ keyframe images +        │
                                                              │ novelty signal plot      │
                                                              └──────────────────────────┘
```

Each box maps to a file in `src/`:

| Stage | File |
|---|---|
| sample | `src/sampling.py` |
| embed | `src/features.py` |
| memory + novelty + peaks | `src/novelty.py` |
| keyframe pick | `src/keyframes.py` |
| visualization | `src/viz.py` |
| orchestration / CLI | `src/main.py` |

## 6. Detailed methodology, step by step

### Step 1 — Frame sampling

Reading every frame of a 30-FPS video is wasteful: semantic scenes last
seconds, so adjacent frames are nearly identical embeddings. We decode the
video with **PyAV** (a FFmpeg binding) and yield approximately one frame every
`src_fps / target_fps` decoded frames. Default `target_fps = 3`. At this rate
a 10-minute video produces 1,800 frames — manageable on a laptop GPU and even
on CPU in a few minutes.

**Why decode every frame and drop, instead of seeking?** Seeking inside a
compressed video is unreliable for non-keyframes; decoding sequentially and
dropping is both faster and more accurate for short videos. For very long
videos, switch to keyframe-only decoding (`stream.codec_context.skip_frame =
"NONKEY"`) — a one-line change we'll add when needed.

### Step 2 — DINOv3 feature extraction

For each sampled frame we run DINOv3 ViT-S/16 and read out:

- the **CLS embedding** (1 × D vector, D ≈ 384 for ViT-S)
- the **patch embeddings** (~196 × D)

We L2-normalize the CLS, mean-pool the patches and L2-normalize the result,
then concatenate `[CLS, patch_mean]` and L2-normalize the concatenation. Call
this the **combined embedding**; it is the per-frame vector for the rest of
the pipeline. Dimension: 2D ≈ 768.

If DINOv3 weights aren't accessible (HF auth, offline, etc.) the code falls
back to **DINOv2-small** with the same interface. The pipeline does not depend
on a specific version of DINO; you can also pass `--model openai/clip-vit-base-patch16`
or any HuggingFace vision model that exposes `last_hidden_state`.

### Step 3 — Multi-scale scene representation

"Multi-scale" here means **two scales**: global (CLS) and local-averaged
(patch-mean). The CLS captures the gist ("urban intersection, daytime,
overhead view"); the patch-mean captures average local content ("lots of car
patches, asphalt, pedestrian crosswalk"). Concatenating both gives the
downstream cosine comparison **two independent shots** at noticing a change:
either the gist shifts, or the typical local content shifts, or both.

We deliberately do **not** use per-patch comparison at this stage. That would
be more expressive (you could spot a small region changing while the rest
stays still) but adds complexity and noise; it's on the roadmap, not in the
prototype.

### Step 4 — Temporal memory bank

A ring buffer `M` of the most recent `K` embeddings (default `K = 16`,
warmup 4). It is **strictly causal**: at time `t`, the buffer contains frames
`t−K, …, t−1`. The buffer represents "what the recent past looks like."

Why a fixed-size buffer rather than the entire history?

- We want to detect *recent* changes. A scene from 10 minutes ago should not
  influence whether the current frame is novel.
- Bounded memory makes the system suitable for **streaming**: the cost per
  frame is constant in video length.

Default `K = 16` at 3 FPS corresponds to roughly the last 5 seconds — long
enough to span a stable scene, short enough that we still notice fresh
content.

### Step 5 — Novelty detection

For each frame `t` we compute

```
novelty(t) = 1 − max_{m ∈ M}  cos(e_t, m)
```

A few details:

- **Cosine, not Euclidean.** Embeddings are L2-normalized; cosine similarity
  is the natural metric on the unit sphere. It's also robust to vector
  magnitude.
- **Max, not mean.** If we used mean similarity, a long run of similar
  embeddings would *average* the buffer toward the dominant scene; a later
  normal frame from that same scene could then look "less average" than
  expected and trigger a false peak. Max similarity asks "is there *any*
  recent frame this one resembles?" which is exactly the question we want.
- **Warmup.** For the first few frames the buffer is too small to be
  informative; we report novelty 0 during warmup to avoid spurious early
  peaks.

The output is a 1D signal `scores[t]` in `[0, 2]`, though in practice values
above ~0.4 are rare.

### Step 6 — Adaptive peak detection

We light-smooth the signal with a length-3 moving average (configurable). Then
compute an **adaptive threshold** from the data:

```
threshold = median(scores) + k · 1.4826 · MAD(scores)
```

where `MAD = median(|scores − median|)` and `k` (default 2.0) is the
"prominence" parameter. The constant 1.4826 makes MAD an unbiased estimator
of Gaussian standard deviation, so `k=2` behaves like "two sigma above
baseline."

Why MAD instead of standard deviation? **Robustness.** Real videos have a few
genuine outliers (the boundaries themselves) plus occasional noise spikes.
Standard deviation gets inflated by those outliers, raising the threshold and
hiding real boundaries. Median/MAD ignore them.

We then call SciPy's `find_peaks` with this threshold and a `distance =
min_gap` argument (default 8 frames ≈ 2.7 s at 3 FPS) to enforce a minimum
separation between boundaries. The `min_gap` prevents the same transition
from being detected twice if the novelty plateaus.

### Step 7 — Scene boundary generation

Peaks partition the sampled timeline into segments. If peaks occur at sampled
indices `[p1, p2, p3]` over `T` frames, the segments are
`[0, p1−1], [p1, p2−1], [p2, p3−1], [p3, T−1]`. We convert sampled indices
back to seconds using the original frame presentation timestamps (PTS).

### Step 8 — Keyframe extraction

For each segment we compute the **centroid embedding** (mean of all
embeddings in the segment, then L2-normalized) and pick the frame whose
embedding has the **highest cosine similarity to the centroid**. That frame
is the most "central" / "average" representative of the scene.

Alternative (`--keyframe-method peak`): pick the frame with the highest
novelty inside the segment. That gives you the moment of transition rather
than a typical view of the scene. Useful for highlight reels.

## 7. Why this design?

### Why DINOv3 (vs. CLIP, SigLIP, video models)?

- **CLIP / SigLIP** are *text-image contrastive* models. Their embeddings are
  optimized to align with captions. Good for retrieval, but they tend to
  abstract away spatial detail (whatever doesn't help match a caption).
  Boundary detection often hinges on *exactly* the spatial detail captions
  ignore.
- **DINOv3** is *self-supervised on images alone*. It is the current
  state-of-the-art for dense (per-patch) representations and explicitly
  preserves spatial structure. That makes its patch tokens useful, not just
  its CLS.
- **Video models (V-JEPA 2, VideoMAE V2, InternVideo2)** are arguably more
  appropriate — they natively encode temporal context. They are also heavier
  to run and harder to extract single-frame embeddings from cleanly. We treat
  them as a planned baseline (see [§12](#12-roadmap-and-experiments)).

### Why a memory bank instead of pairwise distances?

Pairwise (consecutive-frame) cosine distance is very sensitive to
high-frequency noise (camera shake, micro-motion). A memory bank effectively
denoises: a normal frame in a steady scene matches *something* in the buffer
even if it doesn't match the immediately preceding frame.

### Why MAD-based thresholding?

Because we don't know a priori how "novel" novel will look. A static
surveillance camera will have very low baseline novelty (~0.01) and a peak of
0.2 should fire. A gameplay video may have baseline 0.1 and a peak of 0.4
should fire. A fixed threshold like "0.3" gets one of those right and the
other wrong. MAD-based thresholding adapts.

### Why centroid keyframe selection?

We want the **representative** frame, not the **transition** frame. A user
scrolling through scene thumbnails sees a typical view of each scene, not
the chaotic moment it began. (For highlight reels, switch to `--keyframe-method
peak`.)

## 8. Honest limitations

**(a) "Semantic" is doing a lot of work.** DINOv3 is an *image* model. It
will detect transitions that have *some* visual correlate — pixels of
sufficiently different distribution. The traffic-light example in the
proposal works not because DINOv3 understands "red means stop and green
means go," but because the red→green pixel change plus the consequent
vehicle motion shows up in the embedding. A truly invisible state change
(e.g. a button is now armed but looks identical) will not fire.

**(b) Patch-mean is a coarse summary.** A small region changing in a
large field may not move the patch-mean enough to fire. A planned fix is
**per-patch novelty**: compare patch sets between current and memory frames
using optimal transport or top-k mean.

**(c) Adaptive threshold has corner cases.** If the *whole* video is one
scene, MAD collapses, the threshold approaches the maximum of the signal,
and we may still fire one or two spurious peaks. A safeguard is to require
the threshold to exceed an absolute floor (e.g. 0.05).

**(d) Sampling rate matters.** At 1 FPS you can miss short scenes; at 10 FPS
you waste compute and amplify noise. 2–5 FPS is the sweet spot for most
content; for sports or gameplay, push to 5–8 FPS.

**(e) Prior art exists.** This is a contribution to the **Generic Event
Boundary Detection** (GEBD) literature. Notable prior methods include
BaSSL, DDM-Net, SC-Transformer, and TransNet V2 (for cuts specifically).
A serious evaluation must include Kinetics-GEBD or TAPOS benchmarks.
This prototype does not yet ship that evaluation.

**(f) No false-positive rate guarantee.** We report a single threshold; we
don't currently estimate or bound the false-alarm rate. A principled
extension is to learn the null distribution per video (e.g. permutation
test) and report a p-value per peak.

## 9. Quick start

```bash
# 1. install deps
pip install -r requirements.txt

# 2. point at a video and an output dir
python -m src.main \
  --video sample_videos/clip.mp4 \
  --out outputs/run1 \
  --fps 3

# 3. results
ls outputs/run1/
#   boundaries.json    # full result, with timestamps and scene metadata
#   novelty.png        # the novelty signal with detected peaks
#   keyframes/         # scene_000.jpg, scene_001.jpg, ...
```

First run will download the DINOv3 (or DINOv2 fallback) weights from
HuggingFace, ~85 MB for ViT-S.

## 10. Configuration reference

| Flag | Default | What it controls |
|---|---|---|
| `--video` | — | Path to input video. |
| `--out` | — | Output directory; created if absent. |
| `--fps` | `3.0` | Sampling rate. Lower = faster, may miss short scenes. |
| `--model` | DINOv3 ViT-S/16 | Any HF vision model exposing `last_hidden_state`. |
| `--batch-size` | `8` | Embedding batch size. Raise on a GPU; lower on CPU. |
| `--memory` | `16` | Memory bank length in sampled frames. |
| `--peak-prom` | `2.0` | Threshold k in `median + k·MAD`. Lower = more peaks. |
| `--min-gap` | `8` | Minimum sampled-frame distance between boundaries. |
| `--no-patches` | off | Use CLS-only embeddings instead of `[CLS, patch_mean]`. |
| `--keyframe-method` | `centroid` | `centroid` or `peak`. |
| `--smoothing` | `3` | Length of moving-average filter on the novelty signal. |
| `--save-embeddings` | off | Cache `(T, D)` embeddings to `outputs/<run>/embeddings.npz`. |

## 11. Output format

`boundaries.json`:

```json
{
  "video": "sample_videos/clip.mp4",
  "model": "facebook/dinov3-vits16-pretrain-lvd1689m",
  "fps_sampled": 3.0,
  "n_frames_sampled": 540,
  "memory": 16,
  "peak_prom": 2.0,
  "min_gap": 8,
  "use_patches": true,
  "threshold": 0.0843,
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
    },
    "..."
  ]
}
```

`keyframes/scene_XXX.jpg`: one image per scene, JPEG quality 88, full
resolution of the sampled frame.

`novelty.png`: the novelty signal in time, with the adaptive threshold drawn
as a dashed line and detected peaks as vertical red lines.

## 12. Roadmap and experiments

The prototype's value is that each pipeline stage is a single small file you
can swap. Planned ablations:

1. **CLS-only vs. CLS+patches.** Does the patch-mean help? Run with and
   without `--no-patches`; compare F1 on Kinetics-GEBD.
2. **DINOv3 vs. DINOv2 vs. CLIP vs. SigLIP.** Quantify the gain from
   self-supervised vs. contrastive features.
3. **Memory length sweep.** 4, 8, 16, 32, 64 frames; expect a sweet spot.
4. **Per-patch novelty.** Replace `1 − max cos(e_t, m_k)` with a
   patch-token-set distance (e.g. Sinkhorn / 2-Wasserstein over patch
   embeddings). Hypothesis: catches small-region changes the mean misses.
5. **Video-native backbones.** V-JEPA 2 features over short clip windows
   instead of single frames.
6. **Benchmark.** Kinetics-GEBD precision/recall/F1 against reported
   numbers for BaSSL, DDM-Net, SC-Transformer.
7. **Online mode.** True streaming: the current code is already O(1)
   per-frame in memory, but main.py expects an offline file. Wrap it as a
   producer/consumer for RTSP.


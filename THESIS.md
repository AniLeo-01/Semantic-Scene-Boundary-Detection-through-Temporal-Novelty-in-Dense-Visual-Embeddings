# Thesis

## Semantic Scene Boundary Detection via Embedding Novelty

A research note on detecting *when meaning changes in a video* without
detecting any object, tracking anything, or learning from any label.

This document is the conceptual companion to the codebase. It explains
**what we are doing, why we are doing it, what we expect to be true, and
where we expect to be wrong.** If you want to know how to install and run
the system, see `README.md`. If you want to know what we believe and why,
read on.

---

## Table of contents

1. [The question](#1-the-question)
2. [What a "scene" actually is](#2-what-a-scene-actually-is)
3. [Why current methods miss this](#3-why-current-methods-miss-this)
4. [The hypothesis](#4-the-hypothesis)
5. [What we need to believe for this to work](#5-what-we-need-to-believe-for-this-to-work)
6. [Background: the vocabulary](#6-background-the-vocabulary)
7. [The method, in plain language](#7-the-method-in-plain-language)
8. [The method, more formally](#8-the-method-more-formally)
9. [Design choices and the alternatives we rejected](#9-design-choices-and-the-alternatives-we-rejected)
10. [Where this will fail — and how we'd know](#10-where-this-will-fail--and-how-wed-know)
11. [How this relates to prior work](#11-how-this-relates-to-prior-work)
12. [What a successful outcome looks like](#12-what-a-successful-outcome-looks-like)
13. [Open research questions](#13-open-research-questions)

---

## 1. The question

**Can a model that has never been told what a "scene" is learn to find scene
boundaries just by noticing when the visual world stops resembling its own
recent past?**

That's the whole project, condensed. We want a video segmentation system
that:

- requires no labels,
- requires no object detector,
- requires no tracker,
- handles arbitrary scenes — including objects, environments, and
  activities the model has never seen,
- detects *semantic* state transitions, not just pixel cuts,
- and runs on a laptop.

If we can do this, we have a generic, drop-in tool for video summarization,
surveillance triage, process monitoring, sports/gameplay analytics, and any
other domain where "where did the situation change?" matters.

## 2. What a "scene" actually is

The word "scene" is overloaded. In film it means a continuous shot, or a
narrative beat. In computer vision it usually means a shot — the footage
between two camera cuts. Neither matches what we mean here.

**For us, a scene is a stretch of time over which the situation stays
roughly the same.** A scene boundary is the moment the situation changes.
"Situation" is the slippery part. It includes:

- the environment (where we are),
- the actors (what's present),
- the state of those actors (red light vs. green light, assembled vs.
  unassembled),
- and the relationships between them (people queued vs. people scattered).

Critically, a scene boundary can happen with **no camera cut and no
substantial pixel change**. The traffic light flips red to green. The
manufacturing cell finishes a unit. The player stops exploring and combat
begins. To a human these are obvious; to a frame-differencer they are
silent.

That gap — between what humans perceive as a transition and what pixel
statistics measure as a transition — is what this project tries to close.

## 3. Why current methods miss this

Most off-the-shelf scene/shot detectors look at one of:

- **Pixel differences between frames.** Catches camera cuts; misses slow,
  meaningful change.
- **Color histograms.** Same failure plus sensitivity to lighting.
- **Optical flow.** Triggers on motion, not on *meaningful* motion. A
  windy tree generates more flow than a traffic light changing.
- **Trained shot boundary detectors** (e.g., TransNet V2). Excellent at
  film cuts; built for cuts, not for slow scene change inside a continuous
  shot.
- **Object detection + tracking.** Works only if "scene" is defined by
  which objects appear, and only if you have a detector that recognizes the
  right things. The moment the relevant change is about *what objects are
  doing* rather than *which exist*, this approach breaks.

What's missing is a method that **understands the scene as a whole** and
notices when that whole has shifted, without enumerating its parts.

## 4. The hypothesis

> **A scene can be represented as a single high-dimensional vector — its
> semantic embedding. If the current frame's embedding cannot be well
> explained by the embeddings of recently observed frames, a semantic
> transition has occurred.**

This is a *novelty detection* framing of scene boundary detection. Instead
of comparing frame to frame, we compare the current frame to a short window
of recent history. If the recent past already contains something that looks
like the current frame, no transition. If it doesn't, transition.

The hypothesis says **boundaries are spikes in a novelty signal.**

## 5. What we need to believe for this to work

The hypothesis only pays off if two empirical things are true:

**1. Foundation-model embeddings carry semantic content beyond appearance.**
That is, two frames that *look* similar but *mean* different things should
end up with measurably different embeddings. This is the bet on DINOv3
(and on self-supervised vision models in general). The DINO line of work
shows that nearest neighbors in feature space share object identity and
scene type, not just color palette. We are extending that bet by hoping
the embeddings also distinguish *states* of the same scene.

**2. Semantic change is rare on short timescales.** Most consecutive
frames are minor variations on the same situation. So a novelty score
computed against a memory bank will have a clear background level — a
quiet baseline — punctuated by genuine transitions. Without this
assumption, "peak above baseline" is meaningless.

Bet #2 is broadly true for the videos we care about (surveillance, sports,
gameplay, instructional, manufacturing). Bet #1 is the risky one and is
the bet most likely to fail for transitions where appearance is *truly*
unchanged.

## 6. Background: the vocabulary

For readers unfamiliar with the machinery. Skim if known.

**Frame.** A single still image extracted from a video.

**Frame rate (FPS).** Frames per second. We sample at 2–5 FPS because
scenes last several seconds; processing 30 FPS wastes compute.

**Embedding.** A vector of numbers that represents an image. Similar
images get similar vectors. Modern image embeddings have a few hundred to
a few thousand dimensions.

**L2 normalization.** Scaling a vector so its length is 1. After
normalizing, comparing two vectors by cosine similarity is just their dot
product.

**Cosine similarity.** The cosine of the angle between two vectors.
Ranges from 1 (identical direction) to −1 (opposite). For L2-normalized
vectors, cosine similarity equals the dot product. We measure "similar
content" with cosine similarity throughout.

**DINOv3.** Meta's self-supervised vision transformer, released in 2025.
"Self-supervised" means it learned from images alone, without labels, by
predicting transformations of its own inputs. The result is a
general-purpose feature extractor whose embeddings capture object
identity, scene structure, and spatial layout. We use the **ViT-S/16**
variant: a Vision Transformer (ViT) with patch size 16 and the "small"
parameter count (~21M).

**Vision Transformer (ViT).** A neural network that chops an image into
fixed-size patches (here, 16×16 pixels), embeds each patch, and processes
them like tokens in a language model. A special "[CLS] token" is prepended;
its output represents the whole image.

**CLS embedding.** The output vector at the [CLS] position. A single
holistic image-level vector, ~384-dim for ViT-S. The model's global
summary of the frame.

**Patch (dense) embedding.** One output vector per image patch. A
224×224 image with patch size 16 yields 14×14 = 196 patches and 196
patch embeddings. They capture *local* semantics — what's in this corner
of the frame.

**Patch-mean embedding.** Average of all patch embeddings, then
L2-normalized. A coarse pooled summary of local content. Cheap and
parameter-free.

**Combined embedding.** The concatenation `[CLS, patch_mean]`, then
L2-normalized. The per-frame vector we actually feed to the novelty
detector. Sensitive to both the global gestalt (CLS) and the average
local content (patch-mean) — useful because some transitions show up in
one signal but not the other.

**Memory bank.** A short queue of recent embeddings — a "ring buffer."
Default length: 16 frames. At 3 FPS this corresponds to roughly the last
5 seconds.

**Novelty score.** A number per frame in `[0, 2]` measuring how unlike
the memory bank the current frame is. Formally:
`novelty(t) = 1 − max_k cos(e_t, m_k)` where `m_k` ranges over memory
entries. Higher = more novel.

**Peak detection.** Finding local maxima in a 1D signal that
(a) clear an absolute height floor, (b) stand out by at least a given
**prominence** from their local neighbourhood, and (c) respect a minimum
spacing. We use SciPy's `find_peaks` with all three constraints. Using
prominence alongside an absolute floor catches modest real peaks above a
noisy baseline — peaks an absolute-only rule misses — while still
rejecting noise spikes that don't clear the floor.

**Prominence.** The vertical distance from a peak down to the lowest
contour line that doesn't enclose a higher peak. Operationally: how far
the peak rises above the surrounding signal. Robust to a noisy global
baseline; what matters is local shape, not absolute value.

**Median / MAD.** The median is the middle value of a list. **MAD** is
the "median absolute deviation," the median of `|x − median|`. MAD is a
robust (outlier-resistant) estimator of spread, analogous to standard
deviation. We derive both detector knobs from MAD:
the height floor `0.6 · (median + k · 1.4826 · MAD)`
and the prominence requirement `k · 1.4826 · MAD`. The constant 1.4826
makes MAD a Gaussian-std estimator, so `k = 2` behaves like "two sigma."

**Scene segment.** A contiguous range of sampled frame indices
`[start, end]` between two consecutive boundaries (or between the start /
end of the video and the nearest boundary).

**Keyframe.** A single frame chosen to represent a scene. We pick the
one whose embedding is closest (highest cosine similarity) to the
segment's embedding centroid — the most "average" or "typical" frame.
Alternative: highest-novelty frame inside the segment.

## 7. The method, in plain language

1. Walk through the video and grab one frame every third of a second.
2. For each frame, get a numerical fingerprint from DINOv3.
3. Keep the last 16 fingerprints in a sliding window — the **memory bank**.
4. For each new frame, ask: *how similar is this fingerprint to the most
   similar one in the bank?* If very similar, this frame "fits" the recent
   past — nothing new is happening. If not similar at all, this frame is
   **novel** — something changed.
5. Plot the novelty score over time. Mostly flat with a few sharp peaks.
6. Pick the peaks. A peak counts if it (a) clears a loose absolute floor
   derived from the baseline (median + 2 × MAD) and (b) rises by at
   least 2 × MAD above its immediate surroundings. Each accepted peak is
   a scene boundary.
7. Between two boundaries is a scene. Pick the most representative frame
   in each scene — the one closest to the average of that scene's
   fingerprints — as the keyframe.

That's the entire algorithm. No training. No labels. No detectors. No
trackers. Less than 300 lines of code in total.

## 8. The method, more formally

Given a video, sample frames at `f` FPS to obtain a sequence
`I_1, …, I_T`.

For each `I_t`, extract from DINOv3:

- the [CLS] token output `c_t ∈ ℝ^D`,
- the patch token outputs `{p_t^j} ⊂ ℝ^D` for `j = 1, …, N`,

and form the per-frame vector

```
e_t = normalize( [ normalize(c_t),  normalize( (1/N) Σ_j p_t^j ) ] ) ∈ ℝ^{2D}
```

Maintain a memory bank `M_t = { e_{t−1}, e_{t−2}, …, e_{t−K} }` of length
`K` (the ring buffer).

Compute novelty:

```
novelty(t) = 1 − max_{m ∈ M_t}  ⟨ e_t, m ⟩
```

(For L2-normalized vectors `⟨·, ·⟩` is cosine similarity.) Smooth the
signal with a length-`w` moving average. Derive two adaptive thresholds
from the data:

```
τ_height = 0.6 · ( median(novelty) + k · 1.4826 · MAD(novelty) )
τ_prom   =        k · 1.4826 · MAD(novelty)
```

with `k ≈ 2`. Accept a sampled-frame index `t` as a boundary iff

- `novelty(t) ≥ τ_height`         (absolute floor — kill obvious noise),
- prominence of the peak at `t` is `≥ τ_prom`   (the peak stands out from
  its local surroundings), and
- it is at least `g` frames from any previously accepted peak.

The prominence rule is what makes the detector robust to a noisy global
baseline: a peak counts when it is a *local* event of sufficient height,
even if the absolute value is only modestly above baseline. The height
floor is a safeguard so the prominence rule isn't applied to garbage
near zero novelty.

Peaks partition `[1, T]` into segments `S_1, …, S_n`. For each segment
`S_i`, compute the embedding centroid

```
μ_i = normalize( (1/|S_i|) Σ_{t ∈ S_i} e_t )
```

and select the keyframe

```
k_i = argmax_{t ∈ S_i}  ⟨ e_t, μ_i ⟩.
```

Output: `( S_i, k_i )_{i=1}^n`. Done.

## 9. Design choices and the alternatives we rejected

### Why DINOv3 and not CLIP or SigLIP?

CLIP and SigLIP are text-image contrastive models; their embeddings are
optimized for caption alignment. Excellent for retrieval. But contrastive
training abstracts away whatever doesn't help match a caption — exactly
the spatial detail that scene transitions often hinge on. DINOv3 is
self-supervised on images alone and is currently the strongest model for
dense (per-patch) representations. Its patch tokens are usable, not just
its [CLS].

### Why not a video model (V-JEPA 2, VideoMAE, InternVideo)?

Video models natively encode temporal context — arguably the right choice
for any video task. They are also heavier to run, slower to extract
single-frame embeddings from, and harder to drop into a streaming
pipeline. We treat them as a planned baseline, not a starting point.

### Why a memory bank and not consecutive-frame distance?

Frame-to-frame cosine distance is very sensitive to high-frequency noise
(camera shake, micro-motion, breathing). A memory bank effectively
denoises: a normal frame in a steady scene matches *something* in the
buffer even if it doesn't match the immediately preceding frame.

### Why max-similarity and not mean-similarity?

Imagine a long stretch of similar frames in the memory bank, all close to
some center `c`. The *mean* of the buffer is also close to `c`. Now
another normal frame from the same scene arrives. Its similarity to the
mean is high, but its similarity to *any individual* memory entry is
slightly higher. Using mean inflates "distance from the mean" and risks
false peaks. Max-similarity asks the right question: *is there any frame
in recent memory that this one resembles?*

### Why MAD-based threshold and not a fixed cut-off?

A static surveillance camera has baseline novelty around 0.01; a peak of
0.2 should fire. A gameplay video has baseline ~0.1; a peak of 0.4
should fire. A fixed threshold gets one of those right and the other
wrong. MAD-based thresholding adapts to whatever baseline the video
happens to have, and is robust to the outliers (the boundaries
themselves) that would inflate a standard-deviation-based estimate.

### Why height *and* prominence, instead of height alone?

An earlier version of the detector used the absolute MAD-based threshold
only. It failed in a specific way: on noisy videos the baseline isn't
clean. Real boundaries produce peaks that are only modestly above the
*global* threshold, while clearly being local maxima. Raising the
threshold to suppress noise hid those real peaks; lowering it
re-introduced noise. The trade-off had no good setting.

Switching to a height-plus-prominence rule fixes that. **Prominence**
asks a different question: not "is the peak's absolute value high
enough?" but "does the peak rise enough above its immediate
surroundings?" Even on a noisy baseline, a real boundary causes a sharp
local rise — its prominence is large even when its absolute value is
modest. The absolute height floor (scaled to `0.6 × MAD-threshold`)
remains as a safety net so the prominence rule isn't applied to
near-zero noise.

Both knobs are tied to the same `prominence_k`. A user who wants more
peaks lowers `k` and gets a looser height floor *and* a looser
prominence requirement. A user who wants fewer raises `k`. The
`min_gap` and `smoothing` knobs control *clustering* (the same
transition firing multiple times), which is orthogonal — raising `k`
to fix clusters silently hides real peaks elsewhere, while raising
`min_gap` doesn't.

### Why centroid keyframe selection and not the highest-novelty frame?

For a thumbnail-style summary, the user wants the **representative**
frame of a scene, not the **chaotic transition moment** at its start.
Centroid-closest gives a calm, typical view. For highlight reels, the
opposite is true, so we expose `--keyframe-method peak`.

### Why concatenate CLS and patch-mean, instead of using one?

Originally we hypothesized CLS would capture the gist while patch-mean
captures average local content; concatenating gives the cosine
comparison two independent shots at noticing a change.

**The ablation has been run on Charades** and the patch-mean term
contributes nothing measurable (CLS-only F1 ≈ pooled F1 within 0.005).
The patch-mean averages away exactly the localised information it was
supposed to add. The current method effectively reduces to CLS-only
novelty for that benchmark.

The replacement direction is *per-patch* novelty rather than
pooled-patch novelty — see [§13](#13-open-research-questions) item 4.
Each query patch is matched against the patch sets of recent frames
without ever being averaged, so localised changes can't be washed out.

## 10. Where this will fail — and how we'd know

This is the section to read if you suspect the method is too good to be
true. **It is.** Several known failure modes:

**(a) "Semantic" is doing a lot of work in the title.** DINOv3 is an
image model. It will only detect transitions that have *some* visual
correlate. The traffic-light example in the proposal works not because
DINOv3 understands "red means stop and green means go," but because the
red→green pixel change and the consequent vehicle motion both show up in
the embedding. A truly invisible state change — a button is now armed but
looks identical — will not fire. **How we'd know:** construct a paired
test where two clips have identical appearance but different meaning. If
novelty stays flat, the claim is invalid.

**(b) Patch-mean is a coarse summary.** A small region changing inside a
large field may not move the patch-mean enough to fire. **Fix on the
roadmap:** per-patch novelty using optimal transport or top-k matching
over patch token sets, so localized changes can dominate the score.

**(c) Adaptive threshold has corner cases.** If the *whole* video is one
scene, MAD collapses, both the height floor and the prominence
requirement collapse with it,
and we may still fire spurious peaks. **Fix:** require the threshold to
exceed an absolute floor (e.g., 0.05).

**(d) Sampling rate is a hidden hyperparameter.** Below 1 FPS we miss
short scenes; above 8 FPS we waste compute and amplify noise. Hard to
choose a priori. **Fix on the roadmap:** auto-tune via signal SNR.

**(e) No false-positive rate guarantee.** We report a single threshold;
we do not estimate or bound the false-alarm rate. **Fix:** permutation
test to learn the null novelty distribution per video, then report a
p-value per peak.

## 11. How this relates to prior work

The problem sits squarely in the literature on **scene/event boundary
detection in video**. Notable prior methods include:

- **TransNet V2** — purpose-built shot boundary detector. Strong on
  cuts; not designed for slow semantic transitions.
- **BaSSL** — boundary-aware self-supervised learning for movies.
- **DDM-Net** — dense difference module for event boundaries.
- **SC-Transformer** — structured context transformer for event
  boundaries.

What's novel about our approach relative to these:

- **Fully training-free.** No fine-tuning of the backbone. No learned
  boundary head. The whole pipeline is signal processing over frozen
  DINOv3 features.
- **Memory-bank novelty rather than learned scoring.** Most modern
  boundary-detection methods learn a per-frame boundary probability.
  We argue that with a strong enough feature extractor, plain
  max-cosine-against-recent-memory is competitive, and that argument
  is empirically falsifiable.
- **Two scales of representation.** CLS + patch-mean, treated as a single
  vector. Most baselines use one or the other.

If the ablations land, the contribution is **a strong, fully
self-supervised, training-free baseline for narrative-scene boundary
detection.** That is a small but real publishable result. If the
ablations show DINOv3 is not enough and that video models are required,
the negative result is still informative.

## 12. What a successful outcome looks like

Concretely, we declare success if:

- On Charades (or another temporal-action-segment benchmark), F1 at
  a moderate tolerance (`rel_dis ≈ 0.1`) is within ~5 points of trained
  baselines despite using zero training.
- On a hand-curated set of "slow semantic transition" clips (traffic
  lights, manufacturing, gameplay), the system fires within ±1 second
  of the true transition in ≥ 70% of cases.
- Inference runs at ≥ real-time (≥ 30 FPS effective at 3 FPS sampling)
  on a modest GPU.
- Removing the patch-mean (CLS-only) measurably hurts recall on the
  slow-transition set, justifying the multi-scale design.

If all four hit, this is a credible research contribution and a useful
piece of infrastructure. If only the first three hit, the patch-mean
isn't pulling its weight and the method simplifies. If none hit, we've
falsified the hypothesis and should move to video-native backbones.

## 13. Open research questions

The prototype is small on purpose. Each pipeline stage is a single short
file, so the following are days-of-work changes, not weeks:

1. **CLS-only vs. CLS + patches.** Does the patch term actually help?
2. **DINOv3 vs. DINOv2 vs. CLIP vs. SigLIP.** Quantify the gain from
   self-supervised vs. contrastive features for this task.
3. **Memory length sweep.** 4, 8, 16, 32, 64 frames. Where is the sweet
   spot? Does it depend on video content?
4. **Per-patch novelty.** Replace `1 − max cos(e_t, m_k)` with a
   patch-token-set distance. A Chamfer formulation (each query patch's
   single best match against all memory patches, aggregated mean / topk
   / min across the N query patches) is implemented as
   `src/novelty.py:compute_patch_novelty`; access it via
   `--patch-novelty --patch-agg {mean,topk,min}` on any runner.
   Hypothesis: catches small-region changes the pooled-mean misses.
   Status: code in place, F1 measurement pending.
5. **Video-native backbones.** Swap the DINOv3 per-frame feature for
   V-JEPA 2 short-clip features. How much closer does it get to
   "semantic" transitions that have no per-frame visual signal?
6. **Permutation-test p-values.** Learn the null novelty distribution
   per video and emit calibrated p-values per peak, not a single
   adaptive threshold.
7. **Online / streaming mode.** The current code is O(1) per frame in
   memory. Wrap it as a producer-consumer for RTSP streams.
8. **Hierarchical segmentation.** Re-run the detector at multiple memory
   lengths to get coarse-to-fine scene structure (acts → scenes →
   beats).
9. **Multi-modal.** Add audio (CLAP) and ASR text (Whisper) tokens to
   the memory bank. Many "scene changes" are signaled first by sound.

The version-1 paper writes itself from items 1–4 alone, even before any
of the speculative extensions.

---

*This document is a thesis statement, not a finished study. Every claim
above is meant to be testable, and most of the testing has not yet been
done. The code in this repository exists to make those tests cheap.*

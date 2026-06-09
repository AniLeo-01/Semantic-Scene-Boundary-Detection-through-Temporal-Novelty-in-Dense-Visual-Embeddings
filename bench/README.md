# Validation benchmarks

Currently implemented:

- **Charades** (`bench/charades.py`) — temporal action segments treated
  as scene-like boundaries. Direct download from AI2's S3 (no YouTube,
  no auth, no scraping).
- **Custom** (`bench/custom.py`) — hand-annotated videos you provide.

The scoring logic in `bench/metrics.py` is dataset-agnostic — F1@rel_dis
with greedy 1-to-1 matching — so adding a new dataset is just a label
loader and a `run_eval()` wrapper.

## Charades — quick start

The [Charades](https://prior.allenai.org/projects/charades) dataset
contains **9,848 indoor-activity videos** with dense temporal action
segments. Each segment is "do activity X from t1 to t2"; we treat the
start and end of every segment as boundary candidates. The result is a
scene-like benchmark where "scenes" are coherent activity intervals.

```bash
# 1. annotations (~3 MB, fast) — grab this first to inspect labels
python -m bench.fetch_charades --out data/charades --annotations

# 2. videos (~13 GB, takes a while)
python -m bench.fetch_charades --out data/charades --videos-only

# 3. run the bench
python -m bench.charades \
  --annotations data/charades/Charades/Charades_v1_test.csv \
  --videos      data/charades/Charades_v1_480 \
  --out         outputs/charades_run1 \
  --model       facebook/dinov3-vits16-pretrain-lvd1689m \
  --fps 5 --memory 12 --peak-prom 1.8 --min-gap 3 --batch-size 64 \
  --max-videos 500
```

Charades videos average ~30 s, so the default `rel_dis` grid
`(0.05, 0.1, 0.2, 0.3, 0.4, 0.5)` translates to 1.5–15 s tolerance.
Headline number: F1 at `rel_dis = 0.1` (3 s tolerance on a 30 s clip).

### Important caveats

1. **Activity segments overlap.** Charades is multi-label — multiple
   actions can happen simultaneously, so we de-duplicate boundaries
   within `--dedup-tol-s` seconds (default 0.5). Experiment with
   stricter or looser dedup if results look off.
2. **Some test videos have zero actions.** The bench drops these from
   the F1 calculation and reports the effective N.
3. **"Scene" here is "activity interval."** It's not a narrative scene
   in the cinematic sense (Charades is one continuous camera in one
   room). The boundaries you're scoring against are action-state
   transitions, which is closer to what GEBD attempted — but with clean
   labels and no YouTube downloads.

### Parameter notes for short clips

| Knob | Value | Why |
|---|---|---|
| `--fps 5` | 5 FPS | 30 s clip × 5 FPS = 150 frames; enough for transitions |
| `--memory 12` | ~2.4 s | activities last ~5–15 s |
| `--peak-prom 1.8` | loose | small per-video sample, want recall |
| `--min-gap 3` | ~0.6 s | activity transitions can be close together |

## What gets measured

For each video, predicted boundary times in seconds are matched
1-to-1 against ground-truth boundary times. A prediction is a true
positive iff it lies within `rel_dis × video_duration` seconds of an
unmatched GT boundary. Greedy nearest-neighbour assignment.

We report F1 at six `rel_dis` tolerances; precision and recall are
reported alongside for diagnosis.

## Cached re-scoring

`predictions.json` (the boundary timestamps per video) is cached, so
sweeping `rel_dis` grids or evaluating against a different label split
is free:

```bash
python -m bench.charades \
  --annotations data/charades/Charades/Charades_v1_test.csv \
  --predictions outputs/charades_run1/predictions.json \
  --out         outputs/charades_rescored \
  --eval-only
```

Changing `--peak-prom`, `--memory`, `--fps`, `--model` etc. requires
fresh predictions since those run at prediction time.

## Suggested ablation order

1. Baseline DINOv3 + patches at default settings.
2. `--no-patches` — does the patch-mean term help?
3. `--model facebook/dinov2-small` — backbone ablation.
4. `--peak-prom 1.5, 2.0, 2.5, 3.0` — precision-recall curve.
5. `--memory 6, 12, 24, 48` — how much "recent past" matters for short clips.
6. `--fps 3, 5, 10` — temporal resolution.

## Custom videos — annotate and evaluate your own clips

When you want to test on your own footage — domain videos (surveillance,
gameplay, manufacturing), a few movie trailers, anything — use
`bench/custom.py`. You bring the videos and a list of boundary
timestamps; the bench runs the same pipeline and scores F1@rel_dis.

### Step 1 — annotate

Pick whichever label format suits you. All three are accepted by
`bench/custom.py`.

**(a) Plain text** — quickest, no headers, durations auto-read from the
videos:

```
# data/custom/labels.txt
# format: <filename>  <comma-or-space-separated boundary times>
# times can be plain seconds or HH:MM:SS.mmm
clip_01.mp4    12.5, 89.3, 110.0
clip_02.mp4    00:00:22.1  00:00:41.0  00:00:55.4
gameplay_a.mp4  4.0  18.5  62.0  91.3  120.0
```

**(b) JSON** — best if you want multiple annotators or explicit durations:

```json
{
  "clip_01": {"duration_s": 142.3, "boundaries": [12.5, 89.3, 110.0]},
  "clip_02": {"duration_s":  73.8, "boundaries": [22.1, 41.0, 55.4, 65.0]},
  "gameplay_a": {"duration_s": 180.0,
                 "boundaries": [[4.0, 18.5, 62.0], [4.2, 18.0, 61.5]]}
}
```

**(c) CSV** — best for editing in a spreadsheet:

```
video,boundary_s,duration_s
clip_01,12.5,142.3
clip_01,89.3,142.3
clip_02,22.1,73.8
clip_02,41.0,73.8
clip_02,55.4,73.8
```

**How to actually find the timestamps:**

- Easiest: open the video in **VLC** (`Ctrl-T` shows current time), scrub
  to each boundary, write down the timestamp.
- A bit better: **MPV** with `mpv --osd-fractions clip.mp4` shows the
  fractional second in the OSD. Same drill.
- Even better: use the project's own `viewer.html` (auto-generated by
  `python -m src.main`) — it puts the video next to a chart with a
  clickable timeline; you can scrub frame-by-frame and read the time off.

Tip: don't overthink it. Be within ±0.5s of where you think a boundary
is; the bench's `rel_dis` tolerance is forgiving.

### Step 2 — run

```bash
python -m bench.custom \
  --labels  data/custom/labels.json \
  --videos  data/custom/videos \
  --out     outputs/custom_run1 \
  --model   facebook/dinov3-vits16-pretrain-lvd1689m \
  --fps 3 --memory 16 --peak-prom 2.0 --min-gap 6 --batch-size 64
```

The runner prints a console line per video showing predicted-vs-GT
boundary counts, caches `predictions.json`, and at the end prints the
familiar `rel_dis | n_videos | P | R | F1` table.

### Step 3 — iterate

The cached predictions don't change with `rel_dis` tolerance, so any
threshold sweep is free:

```bash
python -m bench.custom \
  --labels      data/custom/labels.json \
  --predictions outputs/custom_run1/predictions.json \
  --out         outputs/custom_rescored \
  --eval-only
```

Anything else (FPS, memory, peak prominence, backbone) requires fresh
predictions.

### Reasonable defaults by clip length

| Clip length | `--fps` | `--memory` | `--min-gap` |
|---|---|---|---|
| < 30 s | 5 | 10–14 | 3 |
| 30 s – 5 min | 3–5 | 16–24 | 6 |
| 5 – 30 min | 3 | 24–48 | 12 |
| > 30 min | 2–3 | 48–96 | 30 |

A useful sanity check before running on dozens of videos: run on **one**
clip, then open the generated `viewer.html` (via `python -m src.main`,
not via `bench.custom`) and eyeball whether the detected peaks line up
with your annotated boundaries. If they don't, your params are off and
no amount of F1 sweeping will fix it.

## Adding a new dataset

The pattern is short:

1. Implement `load_<dataset>_labels(...) -> Dict[video_id, {"duration_s", "boundaries"}]`.
2. Implement `predict_one(...)` (basically identical to the one in
   `bench/charades.py` — feel free to import-reuse).
3. Wrap them in `run_eval(...)` that calls `bench.metrics.f1_grid`.
4. Add a `parse_args()` + `__main__` block.
5. Add a one-paragraph section in this README.

Total: ~150 lines per new dataset.

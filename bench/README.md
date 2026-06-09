# Validation benchmarks

Currently implemented: **BBC Planet Earth** (`bench/bbc.py`).

The scoring logic in `bench/metrics.py` is dataset-agnostic — F1@rel_dis
with greedy 1-to-1 matching — so adding a new dataset is just a label
loader and a `run_eval()` wrapper.

## BBC Planet Earth — quick start

The BBC Planet Earth dataset has 11 ~50-minute nature-documentary
episodes with annotated *narrative* scene boundaries (i.e. when the
documentary switches between scenes / topics, not just between camera
cuts). It's the canonical small benchmark for what this project
actually tries to do.

```bash
# 1. obtain the videos (academic mirrors — links rot, you may need to
#    search "BBC Planet Earth dataset scene boundary").
#    Put them in data/bbc/episodes/EP01.mp4 ... EP11.mp4

# 2. obtain the labels (JSON or CSV, see bench/bbc.py for format).
#    Put them in data/bbc/labels.json

# 3. run the bench
python -m bench.bbc \
  --labels  data/bbc/labels.json \
  --videos  data/bbc/episodes \
  --out     outputs/bbc_run1 \
  --model   facebook/dinov3-vits16-pretrain-lvd1689m \
  --fps 3 --memory 24 --peak-prom 2.0 --min-gap 30 \
  --batch-size 64
```

Expected runtime on an L4 GPU: ~10 min per episode at 3 FPS, so ~2 hours
for all 11 episodes. Cache the predictions and re-score with different
hyperparameters cheaply.

## Label-file formats accepted

**JSON, single annotator** (simplest):

```json
{
  "EP01": {"fps": 25, "duration_s": 3010.0,
           "boundaries": [12.4, 81.0, 154.6, ...]},
  "EP02": {...}
}
```

**JSON, multiple annotators** (some BBC mirrors ship 3 annotators):

```json
{
  "EP01": {"duration_s": 3010.0,
           "boundaries": [[12.4, 81.0, ...], [13.0, 80.5, ...]]},
  "EP02": {...}
}
```

**CSV** (with companion `--durations episode_durations.json`):

```
episode,boundary_s
EP01,12.4
EP01,81.0
EP02,9.8
```

## What gets measured

For each episode, predicted boundary times in seconds are matched
1-to-1 against ground-truth boundary times. A prediction is a true
positive iff it lies within `rel_dis × episode_duration` seconds of an
unmatched GT boundary. Greedy nearest-neighbour assignment.

We report F1 at a tight grid suited to long-form documentary
(boundaries are sparse, so absolute tolerances of 15–600 seconds on a
~3000 s episode are reasonable):

| rel_dis | absolute tolerance on a 50-min episode |
|---|---|
| 0.005 | 15 s |
| 0.01 | 30 s |
| 0.02 | 60 s |
| 0.05 | 2.5 min |
| 0.1 | 5 min |
| 0.2 | 10 min |

For published BBC numbers, look at the *strict* end (`rel_dis ≈ 0.005`).
Recent papers report F1@strict in the 0.40–0.70 range on this dataset
with various trained methods.

## Cached re-scoring

`predictions.json` (the boundary timestamps per episode) is cached, so
sweeping rel_dis grids or threshold parameters that don't change the
underlying signal is free:

```bash
python -m bench.bbc \
  --labels      data/bbc/labels.json \
  --predictions outputs/bbc_run1/predictions.json \
  --out         outputs/bbc_rescored \
  --eval-only
```

(`--eval-only` only avoids re-running the pipeline when the predictions
are already cached. Changing `--peak-prom` requires fresh predictions
since the peak rule runs at prediction time.)

## Suggested ablation order

1. Baseline DINOv3+patches at default settings.
2. `--no-patches` — does the patch-mean term help on long documentary?
3. `--memory 12, 24, 48, 96` — how much "recent past" matters for
   minute-scale scenes.
4. `--peak-prom 1.5, 2.0, 2.5, 3.0` — precision-recall curve.
5. `--model facebook/dinov2-small` — backbone ablation to compare with v3.
6. Eventually: swap in V-JEPA 2 features (see THESIS.md §13).

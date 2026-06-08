# Kinetics-GEBD validation

This folder contains the benchmark harness for the
[Generic Event Boundary Detection](https://github.com/StanLei52/GEBD)
task (Shou et al., ICCV 2021) — the canonical evaluation for the
problem this project addresses.

## What gets measured

For each video, a *predicted* set of boundary timestamps `{p_i}` (in
seconds) is compared against *each* annotator's ground-truth set
`{g_j^k}` (each video is annotated by ~5 humans). A prediction is a
true positive if it falls within `rel_dis · video_duration` seconds
of an unmatched ground-truth boundary.

The official metric is **F1 averaged over annotators, then averaged
over videos**, reported at several `rel_dis` tolerances (0.05 strict,
0.5 lenient). The headline number is **F1 @ rel_dis = 0.05**.

## End-to-end procedure

```bash
# 1) install benchmark deps
pip install yt-dlp                # for video download
# (the harness itself uses numpy + the existing src/ stack)

# 2) get labels (~5 MB pickle from the GEBD repo)
mkdir -p data/gebd && cd data/gebd
curl -L -o k400_mr345_val_min_change_duration0.3.pkl \
  https://github.com/StanLei52/GEBD/raw/main/data/export/k400_mr345_val_min_change_duration0.3.pkl
cd ../..

# 3) download a sample of videos (full val set is ~4k after attrition;
#    start with 200 to verify your pipeline runs)
python -m bench.fetch_kinetics_gebd \
  --labels data/gebd/k400_mr345_val_min_change_duration0.3.pkl \
  --out    data/gebd/val_videos \
  --max    200

# 4) run predictions + score
python -m bench.kinetics_gebd \
  --labels data/gebd/k400_mr345_val_min_change_duration0.3.pkl \
  --videos data/gebd/val_videos \
  --out    outputs/gebd_run1 \
  --fps 3 --memory 16 --peak-prom 2.0 --min-gap 8 \
  --batch-size 64 \
  --max-videos 200
```

Console output:

```
rel_dis  n_videos       P       R      F1
   0.05       N    0.???   0.???   0.???
   0.10       N    0.???   0.???   0.???
   0.20       N    0.???   0.???   0.???
   ...
```

and the same numbers land in `outputs/gebd_run1/summary.json`.

Predictions are cached at `outputs/gebd_run1/predictions.json` so
ablations don't have to re-extract features:

```bash
# re-score with looser peak detection — no GPU needed
python -m bench.kinetics_gebd \
  --labels data/gebd/k400_mr345_val_min_change_duration0.3.pkl \
  --predictions outputs/gebd_run1/predictions.json \
  --eval-only --out outputs/gebd_rescored
```

(Note: re-scoring only applies if you change the *post-prediction*
config. Changing FPS, memory, model, etc. requires fresh predictions.)

## Reference numbers

Published F1 @ rel_dis = 0.05 on the GEBD val set:

| Method | F1@0.05 | Trained? |
|---|---|---|
| Random uniform | ~0.30 | no |
| BMN baseline | 0.49 | yes |
| TransNet V2 (shot detector, not GEBD-tuned) | ~0.46 | yes |
| BaSSL | 0.65 | yes (self-sup) |
| DDM-Net | 0.76 | yes (sup) |
| SC-Transformer | 0.78 | yes (sup) |

Numbers above ~0.65 with **no training** would be a real result.

## What to look at in the output

`summary.json`:

```json
{
  "config": { "fps": 3.0, "memory": 16, "peak_prom": 2.0, ... },
  "n_predicted": 187,
  "n_evaluated": 187,
  "metrics": [
    { "rel_dis": 0.05, "precision": 0.??, "recall": 0.??, "f1": 0.?? },
    ...
  ]
}
```

`predictions.json` is `{ video_id: [boundary_seconds, ...] }`.

## Ablations worth running

Each is one CLI invocation against the cached `predictions.json` is
*not* enough — these change feature extraction. Sweep these:

| Knob | Values | Hypothesis |
|---|---|---|
| `--peak-prom` | 1.5, 2.0, 2.5, 3.0 | sensitivity / precision-recall tradeoff |
| `--memory` | 4, 8, 16, 32, 64 | how much "recent past" helps |
| `--no-patches` | on/off | does the patch-mean term actually help? |
| `--model` | dinov3-vits16, dinov3-vitb16, dinov2-small, dinov2-base | architecture vs. params |
| `--fps` | 2, 3, 5 | temporal resolution |

A precision-recall curve over `peak_prom` is the most useful single plot.

## Other datasets

`bench/kinetics_gebd.py` is the only validator implemented today.
Planned (each is a small adapter over the same scoring code):

- **MovieScenes** (MovieNet) — narrative-scene labels in addition to
  shot labels. Tests the project's biggest limitation: do we recover
  scenes or just shots? See THESIS.md §10.
- **BBC Planet Earth** — small, classic, easy to demo.
- **TAPOS** — within-action sub-boundaries.

Open a PR or ping the author if you start one of these.

"""Combine predictions from two methods and score the union (or intersection).

Two methods can have different P/R profiles and may detect *different*
true boundaries. Combining them tells you whether they're complementary
(union F1 > both, intersection F1 > both at higher precision) or
redundant (union/intersection ≈ better of the two).

Strategies
----------

* ``union`` — take every predicted timestamp from A and B, sort, then
  collapse anything within ``--min-gap-s`` seconds into a single peak
  (we keep the midpoint of the cluster). Best for combining
  high-recall methods.
* ``intersection`` — keep a timestamp only if methods A and B agree
  within ``--match-tol-s`` seconds. Best for boosting precision.
* ``weighted`` — same as union, but only keep clusters that include
  predictions from *both* A and B, OR an isolated prediction from one
  method that's not within ``--match-tol-s`` of any prediction in the
  other. This is the middle ground: "consensus or unique".

CLI
---

::

    # union of patch_topk and frame_diff on Charades
    python -m bench.combine \\
        --preds-a outputs/charades_sweep_patches_tighter/predictions.json \\
        --preds-b outputs/charades_framediff_sweep2/predictions.json \\
        --dataset charades \\
        --annotations data/charades/Charades/Charades_v1_test.csv \\
        --out outputs/charades_union \\
        --strategy union --min-gap-s 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .metrics import f1_grid, print_table


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------

def union_with_dedup(a: Sequence[float], b: Sequence[float], min_gap_s: float) -> List[float]:
    """Union of two sorted-or-unsorted sets, collapsing clusters within
    ``min_gap_s`` seconds into a single midpoint."""
    pts = sorted([float(x) for x in a] + [float(x) for x in b])
    if not pts:
        return []
    clusters: List[List[float]] = [[pts[0]]]
    for p in pts[1:]:
        if p - clusters[-1][-1] <= min_gap_s:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [sum(c) / len(c) for c in clusters]


def intersection_only(a: Sequence[float], b: Sequence[float], match_tol_s: float) -> List[float]:
    """Keep a timestamp only if both A and B have something within
    ``match_tol_s`` of it. Returns the midpoint of each matched pair."""
    a_sorted = sorted(float(x) for x in a)
    b_sorted = sorted(float(x) for x in b)
    out: List[float] = []
    bi = 0
    used_b = [False] * len(b_sorted)
    for ap in a_sorted:
        # find the nearest unused b
        best_j, best_d = -1, float("inf")
        for j in range(len(b_sorted)):
            if used_b[j]:
                continue
            d = abs(ap - b_sorted[j])
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= match_tol_s:
            used_b[best_j] = True
            out.append((ap + b_sorted[best_j]) / 2.0)
    return sorted(out)


def consensus_or_unique(a: Sequence[float], b: Sequence[float],
                        match_tol_s: float, min_gap_s: float) -> List[float]:
    """Weighted union:
      * agree predictions (midpoint kept)
      * isolated predictions from either side, if there's no nearby
        prediction in the other set within ``match_tol_s``.
    """
    a_sorted = sorted(float(x) for x in a)
    b_sorted = sorted(float(x) for x in b)
    used_b = [False] * len(b_sorted)
    out: List[float] = []
    for ap in a_sorted:
        best_j, best_d = -1, float("inf")
        for j in range(len(b_sorted)):
            if used_b[j]:
                continue
            d = abs(ap - b_sorted[j])
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= match_tol_s:
            used_b[best_j] = True
            out.append((ap + b_sorted[best_j]) / 2.0)   # consensus
        else:
            out.append(ap)                              # unique from A
    for j, bp in enumerate(b_sorted):
        if not used_b[j]:
            out.append(bp)                              # unique from B
    # cluster-dedup the final set
    return union_with_dedup(out, [], min_gap_s)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_combine(
    preds_a_path: str, preds_b_path: str, out_dir: str,
    *,
    dataset: str,
    annotations: str | None = None,
    labels: str | None = None,
    strategy: str = "union",
    min_gap_s: float = 0.6,
    match_tol_s: float = 1.0,
    rel_dis_grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    with open(preds_a_path) as f:
        preds_a: Dict[str, List[float]] = json.load(f)
    with open(preds_b_path) as f:
        preds_b: Dict[str, List[float]] = json.load(f)

    if dataset == "charades":
        from .charades import load_charades_labels
        if not annotations:
            sys.exit("error: --annotations required for dataset=charades")
        labels_map = load_charades_labels(annotations)
    elif dataset == "custom":
        from .custom import load_custom_labels
        if not labels:
            sys.exit("error: --labels required for dataset=custom")
        labels_map = load_custom_labels(labels, None)
    else:
        sys.exit(f"unknown dataset {dataset!r}; expected charades|custom")

    shared = sorted(set(preds_a) & set(preds_b))
    print(f"[combine] {len(shared)} videos common to both prediction sets")
    print(f"[combine] strategy={strategy}  min_gap_s={min_gap_s}  match_tol_s={match_tol_s}")

    combined: Dict[str, List[float]] = {}
    for vid in shared:
        a = preds_a[vid]; b = preds_b[vid]
        if strategy == "union":
            combined[vid] = union_with_dedup(a, b, min_gap_s)
        elif strategy == "intersection":
            combined[vid] = intersection_only(a, b, match_tol_s)
        elif strategy == "weighted":
            combined[vid] = consensus_or_unique(a, b, match_tol_s, min_gap_s)
        else:
            sys.exit(f"unknown strategy {strategy!r}")

    # Report counts: how many came from A only, B only, both
    a_only = b_only = both = 0
    for vid in shared:
        a = sorted(preds_a[vid]); b = sorted(preds_b[vid])
        # greedy match within match_tol_s
        used_b = [False] * len(b)
        for ap in a:
            best_j, best_d = -1, float("inf")
            for j, bp in enumerate(b):
                if used_b[j]:
                    continue
                d = abs(ap - bp)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= match_tol_s:
                used_b[best_j] = True
                both += 1
            else:
                a_only += 1
        b_only += sum(1 for u in used_b if not u)
    total_a = sum(len(preds_a[v]) for v in shared)
    total_b = sum(len(preds_b[v]) for v in shared)
    total_c = sum(len(combined[v]) for v in shared)
    print(f"[combine] A predictions: {total_a}  ({a_only} unique)")
    print(f"[combine] B predictions: {total_b}  ({b_only} unique)")
    print(f"[combine] agreed (within {match_tol_s}s): {both}")
    print(f"[combine] combined output:    {total_c}")
    print()

    # Score
    nonempty = [v for v in combined if labels_map.get(v, {}).get("boundaries")]
    gt = {v: labels_map[v]["boundaries"] for v in nonempty}
    dur = {v: labels_map[v].get("duration_s", 0.0) for v in nonempty}
    preds_for_eval = {v: combined[v] for v in nonempty}

    results = f1_grid(preds_for_eval, gt, dur, rel_dis_grid)
    print_table(results)

    with open(out / "predictions.json", "w") as f:
        json.dump(combined, f)

    summary = {
        "preds_a": str(preds_a_path),
        "preds_b": str(preds_b_path),
        "dataset": dataset,
        "strategy": strategy,
        "min_gap_s": min_gap_s,
        "match_tol_s": match_tol_s,
        "n_videos_common": len(shared),
        "n_evaluated": results[0]["n_videos"] if results else 0,
        "counts": {"a_total": total_a, "b_total": total_b,
                   "a_only": a_only, "b_only": b_only,
                   "agreed": both, "combined_total": total_c},
        "metrics": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-a", required=True, help="first predictions.json")
    p.add_argument("--preds-b", required=True, help="second predictions.json")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", required=True, choices=["charades", "custom"])
    p.add_argument("--annotations", default=None, help="charades CSV (charades dataset)")
    p.add_argument("--labels", default=None, help="custom labels (custom dataset)")
    p.add_argument("--strategy", choices=["union", "intersection", "weighted"], default="union")
    p.add_argument("--min-gap-s", type=float, default=0.6,
                   help="seconds within which two timestamps collapse into one (union/weighted)")
    p.add_argument("--match-tol-s", type=float, default=1.0,
                   help="seconds within which A and B count as agreeing (intersection/weighted)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_combine(
        preds_a_path=args.preds_a,
        preds_b_path=args.preds_b,
        out_dir=args.out,
        dataset=args.dataset,
        annotations=args.annotations,
        labels=args.labels,
        strategy=args.strategy,
        min_gap_s=args.min_gap_s,
        match_tol_s=args.match_tol_s,
    )

"""Plot novelty signal with detected peaks."""
from __future__ import annotations

from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_novelty(
    scores: np.ndarray,
    peak_idxs: Iterable[int],
    threshold: float,
    pts_s: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pts_s, scores, lw=1.2, label="novelty")
    ax.axhline(threshold, color="gray", ls="--", lw=0.8, label=f"threshold={threshold:.3f}")
    for p in peak_idxs:
        ax.axvline(pts_s[p], color="crimson", lw=0.8, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("novelty (1 - max cosine)")
    ax.set_title("Semantic novelty over time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

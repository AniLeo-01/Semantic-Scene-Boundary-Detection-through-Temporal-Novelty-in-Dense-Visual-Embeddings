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
    height_floor: float | None = None,
    prominence: float | None = None,
) -> None:
    """Plot the novelty signal with the height floor and detected peaks.

    If ``height_floor`` is provided, it is drawn as the dashed reference
    line instead of ``threshold`` (since the detector now uses
    ``height_floor`` as its absolute cut). The ``prominence`` value is
    reported in the legend so users can read the detector state from
    the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pts_s, scores, lw=1.2, label="novelty")

    cut = height_floor if height_floor is not None else threshold
    label = f"height floor={cut:.3f}"
    if prominence is not None:
        label += f"  |  prominence={prominence:.3f}"
    ax.axhline(cut, color="gray", ls="--", lw=0.8, label=label)

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

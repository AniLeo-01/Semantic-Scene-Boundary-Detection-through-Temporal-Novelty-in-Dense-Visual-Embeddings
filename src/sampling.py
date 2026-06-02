"""Frame sampling via PyAV. Returns RGB numpy frames at a target FPS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import av
import numpy as np


@dataclass
class SampledFrame:
    idx: int          # index in the sampled stream
    pts_s: float      # presentation timestamp in seconds
    image: np.ndarray  # HxWx3 uint8 RGB


def sample_frames(video_path: str, target_fps: float) -> Iterator[SampledFrame]:
    """Yield SampledFrame at approximately target_fps. Decodes once, drops in between."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    src_fps = float(stream.average_rate or 30)
    step = max(src_fps / float(target_fps), 1.0)

    next_decode = 0.0
    out_idx = 0
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx + 1e-9 < next_decode:
            continue
        img = frame.to_ndarray(format="rgb24")
        pts_s = float(frame.pts * stream.time_base) if frame.pts is not None else frame_idx / src_fps
        yield SampledFrame(idx=out_idx, pts_s=pts_s, image=img)
        out_idx += 1
        next_decode += step

    container.close()


def video_meta(video_path: str) -> Tuple[float, float, int]:
    """Return (duration_s, src_fps, total_frames)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 30)
    n = stream.frames or 0
    dur = float(stream.duration * stream.time_base) if stream.duration else (n / fps if n else 0.0)
    container.close()
    return dur, fps, n

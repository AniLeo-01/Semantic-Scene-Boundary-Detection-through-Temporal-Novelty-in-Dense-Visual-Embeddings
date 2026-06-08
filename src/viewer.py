"""Generate a self-contained viewer.html for a run.

The viewer contains:
  * a <video> player on the left,
  * a canvas-rendered novelty chart underneath the video — peaks marked,
    threshold + prominence drawn, with a live cursor that tracks
    ``video.currentTime`` and is also clickable to seek,
  * a right-hand keyframe panel; the card for the currently-playing
    scene is highlighted and auto-scrolled into view; clicking a card
    seeks the video to that scene's start.

All numeric data is inlined as a JS literal, so the file works over
``file://`` with no fetch/CORS issues. The video is referenced by a
relative path from the output directory; keyframes by relative paths
into ``keyframes/``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scene Boundary Viewer — __VIDEO_NAME__</title>
<style>
  :root {
    --bg: #0e1116;
    --panel: #161b22;
    --panel-2: #1f242c;
    --fg: #e6edf3;
    --fg-dim: #8b949e;
    --accent: #ff4d6d;
    --accent-2: #2dd4bf;
    --cursor: #ffd166;
    --grid: #222a33;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter,
                  "Helvetica Neue", Arial, sans-serif;
    height: 100%;
  }
  body { display: flex; flex-direction: column; }
  header {
    padding: 10px 16px; border-bottom: 1px solid var(--grid);
    display: flex; justify-content: space-between; align-items: baseline;
    background: var(--panel);
  }
  header h1 { font-size: 14px; font-weight: 600; margin: 0; }
  header .meta { font-size: 12px; color: var(--fg-dim); }
  main {
    flex: 1; min-height: 0; display: grid;
    grid-template-columns: 1fr 320px; gap: 0;
  }
  .player {
    display: flex; flex-direction: column; min-width: 0;
    border-right: 1px solid var(--grid);
  }
  .video-wrap {
    flex: 1; min-height: 0; display: flex; align-items: center;
    justify-content: center; background: #000; padding: 8px;
    position: relative;
  }
  video {
    max-width: 100%; max-height: 100%; outline: none; border-radius: 4px;
  }
  #viderr {
    display: none; position: absolute; inset: 8px;
    background: rgba(15,18,22,0.92); color: var(--fg);
    border: 1px solid var(--accent); border-radius: 4px;
    padding: 16px; font-size: 12px; overflow: auto;
  }
  #viderr h3 { margin: 0 0 6px; font-size: 13px; color: var(--accent); }
  #viderr code { background: var(--panel-2); padding: 1px 5px; border-radius: 3px; word-break: break-all; }
  .chart-wrap {
    height: 200px; padding: 8px 16px 12px; background: var(--panel);
    border-top: 1px solid var(--grid); position: relative;
  }
  .chart-wrap .legend {
    position: absolute; top: 6px; right: 16px; font-size: 11px;
    color: var(--fg-dim); display: flex; gap: 12px; pointer-events: none;
  }
  .chart-wrap .legend span::before {
    content: ""; display: inline-block; width: 10px; height: 10px;
    margin-right: 4px; vertical-align: middle; border-radius: 1px;
  }
  .chart-wrap .legend .l-nov::before { background: var(--accent-2); }
  .chart-wrap .legend .l-peak::before { background: var(--accent); }
  .chart-wrap .legend .l-cur::before { background: var(--cursor); }
  canvas { width: 100%; height: 100%; display: block; cursor: pointer; }
  .keyframes {
    background: var(--panel); overflow-y: auto; padding: 8px;
  }
  .keyframes h2 {
    font-size: 12px; font-weight: 600; color: var(--fg-dim);
    margin: 4px 8px 8px; letter-spacing: 0.5px; text-transform: uppercase;
  }
  .card {
    background: var(--panel-2); border-radius: 6px; margin-bottom: 8px;
    overflow: hidden; cursor: pointer; border: 1px solid transparent;
    transition: border-color 0.12s ease, transform 0.08s ease;
  }
  .card:hover { border-color: var(--fg-dim); }
  .card.active {
    border-color: var(--accent); box-shadow: 0 0 0 2px rgba(255,77,109,0.18);
  }
  .card img { width: 100%; height: auto; display: block; background: #000; }
  .card .meta {
    padding: 6px 10px; font-size: 12px; color: var(--fg-dim);
    display: flex; justify-content: space-between; gap: 8px;
  }
  .card .meta .scene-num { color: var(--fg); font-weight: 600; }
</style>
</head>
<body>
<header>
  <h1>__VIDEO_NAME__</h1>
  <div class="meta">__META_LINE__</div>
</header>
<main>
  <div class="player">
    <div class="video-wrap">
      <video id="vid" src="__VIDEO_REL__" controls preload="metadata"></video>
      <div id="viderr"></div>
    </div>
    <div class="chart-wrap">
      <div class="legend">
        <span class="l-nov">novelty</span>
        <span class="l-peak">peaks</span>
        <span class="l-cur">playhead</span>
      </div>
      <canvas id="chart"></canvas>
    </div>
  </div>
  <aside class="keyframes" id="keyframes">
    <h2>scenes (__N_SCENES__)</h2>
    __KEYFRAME_CARDS__
  </aside>
</main>
<script>
const DATA = __DATA_JSON__;
const vid = document.getElementById("vid");
const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");
const kfPanel = document.getElementById("keyframes");

function fitCanvas() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(r.width * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener("resize", fitCanvas);

const PAD = { l: 36, r: 8, t: 6, b: 18 };

function timeToX(t, w) {
  const span = DATA.duration || (DATA.pts[DATA.pts.length-1] || 1);
  return PAD.l + (t / span) * (w - PAD.l - PAD.r);
}
function valToY(v, h) {
  const ymax = DATA.ymax;
  return PAD.t + (1 - v / ymax) * (h - PAD.t - PAD.b);
}

function drawAxes(w, h) {
  ctx.strokeStyle = "#222a33"; ctx.lineWidth = 1;
  ctx.fillStyle = "#8b949e"; ctx.font = "10px ui-monospace, Menlo, monospace";
  // y grid: 4 lines
  for (let i = 0; i <= 4; i++) {
    const y = PAD.t + (i / 4) * (h - PAD.t - PAD.b);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(w - PAD.r, y); ctx.stroke();
    const v = DATA.ymax * (1 - i / 4);
    ctx.fillText(v.toFixed(2), 4, y + 3);
  }
  // x ticks every ~120px
  const span = DATA.duration;
  const targetTicks = Math.max(4, Math.floor((w - PAD.l - PAD.r) / 120));
  const step = niceStep(span / targetTicks);
  for (let t = 0; t <= span + 1e-9; t += step) {
    const x = timeToX(t, w);
    ctx.beginPath(); ctx.moveTo(x, h - PAD.b); ctx.lineTo(x, h - PAD.b + 3); ctx.stroke();
    ctx.fillText(fmtTime(t), x - 14, h - 4);
  }
}
function niceStep(s) {
  const pow = Math.pow(10, Math.floor(Math.log10(s)));
  const m = s / pow;
  if (m < 1.5) return pow;
  if (m < 3.5) return 2 * pow;
  if (m < 7.5) return 5 * pow;
  return 10 * pow;
}
function fmtTime(s) {
  s = Math.max(0, Math.floor(s));
  const m = Math.floor(s / 60), r = s % 60;
  return m + ":" + String(r).padStart(2, "0");
}

function drawThresholdLine(w, h, v, color, dash, label) {
  if (v == null) return;
  const y = valToY(v, h);
  ctx.save();
  ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(w - PAD.r, y); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.fillText(label, w - PAD.r - 90, y - 3);
  ctx.restore();
}

function drawNovelty(w, h) {
  const N = DATA.novelty.length;
  if (N < 2) return;
  ctx.strokeStyle = "#2dd4bf"; ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = timeToX(DATA.pts[i], w);
    const y = valToY(DATA.novelty[i], h);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawPeaks(w, h) {
  ctx.strokeStyle = "rgba(255,77,109,0.55)"; ctx.lineWidth = 1;
  for (const i of DATA.peak_idxs) {
    const x = timeToX(DATA.pts[i], w);
    ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, h - PAD.b); ctx.stroke();
  }
}

function drawCursor(w, h) {
  const t = vid.currentTime || 0;
  const x = timeToX(t, w);
  ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, h - PAD.b); ctx.stroke();
}

function draw() {
  const r = canvas.getBoundingClientRect();
  const w = r.width, h = r.height;
  ctx.clearRect(0, 0, w, h);
  drawAxes(w, h);
  drawThresholdLine(w, h, DATA.height_floor, "rgba(139,148,158,0.7)", [4, 4], "height floor");
  drawNovelty(w, h);
  drawPeaks(w, h);
  drawCursor(w, h);
}

// rAF loop — only redraws cursor section when video time changes
let lastT = -1;
function loop() {
  const t = vid.currentTime;
  if (Math.abs(t - lastT) > 1e-3) {
    lastT = t;
    draw();
    highlightActiveScene(t);
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// click on chart to seek
canvas.addEventListener("click", (e) => {
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left;
  const span = DATA.duration;
  const t = ((x - PAD.l) / (r.width - PAD.l - PAD.r)) * span;
  vid.currentTime = Math.max(0, Math.min(span, t));
});

// keyframe click → seek
for (const card of document.querySelectorAll(".card")) {
  card.addEventListener("click", () => {
    const s = parseFloat(card.dataset.start);
    vid.currentTime = s + 0.001;
    vid.play().catch(() => {});
  });
}

function highlightActiveScene(t) {
  const cards = document.querySelectorAll(".card");
  let active = -1;
  for (let i = 0; i < DATA.scenes.length; i++) {
    const s = DATA.scenes[i];
    if (t >= s.start_s && t <= s.end_s) { active = i; break; }
  }
  cards.forEach((c, i) => c.classList.toggle("active", i === active));
  if (active >= 0) {
    const card = cards[active];
    const r = card.getBoundingClientRect();
    const pr = kfPanel.getBoundingClientRect();
    if (r.top < pr.top || r.bottom > pr.bottom) {
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
}

vid.addEventListener("loadedmetadata", () => {
  if (!DATA.duration || !isFinite(DATA.duration)) DATA.duration = vid.duration;
  fitCanvas();
});

// Diagnostic overlay when the video fails to load. Shows the URL the
// browser tried so users can fix path / codec / sandbox issues fast.
vid.addEventListener("error", () => {
  const codes = { 1: "ABORTED", 2: "NETWORK", 3: "DECODE", 4: "SRC_NOT_SUPPORTED" };
  const e = vid.error || {};
  const box = document.getElementById("viderr");
  box.innerHTML =
    "<h3>Video failed to load</h3>" +
    "<div>Browser error: <strong>" + (codes[e.code] || e.code || "?") + "</strong>" +
    (e.message ? " — " + e.message : "") + "</div>" +
    "<div style='margin-top:8px'>Tried <code>" + vid.currentSrc + "</code></div>" +
    "<ul style='margin:8px 0 0 16px;padding:0'>" +
    "<li>If the URL above is relative and you opened this via Colab / IPython, the iframe sandbox can't reach it — re-run with <code>copy_video=True</code> or pass <code>video_url=&lt;a hosted URL&gt;</code>.</li>" +
    "<li>If the URL looks right, the codec may be unsupported. Re-encode: <code>ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -c:a aac out.mp4</code>.</li>" +
    "</ul>";
  box.style.display = "block";
});

fitCanvas();
</script>
</body>
</html>
"""


def _card_html(scene_idx: int, kf_rel: str, start_s: float, end_s: float) -> str:
    def fmt(t: float) -> str:
        m = int(t // 60)
        s = int(t % 60)
        return f"{m}:{s:02d}"

    return (
        f'<div class="card" data-start="{start_s:.3f}">'
        f'<img src="{kf_rel}" alt="scene {scene_idx}" loading="lazy">'
        f'<div class="meta">'
        f'<span class="scene-num">scene {scene_idx:03d}</span>'
        f'<span>{fmt(start_s)} → {fmt(end_s)}</span>'
        f'</div></div>'
    )


def build_viewer_html(
    *,
    video_path: str,
    video_relpath: str,
    novelty: Sequence[float],
    pts_s: Sequence[float],
    peak_idxs: Sequence[int],
    threshold: float,
    prominence: float,
    scenes: Sequence[dict],
    keyframe_relpaths: Sequence[str],
    duration_s: float,
    model_name: str,
    fps_sampled: float,
) -> str:
    """Return a self-contained HTML page wiring video + chart + keyframes."""
    assert len(novelty) == len(pts_s)
    assert len(scenes) == len(keyframe_relpaths)

    nov = [float(x) for x in novelty]
    ymax = max(nov + [threshold, prominence, 0.001]) * 1.15
    height_floor = float(threshold) * 0.6

    data = {
        "novelty": nov,
        "pts": [float(x) for x in pts_s],
        "peak_idxs": [int(x) for x in peak_idxs],
        "threshold": float(threshold),
        "prominence": float(prominence),
        "height_floor": height_floor,
        "duration": float(duration_s),
        "ymax": float(ymax),
        "scenes": [
            {
                "scene_idx": int(s["scene_idx"]),
                "start_s": float(s["start_s"]),
                "end_s": float(s["end_s"]),
                "keyframe_idx": int(s["keyframe_idx"]),
                "novelty_peak": float(s.get("novelty_peak", 0.0)),
            }
            for s in scenes
        ],
    }

    cards = "\n".join(
        _card_html(
            scene_idx=s["scene_idx"],
            kf_rel=kf,
            start_s=s["start_s"],
            end_s=s["end_s"],
        )
        for s, kf in zip(scenes, keyframe_relpaths)
    )

    video_name = os.path.basename(video_path)
    meta_line = (
        f'{model_name} · sampled at {fps_sampled:g} FPS · '
        f'threshold={threshold:.3f} · prominence={prominence:.3f} · '
        f'{len(scenes)} scenes'
    )

    html = _HTML_TEMPLATE
    html = html.replace("__VIDEO_NAME__", video_name)
    html = html.replace("__VIDEO_REL__", video_relpath)
    html = html.replace("__META_LINE__", meta_line)
    html = html.replace("__N_SCENES__", str(len(scenes)))
    html = html.replace("__KEYFRAME_CARDS__", cards)
    html = html.replace("__DATA_JSON__", json.dumps(data))
    return html


def write_viewer(
    out_dir: str | os.PathLike,
    video_path: str,
    novelty: Sequence[float],
    pts_s: Sequence[float],
    peak_idxs: Sequence[int],
    threshold: float,
    prominence: float,
    scenes: Sequence[dict],
    duration_s: float,
    model_name: str,
    fps_sampled: float,
    copy_video: bool = False,
    video_url: Optional[str] = None,
) -> Path:
    """Write viewer.html into ``out_dir``.

    Args:
        copy_video: If True, copy the source video into ``out_dir`` so the
            viewer can reference it by filename only. Robust against being
            moved / opened from a sandboxed iframe (Colab, IPython.display).
        video_url: Explicit ``src`` value for the ``<video>`` element.
            Overrides both ``copy_video`` and the default relative path.
            Use this when hosting the video at a known URL.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if video_url is not None:
        video_rel = video_url
    elif copy_video:
        dst = out / os.path.basename(video_path)
        if os.path.abspath(video_path) != str(dst.resolve()):
            shutil.copy2(video_path, dst)
        video_rel = os.path.basename(video_path)
    else:
        # Relative path — viewer works over file:// when the source video
        # stays at its original location relative to the output directory.
        try:
            video_rel = os.path.relpath(os.path.abspath(video_path), start=out.resolve())
        except ValueError:
            # different drives on Windows; fall back to absolute file:// URI
            video_rel = "file://" + os.path.abspath(video_path)

    keyframe_relpaths = [
        f"keyframes/scene_{s['scene_idx']:03d}.jpg" for s in scenes
    ]

    html = build_viewer_html(
        video_path=video_path,
        video_relpath=video_rel,
        novelty=novelty,
        pts_s=pts_s,
        peak_idxs=peak_idxs,
        threshold=threshold,
        prominence=prominence,
        scenes=scenes,
        keyframe_relpaths=keyframe_relpaths,
        duration_s=duration_s,
        model_name=model_name,
        fps_sampled=fps_sampled,
    )
    path = out / "viewer.html"
    path.write_text(html, encoding="utf-8")
    return path

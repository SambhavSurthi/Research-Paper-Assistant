from __future__ import annotations

import json
from typing import Any


def build_graph_html(subgraph: dict[str, list[dict[str, Any]]], height: int = 580) -> str:
    """Render an interactive, self-contained HTML5 Canvas force-directed graph viewer."""
    data_json = json.dumps(subgraph, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: Inter, system-ui, -apple-system, sans-serif; }}
  body {{ background: #0b1020; color: #e2e8f0; overflow: hidden; }}
  #wrap {{ position: relative; width: 100%; height: {height}px; border: 1px solid #1e293b; border-radius: 10px; overflow: hidden; background: radial-gradient(circle at 50% 35%, #131c36 0%, #090d1a 100%); }}
  canvas {{ width: 100%; height: 100%; display: block; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  .hud {{
    position: absolute; top: 12px; left: 12px; display: flex; gap: 8px; flex-wrap: wrap;
    background: rgba(15, 23, 42, 0.86); border: 1px solid #334155; border-radius: 8px;
    padding: 8px 12px; font-size: 12px; backdrop-filter: blur(6px);
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .controls {{
    position: absolute; top: 12px; right: 12px; display: flex; gap: 6px;
  }}
  .btn {{
    background: rgba(15, 23, 42, 0.9); color: #e2e8f0; border: 1px solid #334155;
    border-radius: 6px; padding: 6px 10px; font-size: 12px; cursor: pointer;
  }}
  .btn:hover {{ background: #1e293b; }}
  #inspector {{
    position: absolute; bottom: 12px; left: 12px; right: 12px; max-height: 150px; overflow-y: auto;
    background: rgba(15, 23, 42, 0.92); border: 1px solid #334155; border-radius: 8px;
    padding: 10px 14px; font-size: 12.5px; line-height: 1.45; display: none;
  }}
  #inspector .badge {{
    display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px;
    font-weight: 600; margin-right: 8px; color: #0b1020;
  }}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="graph"></canvas>
  <div class="hud">
    <div class="legend-item"><span class="dot" style="background:#818cf8"></span>Paper (L1)</div>
    <div class="legend-item"><span class="dot" style="background:#f59e0b"></span>Section (L1)</div>
    <div class="legend-item"><span class="dot" style="background:#22d3ee"></span>Chunk (L1)</div>
    <div class="legend-item"><span class="dot" style="background:#34d399"></span>Concept (L2)</div>
    <div class="legend-item"><span class="dot" style="background:#a78bfa"></span>Method (L2)</div>
    <div class="legend-item"><span class="dot" style="background:#fb7185"></span>Dataset (L2)</div>
    <div class="legend-item"><span class="dot" style="background:#fbbf24"></span>Metric (L2)</div>
    <div class="legend-item"><span class="dot" style="background:#4ade80"></span>Finding (L2)</div>
  </div>
  <div class="controls">
    <button class="btn" id="zoomIn">+</button>
    <button class="btn" id="zoomOut">-</button>
    <button class="btn" id="resetView">Reset</button>
  </div>
  <div id="inspector"></div>
</div>
<script>
const raw = {data_json};
const COLORS = {{
  Paper: "#818cf8",
  Section: "#f59e0b",
  Chunk: "#22d3ee",
  Concept: "#34d399",
  Method: "#a78bfa",
  Dataset: "#fb7185",
  Metric: "#fbbf24",
  Finding: "#4ade80",
  Entity: "#34d399"
}};
const RADII = {{
  Paper: 18, Section: 12, Chunk: 8,
  Concept: 11, Method: 11, Dataset: 11, Metric: 10, Finding: 11, Entity: 10
}};

const canvas = document.getElementById("graph");
const ctx = canvas.getContext("2d");
const wrap = document.getElementById("wrap");
const inspector = document.getElementById("inspector");

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = wrap.clientWidth * dpr;
  canvas.height = wrap.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}
resize();
window.addEventListener("resize", resize);

const W = wrap.clientWidth;
const H = wrap.clientHeight;
const nodes = raw.nodes.map((n, i) => {{
  const angle = (i / Math.max(1, raw.nodes.length)) * Math.PI * 2;
  const dist = n.group === "Paper" ? 30 : (n.group === "Section" ? 110 : 190);
  return {{
    ...n,
    x: W / 2 + Math.cos(angle) * dist + (Math.random() - 0.5) * 30,
    y: H / 2 + Math.sin(angle) * dist + (Math.random() - 0.5) * 30,
    vx: 0,
    vy: 0,
    r: RADII[n.group] || 9,
    color: COLORS[n.group] || "#94a3b8"
  }};
}});
const nodeMap = new Map(nodes.map(n => [n.id, n]));
const edges = raw.edges.filter(e => nodeMap.has(e.from) && nodeMap.has(e.to));

let scale = 1, offsetX = 0, offsetY = 0;
let draggingNode = null, panning = false, lastX = 0, lastY = 0, selected = null;
let ticks = 0;

function simulate() {{
  if (ticks > 280 && !draggingNode) return;
  ticks++;
  const kRepel = 1800;
  for (let i = 0; i < nodes.length; i++) {{
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {{
      const b = nodes[j];
      let dx = a.x - b.x, dy = a.y - b.y;
      let dist2 = dx * dx + dy * dy + 0.1;
      if (dist2 > 45000) continue;
      let force = kRepel / dist2;
      let inv = 1 / Math.sqrt(dist2);
      a.vx += dx * inv * force; a.vy += dy * inv * force;
      b.vx -= dx * inv * force; b.vy -= dy * inv * force;
    }}
  }}
  for (const e of edges) {{
    const a = nodeMap.get(e.from), b = nodeMap.get(e.to);
    const dx = b.x - a.x, dy = b.y - a.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const target = e.label === "HAS_CHUNK" ? 55 : (e.label === "NEXT_CHUNK" ? 42 : 90);
    const f = (dist - target) * 0.035;
    a.vx += (dx / dist) * f; a.vy += (dy / dist) * f;
    b.vx -= (dx / dist) * f; b.vy -= (dy / dist) * f;
  }}
  for (const n of nodes) {{
    n.vx += (W / 2 - n.x) * 0.004;
    n.vy += (H / 2 - n.y) * 0.004;
    if (n !== draggingNode) {{
      n.vx *= 0.82; n.vy *= 0.82;
      n.x += Math.max(-10, Math.min(10, n.vx));
      n.y += Math.max(-10, Math.min(10, n.vy));
    }}
  }}
}}

function draw() {{
  simulate();
  ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  for (const e of edges) {{
    const a = nodeMap.get(e.from), b = nodeMap.get(e.to);
    const isHot = selected && (selected.id === a.id || selected.id === b.id);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = isHot ? "#f8fafc" : (
      e.label.startsWith("SIMILAR_TO") ? "rgba(244, 114, 182, 0.55)" :
      e.label === "MENTIONS" ? "rgba(52, 211, 153, 0.3)" :
      e.label === "NEXT_CHUNK" ? "rgba(34, 211, 238, 0.45)" : "rgba(148, 163, 184, 0.28)"
    );
    ctx.lineWidth = isHot ? 1.8 : 1.0;
    ctx.stroke();

    if ((isHot || scale > 1.25) && e.label) {{
      const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      ctx.fillStyle = isHot ? "#f8fafc" : "#94a3b8";
      ctx.font = "9px Inter, sans-serif";
      ctx.fillText(e.label, mx + 3, my - 3);
    }}
  }}

  for (const n of nodes) {{
    const isSel = selected && selected.id === n.id;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r + (isSel ? 3 : 0), 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.fill();
    ctx.lineWidth = isSel ? 2.5 : 1.2;
    ctx.strokeStyle = isSel ? "#ffffff" : "#0f172a";
    ctx.stroke();

    if (n.group !== "Chunk" || scale > 1.1 || isSel) {{
      ctx.fillStyle = "#e2e8f0";
      ctx.font = (n.group === "Paper" ? "bold 11px" : "10px") + " Inter, sans-serif";
      ctx.fillText(n.label, n.x + n.r + 4, n.y + 3);
    }}
  }}

  ctx.restore();
  requestAnimationFrame(draw);
}}

function toWorld(sx, sy) {{
  return {{ x: (sx - offsetX) / scale, y: (sy - offsetY) / scale }};
}}

function hitNode(wx, wy) {{
  for (let i = nodes.length - 1; i >= 0; i--) {{
    const n = nodes[i];
    const dx = wx - n.x, dy = wy - n.y;
    if (dx * dx + dy * dy <= (n.r + 5) * (n.r + 5)) return n;
  }}
  return null;
}}

canvas.addEventListener("mousedown", ev => {{
  const rect = canvas.getBoundingClientRect();
  const w = toWorld(ev.clientX - rect.left, ev.clientY - rect.top);
  const hit = hitNode(w.x, w.y);
  lastX = ev.clientX; lastY = ev.clientY;
  if (hit) {{
    draggingNode = hit;
    selected = hit;
    showInspector(hit);
  }} else {{
    panning = true;
  }}
}});

window.addEventListener("mousemove", ev => {{
  if (draggingNode) {{
    const rect = canvas.getBoundingClientRect();
    const w = toWorld(ev.clientX - rect.left, ev.clientY - rect.top);
    draggingNode.x = w.x; draggingNode.y = w.y;
    ticks = Math.min(ticks, 200);
  }} else if (panning) {{
    offsetX += ev.clientX - lastX;
    offsetY += ev.clientY - lastY;
    lastX = ev.clientX; lastY = ev.clientY;
  }}
}});

window.addEventListener("mouseup", () => {{ draggingNode = null; panning = false; }});

canvas.addEventListener("wheel", ev => {{
  ev.preventDefault();
  const factor = ev.deltaY < 0 ? 1.12 : 0.89;
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  offsetX = mx - (mx - offsetX) * factor;
  offsetY = my - (my - offsetY) * factor;
  scale = Math.max(0.25, Math.min(4.0, scale * factor));
}}, {{ passive: false }});

document.getElementById("zoomIn").onclick = () => {{ scale = Math.min(4.0, scale * 1.2); }};
document.getElementById("zoomOut").onclick = () => {{ scale = Math.max(0.25, scale * 0.82); }};
document.getElementById("resetView").onclick = () => {{ scale = 1; offsetX = 0; offsetY = 0; }};

function showInspector(n) {{
  const deg = edges.filter(e => e.from === n.id || e.to === n.id).length;
  inspector.style.display = "block";
  inspector.innerHTML = `
    <span class="badge" style="background:${{n.color}}">${{n.group}}</span>
    <strong>${{escapeHtml(n.full_label || n.label)}}</strong>
    <span style="color:#94a3b8; margin-left:8px;">(${{deg}} connections)</span>
    <div style="margin-top:6px; color:#cbd5e1;">${{escapeHtml(n.details || "")}}</div>
  `;
}}

function escapeHtml(s) {{
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

requestAnimationFrame(draw);
</script>
</body>
</html>"""

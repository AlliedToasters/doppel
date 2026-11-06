"""One self-contained HTML page per game: stacked panels on a shared ply axis, the board at the hovered ply.

No network access and no external scripts: the page works offline and from a file:// URL. Colors: White = blue,
Black = orange, Maia-3 = aqua, Stockfish and board facts = neutral ink (validated with the dataviz palette checks;
a fourth hue failed the color-blind separation check, so engine-vs-engine panels use ink + aqua).
"""

from __future__ import annotations

import html
import json
import math

import chess.svg
import numpy as np

PIECE_ID = {s: g.split('id="', 1)[1].split('"', 1)[0] for s, g in chess.svg.PIECES.items()}

# Panels, top to bottom. kind: pos = one value per position (x = plies played), move = one value per ply (x = k+1).
PANELS = [
    {"id": "es", "title": "Expected score for White", "unit": "", "domain": [0, 1], "mid": 0.5, "events": True,
     "series": [{"key": "sf_es_w", "label": "Stockfish", "kind": "pos", "color": "ink"},
                {"key": "maia_es_w", "label": "Maia-3", "kind": "pos", "color": "maia"}]},
    {"id": "mat", "title": "Material, White minus Black", "unit": "pawns", "mid": 0, "symmetric": True,
     "series": [{"key": "material_w", "label": "Material", "kind": "pos", "color": "ink", "step": True}]},
    {"id": "vol", "title": "Volatility: rolling std of win %", "unit": "%", "min0": True,
     "series": [{"key": "wc_w_volatility", "label": "Stockfish", "kind": "pos", "color": "ink"},
                {"key": "maia_es_w_volatility", "label": "Maia-3", "kind": "pos", "color": "maia"}]},
    {"id": "sharp", "title": "Sharpness from Maia-3 win/draw/loss (capped at 50; saturates when draws are unlikely)", "unit": "", "min0": True, "sqrt": True,
     "series": [{"key": "sharpness", "label": "Sharpness", "kind": "pos", "color": "maia"}]},
    {"id": "ent", "title": "Maia-3 policy entropy before the move", "unit": "nats", "min0": True,
     "series": [{"key": "maia_entropy", "label": "White", "kind": "move", "side": 0, "color": "white"},
                {"key": "maia_entropy", "label": "Black", "kind": "move", "side": 1, "color": "black"}]},
    {"id": "clock", "title": "Clock after the move", "unit": "s", "min0": True,
     "series": [{"key": "clock_after", "label": "White", "kind": "move", "side": 0, "color": "white"},
                {"key": "clock_after", "label": "Black", "kind": "move", "side": 1, "color": "black"}]},
    {"id": "think", "title": "Think time (bars) and its moving average (line)", "unit": "s", "min0": True,
     "sqrt": True,
     "series": [{"key": "think_s", "label": "White", "kind": "move", "side": 0, "color": "white", "bar": True},
                {"key": "think_s", "label": "Black", "kind": "move", "side": 1, "color": "black", "bar": True},
                {"key": "think_s_ewma", "label": "White avg", "kind": "move", "side": 0, "color": "white",
                 "legend": False},
                {"key": "think_s_ewma", "label": "Black avg", "kind": "move", "side": 1, "color": "black",
                 "legend": False}]},
    {"id": "cpl", "title": "Centipawn loss per move (Stockfish)", "unit": "cp", "min0": True, "sqrt": True,
     "series": [{"key": "cpl", "label": "White", "kind": "move", "side": 0, "color": "white", "bar": True},
                {"key": "cpl", "label": "Black", "kind": "move", "side": 1, "color": "black", "bar": True}]},
]

# Per-move table view and hover readout fields: (key, label, format).
MOVE_FIELDS = [("san", "Move", "s"), ("think_s", "Think s", ".0f"), ("clock_after", "Clock s", ".0f"),
               ("cpl", "CPL", ".0f"), ("cls", "Class", "s"), ("maia_played_rank", "Maia rank", ".0f"),
               ("maia_p_top", "Maia top p", ".2f"), ("maia_entropy", "Entropy", ".2f")]
POS_FIELDS = [("cp_w", "Stockfish cp", ".0f"), ("maia_win_w", "Maia W", ".2f"), ("maia_draw", "Maia D", ".2f"),
              ("maia_loss_w", "Maia L", ".2f"), ("material_w", "Material", ".0f"), ("sharpness", "Sharpness", ".2f"),
              ("wc_w_volatility", "Volatility", ".1f")]
SUMMARY_FIELDS = [("elo", "Rating"), ("moves", "Moves"), ("acpl", "ACPL"), ("blunders", "Blunders"),
                  ("mistakes", "Mistakes"), ("think_s_median", "Median think s"), ("tempo_drift", "Tempo drift"),
                  ("moves_in_time_trouble", "Moves in time trouble"), ("maia_top1_rate", "Maia top-1 rate"),
                  ("maia_surprisal_mean", "Maia surprisal"), ("wc_w_worst_drawdown", "Worst drawdown %"),
                  ("shocks", "Shocks")]


def _clean(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) or math.isinf(float(v)) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _fmt(v, f):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if f == "s":
        return str(v)
    return format(v, f)


def payload(rec, pos, mov, side_tf, pos_tf, evs, summ) -> dict:
    series = {}
    for d in (pos, pos_tf, mov, side_tf):
        for k, v in d.items():
            if k not in ("fen", "uci", "san", "ply") and isinstance(v, np.ndarray) and v.dtype != object:
                series[k] = _clean(v.astype(float))
    if "wc_w" in pos:
        series["sf_es_w"] = _clean((pos["wc_w"] + 1) / 2)
    present = set(series)
    panels = []
    for p in PANELS:
        ss = [s for s in p["series"] if s["key"] in present]
        if ss:
            panels.append({**p, "series": ss})
    return {"meta": {k: rec[k] for k in ("game_id", "white", "black", "white_elo", "black_elo", "base", "inc",
                                         "result", "termination", "source")} | {"end": rec.get("end")},
            "n": len(mov["ply"]), "fens": pos["fen"], "san": mov["san"], "uci": mov["uci"],
            "mover": _clean(mov["mover"]), "cls": [str(c) for c in mov["cls"]] if "cls" in mov else None,
            "series": series, "panels": panels, "events": _clean(evs), "summary": _clean(summ),
            "move_fields": [f for f in MOVE_FIELDS if f[0] in present or f[0] in ("san", "cls")],
            "pos_fields": [f for f in POS_FIELDS if f[0] in present]}


def _summary_table(summ) -> str:
    head = "".join(f"<th scope='col'>{html.escape(s['side'].title())}</th>" for s in summ)
    rows = []
    for key, label in SUMMARY_FIELDS:
        if all(key not in s or s[key] is None for s in summ):
            continue
        cells = "".join(f"<td>{_fmt(s.get(key), '.2f' if isinstance(s.get(key), float) else 's')}</td>" for s in summ)
        rows.append(f"<tr><th scope='row'>{label}</th>{cells}</tr>")
    return f"<table class='summary'><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _move_table(mov, pos, data) -> str:
    fields = data["move_fields"] + data["pos_fields"]
    head = "".join(f"<th scope='col'>{label}</th>" for _, label, _ in fields)
    rows = []
    for k in range(len(mov["ply"])):
        cells = []
        for key, _, f in fields:
            src = mov if key in mov else pos
            v = src.get(key)
            if v is None:
                cells.append("<td></td>")
                continue
            v = v[k + 1] if src is pos else v[k]
            v = v.item() if hasattr(v, "item") else v
            cells.append(f"<td>{html.escape(_fmt(v, f))}</td>")
        who = "W" if mov["mover"][k] == 0 else "B"
        rows.append(f"<tr><th scope='row'>{k // 2 + 1}{'.' if who == 'W' else '...'}</th>{''.join(cells)}</tr>")
    return (f"<table class='moves'><thead><tr><th scope='col'>#</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def render(rec, pos, mov, side_tf, pos_tf, evs, summ) -> str:
    data = payload(rec, pos, mov, side_tf, pos_tf, evs, summ)
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    defs = "".join(chess.svg.PIECES.values())
    m = data["meta"]
    tc = f"{int(m['base'])}+{int(m['inc'])}" if m["base"] else "?"
    title = f"{m['white']} vs {m['black']}"
    sub = (f"{html.escape(m['white'])} ({m['white_elo']}) vs {html.escape(m['black'])} ({m['black_elo']}) · {tc} · "
           f"{html.escape(m['result'])} · {html.escape(str(m['end']))} · {html.escape(str(m['game_id']))}")
    return (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__SUB__", sub)
            .replace("__DEFS__", defs).replace("__PIECE_ID__", json.dumps(PIECE_ID))
            .replace("__SUMMARY__", _summary_table(data["summary"]))
            .replace("__MOVES__", _move_table(mov, pos, data)).replace("__DATA__", blob))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8984;
  --grid: #e4e3df; --white: #2a78d6; --black: #eb6834; --maia: #1baf7a; --mark-ink: #52514e;
  --sq-light: #ece4d4; --sq-dark: #b39b7c; --hl: rgba(42, 120, 214, 0.35);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242423; --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86;
    --grid: #33332f; --white: #3987e5; --black: #d95926; --maia: #199e70; --mark-ink: #c3c2b7;
    --sq-light: #a8a090; --sq-dark: #6f6352; --hl: rgba(57, 135, 229, 0.45);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86;
  --grid: #33332f; --white: #3987e5; --black: #d95926; --maia: #199e70; --mark-ink: #c3c2b7;
  --sq-light: #a8a090; --sq-dark: #6f6352; --hl: rgba(57, 135, 229, 0.45);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0 0 16px; }
.layout { display: grid; grid-template-columns: minmax(260px, 340px) minmax(0, 1fr); gap: 20px; align-items: start; }
.layout > * { min-width: 0; }
@media (max-width: 760px) { .layout { grid-template-columns: minmax(0, 1fr); } }
.side { position: sticky; top: 12px; }
@media (max-width: 760px) { .side { position: static; } }
#board { width: 100%; aspect-ratio: 1; display: block; border-radius: 4px; }
.readout { margin-top: 10px; font-variant-numeric: tabular-nums; }
.readout .mv { font-weight: 600; font-size: 15px; }
.readout dl { display: grid; grid-template-columns: auto 1fr; gap: 1px 12px; margin: 6px 0 0; }
.readout dt { color: var(--ink-2); }
.readout dd { margin: 0; }
.readout .ev { margin-top: 6px; color: var(--ink); }
.hint { color: var(--ink-3); font-size: 12px; margin-top: 8px; }
.panel { margin-bottom: 10px; }
.panel h2 { font-size: 13px; font-weight: 600; margin: 0; color: var(--ink); }
.panel .legend { display: flex; gap: 14px; font-size: 12px; color: var(--ink-2); margin: 2px 0 0; flex-wrap: wrap; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 5px; vertical-align: -1px; background: var(--c); }
svg.chart { width: 100%; height: 118px; display: block; overflow: visible; touch-action: none; }
svg.chart text { fill: var(--ink-3); font-size: 10px; font-variant-numeric: tabular-nums; }
svg.chart .grid { stroke: var(--grid); stroke-width: 1; }
svg.chart .mid { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 2 3; }
svg.chart .cross { stroke: var(--ink-2); stroke-width: 1; }
svg.chart .evline { stroke-width: 1; stroke-dasharray: 3 3; opacity: 0.7; }
svg.chart path.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg.chart .glyph { fill: var(--ink); font-size: 10px; font-weight: 700; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 12px; }
th, td { padding: 3px 8px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th[scope=row] { text-align: left; color: var(--ink-2); font-weight: 500; }
thead th { color: var(--ink-2); font-weight: 600; }
.summary { margin-top: 14px; width: 100%; }
.summary th, .summary td { white-space: normal; }
.sub { overflow-wrap: anywhere; }
details { margin-top: 18px; }
summary { cursor: pointer; color: var(--ink-2); }
.tablewrap { overflow-x: auto; max-height: 480px; overflow-y: auto; margin-top: 8px; }
.charts { min-width: 0; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>
<div class="layout">
  <div class="side">
    <svg id="board" viewBox="0 0 360 360" role="img" aria-label="Board at the selected ply">
      <defs>__DEFS__</defs><g id="squares"></g><g id="pieces"></g>
    </svg>
    <div class="readout" id="readout" aria-live="polite"></div>
    <div class="hint">Hover or drag over a chart to scrub; arrow keys step one ply.</div>
    __SUMMARY__
  </div>
  <div class="charts" id="charts"></div>
</div>
<details><summary>Table view: every move</summary><div class="tablewrap">__MOVES__</div></details>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("data").textContent);
const PIECE_ID = __PIECE_ID__;
const N = D.n, NS = "http://www.w3.org/2000/svg";
const color = c => `var(--${c === "ink" ? "mark-ink" : c})`;
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};
let cur = N;

// ---------- board ----------
const sqG = document.getElementById("squares"), pcG = document.getElementById("pieces");
for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++)
  el("rect", {x: f * 45, y: r * 45, width: 45, height: 45, fill: (r + f) % 2 ? "var(--sq-dark)" : "var(--sq-light)",
              id: `sq${f}${7 - r}`}, sqG);
const hlG = el("g", {}, sqG);
function drawBoard(i) {
  pcG.textContent = ""; hlG.textContent = "";
  if (i > 0) {
    const u = D.uci[i - 1];
    for (const s of [u.slice(0, 2), u.slice(2, 4)]) {
      const f = s.charCodeAt(0) - 97, r = +s[1] - 1;
      el("rect", {x: f * 45, y: (7 - r) * 45, width: 45, height: 45, fill: "var(--hl)"}, hlG);
    }
  }
  const rows = D.fens[i].split(" ")[0].split("/");
  rows.forEach((row, ri) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += +ch; continue; }
      el("use", {href: `#${PIECE_ID[ch]}`, transform: `translate(${f * 45},${ri * 45})`}, pcG);
      f++;
    }
  });
}

// ---------- readout ----------
const fmt = (v, f) => v === null || v === undefined || v === "" ? "–" :
  f === "s" ? String(v) : (+v).toFixed(+f.replace(/[^\d]/g, "") || 0);
function readout(i) {
  const R = document.getElementById("readout");
  let h = "";
  if (i === 0) h += `<div class="mv">Start position</div>`;
  else {
    const k = i - 1, w = D.mover[k] === 0;
    h += `<div class="mv">${Math.floor(k / 2) + 1}${w ? "." : "..."} ${D.san[k]} <span style="color:var(--ink-2);font-weight:400">(${w ? "White" : "Black"}, ply ${i})</span></div><dl>`;
    for (const [key, label, f] of D.move_fields) {
      if (key === "san") continue;
      const v = key === "cls" ? (D.cls ? D.cls[k] : null) : (D.series[key] || [])[k];
      h += `<dt>${label}</dt><dd>${fmt(v, f)}</dd>`;
    }
    h += "</dl>";
  }
  h += "<dl>";
  for (const [key, label, f] of D.pos_fields) h += `<dt>${label}</dt><dd>${fmt((D.series[key] || [])[i], f)}</dd>`;
  h += "</dl>";
  const evs = D.events.filter(e => (e.kind === "time_trouble" || e.kind === "shock" ? e.ply : e.ply + 1) === i);
  if (evs.length) h += `<div class="ev">${evs.map(e => e.label).join(" · ")}</div>`;
  R.innerHTML = h;
}

// ---------- charts ----------
const charts = [];
const M = {l: 34, r: 8, t: 6, b: 16};
function build() {
  const host = document.getElementById("charts");
  host.textContent = "";
  charts.length = 0;
  for (const p of D.panels) {
    const div = document.createElement("div"); div.className = "panel";
    div.innerHTML = `<h2>${p.title}${p.unit ? ` <span style="color:var(--ink-3);font-weight:400">(${p.unit})</span>` : ""}</h2>`;
    const leg = p.series.filter(s => s.legend !== false);
    if (leg.length > 1) div.innerHTML += `<div class="legend">${leg.map(s => `<span style="--c:${color(s.color)}">${s.label}</span>`).join("")}</div>`;
    const svg = el("svg", {class: "chart", role: "img", "aria-label": p.title});
    div.appendChild(svg); host.appendChild(div);
    charts.push({p, svg});
  }
  draw();
}
function vals(s) {
  const arr = D.series[s.key] || [];
  const out = [];
  if (s.kind === "pos") arr.forEach((v, i) => out.push([i, v]));
  else arr.forEach((v, k) => { if (s.side === undefined || D.mover[k] === s.side) out.push([k + 1, v]); });
  return out;
}
function draw() {
  for (const c of charts) {
    const {p, svg} = c;
    svg.textContent = "";
    const W = svg.clientWidth || 600, H = 118;
    const pw = W - M.l - M.r, ph = H - M.t - M.b;
    let lo, hi;
    if (p.domain) [lo, hi] = p.domain;
    else {
      const all = p.series.flatMap(s => vals(s).map(d => d[1])).filter(v => v !== null);
      lo = all.length ? Math.min(...all) : 0; hi = all.length ? Math.max(...all) : 1;
      if (p.min0) lo = 0;
      if (p.symmetric) { const a = Math.max(1, Math.abs(lo), Math.abs(hi)); lo = -a; hi = a; }
      if (hi <= lo) hi = lo + 1;
    }
    const tf = p.sqrt ? v => Math.sqrt(Math.max(0, v)) : v => v;
    const x = i => M.l + (i / Math.max(1, N)) * pw;
    const y = v => M.t + ph - ((tf(v) - tf(lo)) / (tf(hi) - tf(lo) || 1)) * ph;
    // grid and y ticks: 3 ticks at lo, mid, hi (in data units)
    const ticks = p.sqrt ? [lo, lo + (hi - lo) / 4, hi] : [lo, (lo + hi) / 2, hi];
    for (const t of ticks) {
      el("line", {class: "grid", x1: M.l, x2: W - M.r, y1: y(t), y2: y(t)}, svg);
      const lab = el("text", {x: M.l - 4, y: y(t) + 3, "text-anchor": "end"}, svg);
      lab.textContent = Math.abs(hi - lo) >= 10 ? Math.round(t) : (+t.toFixed(2));
    }
    if (p.mid !== undefined && p.mid > lo && p.mid < hi)
      el("line", {class: "mid", x1: M.l, x2: W - M.r, y1: y(p.mid), y2: y(p.mid)}, svg);
    // x ticks: move numbers
    const step = N > 120 ? 40 : N > 60 ? 20 : 10;
    for (let i = 0; i <= N; i += step) {
      const t = el("text", {x: x(i), y: H - 3, "text-anchor": "middle"}, svg);
      t.textContent = i ? `${Math.floor((i - 1) / 2) + 1}` : "0";
    }
    // event rules: shocks and time trouble, side-colored
    for (const e of D.events) if (e.kind === "shock" || e.kind === "time_trouble")
      el("line", {class: "evline", x1: x(e.ply), x2: x(e.ply), y1: M.t, y2: M.t + ph,
                  stroke: color(e.side === 0 ? "white" : "black")}, svg);
    // bars first, then lines
    const bw = Math.max(1, Math.min(8, pw / Math.max(1, N) - 2));
    for (const s of p.series.filter(s => s.bar)) for (const [i, v] of vals(s)) {
      if (v === null) continue;
      const y0 = y(Math.max(lo, 0)), y1 = y(v), top = Math.min(y0, y1), h = Math.max(1, Math.abs(y0 - y1));
      el("rect", {x: x(i) - bw / 2, y: top, width: bw, height: h, rx: Math.min(2, bw / 2), fill: color(s.color)}, svg);
    }
    for (const s of p.series.filter(s => !s.bar)) {
      let d = "", pen = false, prev = null;
      for (const [i, v] of vals(s)) {
        if (v === null) { pen = false; continue; }
        if (s.step && pen) d += `H${x(i).toFixed(1)}`;
        d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
        pen = true; prev = v;
      }
      el("path", {class: "line", d, stroke: color(s.color), "stroke-width": s.legend === false ? 1.5 : 2}, svg);
    }
    // move-class markers on the expected-score panel: dot on the curve, glyph on blunders only
    if (p.events && D.cls) {
      const key = D.series.sf_es_w ? "sf_es_w" : "maia_es_w";
      D.cls.forEach((c, k) => {
        if (!c || c === "inaccuracy") return;
        const v = D.series[key][k + 1];
        if (v === null) return;
        el("circle", {cx: x(k + 1), cy: y(v), r: 4, fill: color(D.mover[k] === 0 ? "white" : "black"),
                      stroke: "var(--surface)", "stroke-width": 2}, svg);
        if (c === "blunder") {
          const t = el("text", {class: "glyph", x: x(k + 1), y: y(v) + (v > 0.5 ? 14 : -7), "text-anchor": "middle"}, svg);
          t.textContent = "??";
        }
      });
    }
    c.cross = el("line", {class: "cross", y1: M.t, y2: M.t + ph}, svg);
    c.x = x;
    // hit area
    const hit = el("rect", {x: M.l, y: 0, width: pw, height: H, fill: "transparent"}, svg);
    const pick = ev => {
      const r = svg.getBoundingClientRect();
      set(Math.round(((ev.clientX - r.left - M.l) / pw) * N));
    };
    hit.addEventListener("pointermove", pick);
    hit.addEventListener("pointerdown", pick);
  }
  set(cur);
}
function set(i) {
  cur = Math.max(0, Math.min(N, i));
  for (const c of charts) if (c.cross) { c.cross.setAttribute("x1", c.x(cur)); c.cross.setAttribute("x2", c.x(cur)); }
  drawBoard(cur); readout(cur);
}
document.addEventListener("keydown", e => {
  if (e.key === "ArrowLeft") { set(cur - 1); e.preventDefault(); }
  if (e.key === "ArrowRight") { set(cur + 1); e.preventDefault(); }
  if (e.key === "Home") set(0);
  if (e.key === "End") set(N);
});
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
build();
})();
</script>
</body>
</html>
"""

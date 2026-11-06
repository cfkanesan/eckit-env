"""Render bench.json as a self-contained HTML report."""

import html
import json
import math
import sys

data = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
DEBUG = {(r["dim"], r["n"], r["q"], r["k"]): r for r in json.load(open(sys.argv[3]))["results"]} if len(sys.argv) > 3 else {}
meta, R = data["meta"], data["results"]

SERIES = [
    ("orig_all", "scipy KDTree, workers=-1 (original)", "s1"),
    ("orig_1", "scipy KDTree, workers=1", "s2"),
    ("new", "eckit Search (current)", "s3"),
]


def fmt_t(t):
    if t < 1e-3:
        return f"{t * 1e6:.0f} µs"
    if t < 1:
        return f"{t * 1e3:.1f} ms"
    return f"{t:.2f} s"


def fmt_n(n):
    return f"{n / 1e6:g}M" if n >= 1e6 else f"{n / 1e3:g}k"


def label(r):
    return f"{r['dim']}D · N={fmt_n(r['n'])} · Q={fmt_n(r['q'])} · k={r['k']}"


def ratio(r, key="orig_all"):
    return r["new"] / r[key]


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ---- dot plot (log x) -------------------------------------------------------
def dot_plot(rows, title_id):
    tmin = min(min(r[k] for k, _, _ in SERIES) for r in rows)
    tmax = max(max(r[k] for k, _, _ in SERIES) for r in rows)
    lo = 10 ** math.floor(math.log10(tmin))
    hi = 10 ** math.ceil(math.log10(tmax))
    left, right, top, rowh = 210, 24, 28, 26
    width = 760
    plot_w = width - left - right
    height = top + rowh * len(rows) + 30

    def x(t):
        return left + plot_w * (math.log10(t) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id}" class="chart">']
    d = lo
    while d <= hi * 1.0001:
        xx = x(d)
        out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top - 8}" y2="{height - 26}" class="grid"/>')
        out.append(f'<text x="{xx:.1f}" y="{height - 10}" class="tick" text-anchor="middle">{fmt_t(d)}</text>')
        d *= 10
    for i, r in enumerate(rows):
        y = top + rowh * i + rowh / 2
        out.append(f'<text x="{left - 12}" y="{y + 4:.1f}" class="rowlab" text-anchor="end">{html.escape(label(r))}</text>')
        xs = [x(r[k]) for k, _, _ in SERIES]
        out.append(f'<line x1="{min(xs):.1f}" x2="{max(xs):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="span"/>')
        for (k, name, cls), xx in zip(SERIES, xs):
            tip = html.escape(f"{label(r)}\n{name}: {fmt_t(r[k])}")
            out.append(
                f'<g class="pt"><circle cx="{xx:.1f}" cy="{y:.1f}" r="11" class="hit"/>'
                f'<circle cx="{xx:.1f}" cy="{y:.1f}" r="5" class="dot {cls}"/>'
                f'<title>{tip}</title></g>'
            )
    out.append("</svg>")
    return "\n".join(out)


# ---- build / query breakdown (horizontal stacked bars, linear) ------------
def breakdown(rows, title_id):
    left, right, top, rowh, barh = 210, 70, 18, 34, 18
    width = 760
    plot_w = width - left - right
    tmax = max(max(r["kd_build"] + (r["orig_1"] - r["kd_build"]), r["new"]) for r in rows)
    step = 10 ** math.floor(math.log10(tmax))
    for m in (1, 2, 5, 10):
        if tmax / (step * m) <= 6:
            step *= m
            break
    hi = math.ceil(tmax / step) * step
    height = top + rowh * len(rows) * 2 + 34

    def w(t):
        return plot_w * t / hi

    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id}" class="chart">']
    t = 0.0
    while t <= hi + 1e-9:
        xx = left + w(t)
        out.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{top - 6}" y2="{height - 28}" class="grid"/>')
        out.append(f'<text x="{xx:.1f}" y="{height - 10}" class="tick" text-anchor="middle">{t:g} s</text>')
        t += step
    i = 0
    for r in rows:
        for name, b, q, cls in (
            ("scipy, 1 thread", r["kd_build"], max(r["orig_1"] - r["kd_build"], 0), "s2"),
            ("eckit", r["ek_build"], max(r["ek_query"], 0), "s3"),
        ):
            y = top + rowh * i
            if i % 2 == 0:
                out.append(f'<text x="{left - 12}" y="{y + barh / 2 + 4:.1f}" class="rowlab" text-anchor="end">{html.escape(label(r))}</text>')
            bw, qw = w(b), w(q)
            out.append(
                f'<g class="pt"><rect x="{left}" y="{y}" width="{max(bw - 1, 0.5):.1f}" height="{barh}" class="build {cls}"/>'
                f'<title>{html.escape(f"{label(r)}, {name}: build {fmt_t(b)}")}</title></g>'
            )
            out.append(
                f'<g class="pt"><rect x="{left + bw + 1:.1f}" y="{y}" width="{max(qw - 1, 0.5):.1f}" height="{barh}" rx="3" class="dot {cls}"/>'
                f'<title>{html.escape(f"{label(r)}, {name}: queries {fmt_t(q)}")}</title></g>'
            )
            out.append(f'<text x="{left + bw + qw + 6:.1f}" y="{y + barh / 2 + 4:.1f}" class="vlab">{name} · {fmt_t(b + q)}</text>')
            i += 1
    out.append("</svg>")
    return "\n".join(out)


# ---- summary ---------------------------------------------------------------
r_all = [ratio(r) for r in R]
r_one = [ratio(r, "orig_1") for r in R]
big = [r for r in R if r["n"] == max(x["n"] for x in R)]
poi = [r for r in R if r["q"] == min(x["q"] for x in R)]
g2g = [r for r in R if r["q"] == max(x["q"] for x in R)]
agree = min(r["same_idx"] for r in R)
max_rel = max(r["max_rel"] for r in R)
ncores = len(meta["cpus"])


def tile(lbl, val, note):
    return f'<div class="tile"><div class="tl">{lbl}</div><div class="tv">{val}</div><div class="tn">{note}</div></div>'


def x_fmt(v):
    return f"{v:.2f}×" if v < 1 else f"{v:.1f}×" if v < 10 else f"{v:.0f}×"


def dbg(r):
    return DEBUG.get((r["dim"], r["n"], r["q"], r["k"]))


tiles = "".join([
    tile("eckit time vs original", x_fmt(geomean(r_all)),
         f"geometric mean over {len(R)} cases, original on {ncores} threads"),
    tile("POI cases (Q = 1k)", x_fmt(geomean([ratio(r) for r in poi])),
         "eckit time vs original, geometric mean; at parity"),
    tile("Grid-to-grid (Q = 100k)", x_fmt(geomean([ratio(r) for r in g2g])),
         "eckit time vs original, geometric mean"),
    tile("Identical neighbours", f"{agree * 100:.4f}%",
         f"worst case; max relative distance difference {max_rel:.1e}"),
])

rows_sorted = sorted(R, key=lambda r: (r["dim"], r["n"], r["q"], r["k"]))
table_rows = "\n".join(
    f"<tr><td>{r['dim']}D</td><td>{r['n']:,}</td><td>{r['q']:,}</td><td>{r['k']}</td>"
    f"<td>{fmt_t(r['orig_all'])}</td><td>{fmt_t(r['orig_1'])}</td><td>{fmt_t(r['new'])}</td>"
    f"<td>{fmt_t(r['kd_build'])}</td><td>{fmt_t(r['ek_build'])}</td>"
    f"<td class='em'>{x_fmt(ratio(r))}</td><td>{x_fmt(ratio(r, 'orig_1'))}</td>"
    f"<td>{r["same_idx"] * 100:.4f}%</td>"
    f"<td>{fmt_t(dbg(r)['new']) if dbg(r) else '–'}</td></tr>"
    for r in rows_sorted
)

legend = "".join(f'<span class="key"><i class="sw {c}"></i>{html.escape(n)}</span>' for _, n, c in SERIES)
dot2 = dot_plot([r for r in rows_sorted if r["dim"] == 2], "c1t")
dot3 = dot_plot([r for r in rows_sorted if r["dim"] == 3], "c2t")
bd = breakdown([r for r in rows_sorted if r["n"] == max(x["n"] for x in R) and r["dim"] == 2 and r["k"] in (1, 36)], "c3t")

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nn_lookup Benchmark</title>
<style>
/* Layout: single reading column, wide figures scroll in their own frame */
:root {{
  --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; color-scheme: dark; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--sans); }}
main {{ max-width: 920px; margin: 0 auto; padding-block: 40px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 32px; }}
h1 {{ font-size: 28px; line-height: 1.2; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }}
h2 {{ font-size: 18px; margin: 0; text-wrap: balance; }}
p {{ margin: 0; max-width: 68ch; color: var(--ink-2); }}
code {{ font-family: var(--mono); font-size: 0.9em; }}
header, section {{ display: grid; gap: 12px; min-width: 0; }}
.meta {{ font-size: 13px; color: var(--muted); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.tile {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 14px 16px; display: grid; gap: 4px; align-content: start; }}
.tl {{ font-size: 13px; color: var(--ink-2); }}
.tv {{ font-size: 28px; font-weight: 600; }}
.tn {{ font-size: 12px; color: var(--muted); }}
.figure {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 16px; display: grid; gap: 10px; min-width: 0; }}
.figure h3 {{ margin: 0; font-size: 14px; }}
.scroll {{ overflow-x: auto; }}
.chart {{ width: 100%; min-width: 640px; height: auto; display: block; font-family: var(--sans); }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.rowlab {{ fill: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }}
.vlab {{ fill: var(--ink-2); font-size: 11px; font-variant-numeric: tabular-nums; }}
.span {{ stroke: var(--axis); stroke-width: 2; stroke-linecap: round; }}
.hit {{ fill: transparent; }}
.dot {{ stroke: var(--surface); stroke-width: 2; }}
.dot.s1, .sw.s1 {{ fill: var(--s1); background: var(--s1); }}
.dot.s2, .sw.s2 {{ fill: var(--s2); background: var(--s2); }}
.dot.s3, .sw.s3 {{ fill: var(--s3); background: var(--s3); }}
.build {{ fill-opacity: 0.35; }}
.build.s2 {{ fill: var(--s2); }} .build.s3 {{ fill: var(--s3); }}
.pt:hover .dot {{ stroke: var(--ink); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2); }}
.key {{ display: inline-flex; align-items: center; gap: 6px; }}
.sw {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.sw.sq {{ border-radius: 2px; }}
.sw.light {{ opacity: 0.35; }}
table {{ border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; min-width: 820px; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
th {{ color: var(--ink-2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }}
td:first-child, th:first-child {{ text-align: left; }}
td.em {{ font-weight: 600; }}
ul {{ margin: 0; padding-left: 20px; color: var(--ink-2); max-width: 68ch; display: grid; gap: 6px; }}
</style>
</head>
<body>
<main>
<header>
  <h1>nn_lookup: scipy KDTree vs eckit Search</h1>
  <p>Time for <code>nn_lookup(grid_coords, points, k)</code> in <code>grid_interp.py</code>, before (scipy
  <code>KDTree</code>) and after (eckit <code>Search.from_xyz</code>). Each run builds the tree and queries every point,
  as <code>nn_lookup</code> does on a cache miss.</p>
  <div class="meta">{html.escape(meta['cpu'])} · pinned to {ncores} cores ({meta['cpus'][0]}–{meta['cpus'][-1]}) ·
  Python {meta['python']} · numpy {meta['numpy']} · scipy {meta['scipy']} · eckit 2.5.0 develop (Release build) ·
  host load {meta['load'][0]:.0f}/256 · {meta['date']}</div>
</header>

<div class="tiles">{tiles}</div>

<section>
  <h2>Time per case</h2>
  <p>Log scale. N grid points (irregular ~1 km lattice, float32), Q query points, k neighbours. 3D cases include
  altitude × 25, as in <code>interp_temp</code>. Hover a dot for its value.</p>
  <div class="legend">{legend}</div>
  <div class="figure"><h3 id="c1t">2D coordinates (nearest, idw, upscaled, linear neighbour lookups)</h3><div class="scroll">{dot2}</div></div>
  <div class="figure"><h3 id="c2t">3D coordinates (temp, nearest_3D, idw_3D, upscaled_3D)</h3><div class="scroll">{dot3}</div></div>
</section>

<section>
  <h2>Where the time goes at N = {fmt_n(max(x['n'] for x in R))}</h2>
  <p>Single-threaded scipy vs eckit, split into tree build (light) and queries (solid).</p>
  <div class="legend"><span class="key"><i class="sw sq light s2"></i><i class="sw sq s2"></i>scipy, 1 thread: build · queries</span>
  <span class="key"><i class="sw sq light s3"></i><i class="sw sq s3"></i>eckit: build · queries</span></div>
  <div class="figure"><h3 id="c3t">2D, k = 1 and 36</h3><div class="scroll">{bd}</div></div>
</section>

<section>
  <h2>Reading the numbers</h2>
  <ul>
    <li>The Release build fixed the tree build: {fmt_t(geomean([r['ek_build'] for r in big]))} for eckit vs {fmt_t(geomean([r['kd_build'] for r in big]))} for scipy at N = {fmt_n(big[0]['n'])}. The earlier Debug build took about 3 s. Overall eckit is {x_fmt(geomean([dbg(r)['new'] / r['new'] for r in R if dbg(r)]))} faster than with Debug (geometric mean; last column of the table).</li>
    <li>With few query points (POIs) the tree build dominates, so eckit and the original are at parity: {x_fmt(geomean([ratio(r) for r in poi]))} the original's time on average.</li>
    <li>With many query points eckit is slower, and the gap grows with k: at Q = 100k it takes {x_fmt(geomean([ratio(r) for r in g2g if r['k'] == 1]))} the original's time at k = 1 and {x_fmt(geomean([ratio(r) for r in g2g if r['k'] == 36]))} at k = 36. Against single-threaded scipy the factors are {x_fmt(geomean([ratio(r, 'orig_1') for r in g2g if r['k'] == 1]))} and {x_fmt(geomean([ratio(r, 'orig_1') for r in g2g if r['k'] == 36]))}.</li>
    <li>Three causes, roughly in order: the original queries on all cores ({ncores} here, up to 256 on the full node) while eckit queries run on one thread; eckit's k-d tree query is slower per thread than scipy's at large k; and the per-point Python loop. A separate single-thread check (1M points, 100k queries) put eckit's own C++ batch query at 0.49 s (k = 4) and 2.05 s (k = 36), with the Python loop about 25% slower than that.</li>
    <li>Results agree: neighbour indices identical in at least {agree * 100:.4f}% of queries, distances within {max_rel:.1e} relative.</li>
  </ul>
</section>

<section>
  <h2>All results</h2>
  <div class="figure scroll"><table>
    <thead><tr><th>Dim</th><th>N</th><th>Q</th><th>k</th><th>scipy -1</th><th>scipy 1</th><th>eckit</th>
    <th>KDTree build</th><th>eckit build</th><th>eckit / scipy -1</th><th>eckit / scipy 1</th><th>Same idx</th><th>eckit Debug</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table></div>
  <p class="meta">Best of 3 runs for small cases, 1 run for large ones. Ratios above 1× mean eckit is slower.</p>
</section>
</main>
</body>
</html>
"""
open(OUT, "w").write(page)
print("wrote", OUT)

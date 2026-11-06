"""Benchmark nn_lookup: original (scipy KDTree) vs current (eckit Search)."""

import ast
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.spatial import KDTree
from eckit.geo import Search

ORIG, NEW, OUT = sys.argv[1:4]


def load(path, names, ns):
    mod = ast.parse(open(path).read())
    body = [
        n
        for n in mod.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))
    ]
    exec(compile(ast.Module(body, []), path, "exec"), ns)
    return ns


orig = load(ORIG, {"nn_lookup"}, {"np": np, "KDTree": KDTree})["nn_lookup"]
src1 = open(ORIG).read().replace("workers=-1", "workers=1")
orig1_ns = {"np": np, "KDTree": KDTree}
exec(compile(ast.Module([n for n in ast.parse(src1).body
                         if isinstance(n, ast.FunctionDef) and n.name == "nn_lookup"], []), "o1", "exec"), orig1_ns)
orig1 = orig1_ns["nn_lookup"]
new_ns = load(NEW, {"nn_lookup", "_to_xyz", "_PLANE_BASIS"}, {"np": np, "Search": Search})
new = new_ns["nn_lookup"]
to_xyz = new_ns["_to_xyz"]

rng = np.random.default_rng(42)
SPACING = 1000.0  # m, ~ICON-CH1 resolution
Z_FACT = 25


def make_grid(n, dim):
    side = int(np.ceil(np.sqrt(n)))
    x, y = np.meshgrid(np.arange(side), np.arange(side))
    xy = np.c_[x.ravel(), y.ravel()][:n] * SPACING + 2.48e6
    xy += rng.normal(0, SPACING * 0.05, xy.shape)  # irregular (avoid ties)
    extent = side * SPACING
    if dim == 2:
        return xy.astype(np.float32), extent
    alt = 1500 + 1000 * np.sin(xy[:, 0] / 2e4) * np.cos(xy[:, 1] / 3e4)
    return np.c_[xy, alt * Z_FACT].astype(np.float32), extent


def make_points(q, dim, extent):
    xy = rng.random((q, 2)) * extent + 2.48e6
    if dim == 2:
        return xy.astype(np.float32)
    alt = rng.random(q) * 3000
    return np.c_[xy, alt * Z_FACT].astype(np.float32)


def timeit(f, repeat):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


CASES = []
for dim in (2, 3):
    for n in (10_000, 100_000, 1_100_000):
        for q in (1_000, 100_000):
            for k in ((1, 4, 36) if dim == 2 else (1, 36)):
                CASES.append((dim, n, q, k))

results = []
for dim, n, q, k in CASES:
    grid, extent = make_grid(n, dim)
    pts = make_points(q, dim, extent)
    rep = 3 if n * 1.0 + q * k * 20 < 3e6 else 1

    t_orig, (d0, i0) = timeit(lambda: orig(grid, pts, k), rep)
    t_orig1, _ = timeit(lambda: orig1(grid, pts, k), rep)
    t_new, (d1, i1) = timeit(lambda: new(grid, pts, k), rep)

    # phases
    t_kd_build, tree = timeit(lambda: KDTree(grid), rep)
    t_ek_build, search = timeit(lambda: Search.from_xyz(*to_xyz(grid)), rep)
    t_ek_query = t_new - t_ek_build

    same_idx = float(np.mean(i0 == i1))
    max_rel = float(np.nanmax(np.abs(d0.astype(np.float64) - d1) / np.maximum(d0, 1e-9)))
    r = dict(dim=dim, n=n, q=q, k=k, rep=rep,
             orig_all=t_orig, orig_1=t_orig1, new=t_new,
             kd_build=t_kd_build, ek_build=t_ek_build, ek_query=t_ek_query,
             same_idx=same_idx, max_rel=max_rel)
    results.append(r)
    print(json.dumps(r), flush=True)

meta = dict(
    cpus=sorted(os.sched_getaffinity(0)),
    cpu=next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), ""),
    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
    host=platform.node(), date=time.strftime("%Y-%m-%d %H:%M"),
    load=os.getloadavg(),
)
json.dump(dict(meta=meta, results=results), open(OUT, "w"), indent=1)

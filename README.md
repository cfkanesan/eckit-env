# eckit-env

Development environment for eckit's `eckit.geo.Search` Python API, and its use in
`grid_process/grid_interp.py`: the temperature interpolations `interp_temp` and
`interp_temp_fixgrad`, with their nearest-neighbour searches done by eckit.

- `eckit/`: eckit (submodule, branch `feature/geo-search-batch` of
  [cfkanesan/eckit](https://github.com/cfkanesan/eckit)), built by spack from this checkout
- `spack.yaml`, `repos/`: spack environment and the local package recipes it uses
- `grid_process/`: `grid_interp.py` (eckit version), `grid_interp_scipy.py` (original
  scipy version), tests, and the benchmark comparing both
- `requirements.txt`: Python packages for the venv

## Getting started from a fresh clone

### Prerequisites

- [spack](https://spack.io) (tested with 1.2.0), with a C++ compiler it knows about
- SSH access to GitHub (the `eckit` submodule URL is `git@github.com:cfkanesan/eckit.git`)

### 1. Clone with the eckit submodule

```bash
git clone --recurse-submodules https://github.com/cfkanesan/eckit-env.git
cd eckit-env
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init`.

### 2. Point spack at the local package repo

`spack.yaml` lists the local package repo by absolute path
(`/users/ckanesan/eckit-env/repos/spack_repo/eckit_dev`). If your clone is anywhere
else, change that line to a path relative to the environment:

```yaml
  repos:
    eckit_dev: $env/repos/spack_repo/eckit_dev
```

### 3. Build eckit and its Python bindings with spack

```bash
. /path/to/spack/share/spack/setup-env.sh
spack -e . concretize -f
spack -e . install -j 16
```

`spack.lock` and `.spack-env/` are not in git, so the first `concretize` resolves
everything from scratch. The build of eckit uses the `./eckit` checkout (a `develop`
spec), in Release mode (`build_type=Release` in `spack.yaml`).

On a shared machine, keep `-j` modest: without it, spack uses every core.

The environment view (`.spack-env/view`) then holds `libeckit.so`, the Python
interpreter (3.14) and the `eckit` Python package.

### 4. Create the venv

The venv is created from the view's Python, the interpreter the eckit bindings are
built for, so the built package can be linked in without compiling anything.

```bash
.spack-env/view/bin/python -m venv venv
venv/bin/pip install -r requirements.txt
ln -s "$PWD/.spack-env/view/lib/python3.14/site-packages/eckit" venv/lib/python3.14/site-packages/eckit
echo "export ECKIT_DIR=$PWD/.spack-env/view" >> venv/bin/activate
```

- `import eckit` finds `libeckit.so` with `findlibs`. That works inside the spack view
  (the library is under `sys.prefix`) but not in the venv, hence `ECKIT_DIR`. Without it
  the import fails with `ModuleNotFoundError: unable to find eckitlib.eckit`. If you run
  `venv/bin/python` without activating the venv, export `ECKIT_DIR` yourself.
- `pyyaml` (in `requirements.txt`) is needed by eckit to build a search.
- `scipy` is only needed by the benchmark (for the original version,
  `grid_interp_scipy.py`). `grid_interp.py` needs only numpy and eckit.

### 5. Check the setup

```bash
source venv/bin/activate
python -c "import eckit.geo; print(eckit.geo.Search)"
```

Then run the tests:

```bash
cd grid_process
python -m pytest
```

Expected: `6 passed`.

## Benchmark: eckit Search vs scipy KDTree

`grid_process/bench.py` times `nn_lookup` from two versions of `grid_interp.py` (tree
build and queries, 2D and 3D, grids of 10k to 1.1M points, 1k or 100k query points,
k = 1, 4, 36) and checks that they return the same neighbours. `report.py` renders the
results as an HTML page (`nn_lookup_benchmark.html` holds the latest results).

```bash
cd grid_process
taskset -c 96-103 ../venv/bin/python bench.py grid_interp_scipy.py grid_interp.py bench.json
../venv/bin/python report.py bench.json nn_lookup_benchmark.html [previous_bench.json]
```

- `taskset` pins the run to 8 cores: the scipy version queries with `workers=-1`, which
  otherwise uses every core of the machine. Pick cores that are free.
- The full run takes a few minutes with a Release build of eckit (much longer with Debug).
- The optional third argument of `report.py` is an earlier results file, shown in the
  table's "eckit Debug" column.

## Working on eckit

Edit the code in `eckit/`, then rebuild and reinstall with `spack -e . install`. The
venv link points into the spack view, so it picks up the new build. If the Python
version of the spack environment changes, the `cpython-314` extensions no longer load:
recreate the venv and the link.

Alternatively, build the bindings into the venv from source with
`pip install ./eckit/python/eckit`. This requires Cython and the eckit headers and
libraries to be found at build time.

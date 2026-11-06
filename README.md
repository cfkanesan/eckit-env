# eckit-env

Development environment for eckit (spack environment, with eckit built from the
`./eckit` checkout) and a Python venv for `grid_interp.py`.

## Using the eckit Python bindings in the venv

The venv (`venv/`) is created from the spack view's Python (3.14.5), the same
interpreter the eckit bindings are built for, so the already built package can be
linked in, without compiling anything.

### 1. Link the package and install its missing dependency

```bash
cd /users/ckanesan/eckit-env
ln -s "$PWD/.spack-env/view/lib/python3.14/site-packages/eckit" venv/lib/python3.14/site-packages/eckit
venv/bin/pip install pyyaml
```

`pyyaml` is needed by eckit to build a search (spec parsing), and is not otherwise
installed in the venv.

### 2. Tell findlibs where `libeckit.so` is

`import eckit` locates the C++ library with `findlibs`. This works inside the spack
view (the library is under `sys.prefix`), but not in the venv. Set `ECKIT_DIR`, for
example in the venv activation script:

```bash
echo "export ECKIT_DIR=$PWD/.spack-env/view" >> venv/bin/activate
source venv/bin/activate
python -c "import eckit.geo; print(eckit.geo.Search)"
```

Without it, the import fails with:

```
ModuleNotFoundError: unable to find eckitlib.eckit
```

If you run `venv/bin/python` without activating the venv, export `ECKIT_DIR`
yourself (e.g. in your shell profile).

### Notes

- The link points to the spack view, so it follows rebuilds of the develop checkout
  of eckit by spack.
- The extensions are built for CPython 3.14 (`cpython-314`). If the Python version
  of the spack environment changes, recreate the venv and the link.
- Alternatively, build the bindings from source with `pip install ./eckit/python/eckit`.
  This requires Cython and the eckit headers and libraries to be found at build time.

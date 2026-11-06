# Local overlay of the builtin eckit recipe that adds Python (Cython) bindings and PROJ.

from spack_repo.builtin.packages.eckit.package import Eckit as BuiltinEckit

from spack.package import *


class Eckit(BuiltinEckit):
    variant("python", default=True, description="Build Python bindings (Cython)")
    variant("proj", default=False, description="Enable PROJ-based projections (eckit::geo)")

    depends_on("proj@9.1.1:", when="+proj")

    with when("+python"):
        extends("python")
        depends_on("python@3.9:", type=("build", "link", "run"))
        depends_on("py-cython@3:", type="build")
        depends_on("py-pyyaml", type="run")
        # eckit/__init__.py uses findlibs.load(), added in findlibs 0.1.1
        depends_on("py-findlibs@0.1.1:", type="run")
        # numpy arrays (batch searches, grid coordinates)
        depends_on("py-numpy", type="run")

    def setup_run_environment(self, env):
        # Let findlibs locate libeckit even outside the environment view
        env.set("ECKIT_HOME", self.prefix)

    def cmake_args(self):
        args = super().cmake_args()
        args.append(self.define_from_variant("ENABLE_PYTHON", "python"))
        args.append(self.define_from_variant("ENABLE_PROJ", "proj"))
        if self.spec.satisfies("+python"):
            args.append(self.define("Python3_EXECUTABLE", self.spec["python"].command.path))
        return args

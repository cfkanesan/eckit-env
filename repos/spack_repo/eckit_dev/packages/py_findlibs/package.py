# Local overlay of the builtin py-findlibs recipe adding newer releases.
# eckit's Python bindings need findlibs.load(), which first appeared in 0.1.1.

from spack_repo.builtin.packages.py_findlibs.package import PyFindlibs as BuiltinPyFindlibs

from spack.package import *


class PyFindlibs(BuiltinPyFindlibs):
    version("0.1.4", sha256="4aa4637d4f18d25a877c8ef16d1585195a4eb7360e0949fd59600b4d9b367a3b")
    version("0.1.3", sha256="49bbe509c8b439ecd9c0d021c301aa9db643a2e6ab4e189a40b505ee4a49db62")
    version("0.1.2", sha256="1f56d220c69686392ebdc4c65b32ee344818bca633643a8c97592957d1728122")
    version("0.1.1", sha256="dc7a0d4fe2bb6635f1e4bdcde60aecec42b6755c4783c2f629069c2f9321d732")

    depends_on("python@3.10:", type=("build", "run"), when="@0.1.3")
    depends_on("python@3.11:", type=("build", "run"), when="@0.1.4:")

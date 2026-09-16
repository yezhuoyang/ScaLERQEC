import os
import sys
import platform
import setuptools
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Base include dirs: pybind11 + your source tree
include_dirs = [
    pybind11.get_include(),
    "QEPG",
    "QEPG/src",
]

extra_compile_args = []
extra_link_args = []


if sys.platform == "win32":
    # --- Windows flags ---
    extra_compile_args = ["/std:c++20", "/EHsc", "/O2"]
    # MSVC's standard OpenMP runtime is redistributable with normal wheels.
    if os.environ.get("SCALERQEC_NO_OPENMP") != "1":
        extra_compile_args += ["/openmp"]
    extra_link_args = []

elif sys.platform == "darwin":
    # --- macOS flags ---
    # OpenMP is auto-detected from Homebrew libomp for local builds.
    # Disabled in wheel builds (cibuildwheel sets CIBUILDWHEEL=1) to avoid
    # libomp/delocate version-target conflicts. Can also be forced off/on:
    #   SCALERQEC_NO_OPENMP=1 pip install .   (force disable)
    #   SCALERQEC_OPENMP=1 pip install .      (force enable)
    extra_compile_args = ["-std=c++20", "-O3"]
    extra_link_args = []

    in_cibuildwheel = os.environ.get("CIBUILDWHEEL", "") == "1"
    force_off = os.environ.get("SCALERQEC_NO_OPENMP", "") == "1"
    force_on = os.environ.get("SCALERQEC_OPENMP", "") == "1"
    homebrew_prefix = os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew")
    omp_inc = os.path.join(homebrew_prefix, "opt", "libomp", "include")
    omp_lib = os.path.join(homebrew_prefix, "opt", "libomp", "lib")

    # Auto-detect: enable if libomp is installed and not in cibuildwheel
    use_openmp = force_on or (
        not in_cibuildwheel and not force_off and os.path.isdir(omp_inc)
    )

    if use_openmp:
        extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
        extra_link_args += ["-lomp"]
        if os.path.isdir(omp_inc):
            extra_compile_args.append(f"-I{omp_inc}")
        if os.path.isdir(omp_lib):
            extra_link_args.append(f"-L{omp_lib}")

else:
    # --- Linux flags ---
    extra_compile_args = ["-std=c++20", "-O3"]
    if os.environ.get("SCALERQEC_NO_OPENMP") != "1":
        extra_compile_args += ["-fopenmp"]
        extra_link_args = ["-fopenmp"]

# Portable wheels use SSE2/NEON/scalar dispatch. AVX2 is an explicit local opt-in.
if os.environ.get("SCALERQEC_NATIVE") == "1" and platform.machine().lower() in {"amd64", "x86_64"}:
    extra_compile_args += ["/arch:AVX2"] if sys.platform == "win32" else ["-mavx2", "-mbmi2"]


ext_modules = [
    Pybind11Extension(
        "scalerqec.qepg",
        [
            "QEPG/bindings.cpp",
            "QEPG/src/QEPG.cpp",
            "QEPG/src/clifford.cpp",
            "QEPG/src/sampler.cpp",
            "QEPG/src/LERcalculator.cpp",
            "QEPG/src/hotspot.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]


setuptools.setup(
    name="scalerqec",
    description="Scalable Quantum Error Correction testing Tools for logical error rate and software correctness",
    author="John Ye",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

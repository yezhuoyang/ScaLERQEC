import os
import sys
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


def _maybe_add_include(root: str, header_subpath: str) -> bool:
    """
    If `root/header_subpath` exists, add `root` to include_dirs (if not already)
    and return True. Otherwise return False.
    """
    if not root:
        return False
    full = os.path.join(root, header_subpath)
    if os.path.exists(full):
        if root not in include_dirs:
            include_dirs.append(root)
        return True
    return False


if sys.platform == "win32":
    # --- Windows flags & includes ---
    extra_compile_args = ["/std:c++20", "/EHsc", "/O2", "/openmp:llvm"]
    extra_link_args = ["/DEBUG"]

    win_includes = [
        r"C:\local\boost_1_87_0",
        r"C:\vcpkg\installed\x64-windows\include"
    ]
    for path in win_includes:
        if os.path.isdir(path) and path not in include_dirs:
            include_dirs.append(path)

else:
    # --- macOS / Linux flags ---
    extra_compile_args = ["-std=c++20", "-O3"]

    # 1) Try to find Boost (for boost/dynamic_bitset.hpp)
    boost_roots = [
        os.environ.get("BOOST_ROOT"),
        "/opt/homebrew/include",   # Homebrew on Apple Silicon
        "/usr/local/include",      # common on macOS / local installs
        "/usr/include",            # common on Linux
    ]
    for root in boost_roots:
        if _maybe_add_include(root, os.path.join("boost", "dynamic_bitset.hpp")):
            break

    # 2) Try to find Eigen
    #
    # If your code does:   #include <Eigen/Dense>
    # then we want include_dirs to contain a directory that has "Eigen/" inside.
    # On many systems, that's /usr/include/eigen3 or /usr/local/include/eigen3.
    eigen_roots = [
        os.environ.get("EIGEN_ROOT"),
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
        "/opt/homebrew/include",
        "/usr/local/include",
        "/usr/include",
    ]
    for root in eigen_roots:
        if _maybe_add_include(root, os.path.join("Eigen", "Dense")):
            break


ext_modules = [
    Pybind11Extension(
        "scaler.qepg",
        [
            "QEPG/bindings.cpp",
            "QEPG/src/QEPG.cpp",
            "QEPG/src/clifford.cpp",
            "QEPG/src/sampler.cpp",
            "QEPG/src/LERcalculator.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]


setuptools.setup(
    name="scaler",
    version="0.0.1",
    description="ScaLER + QEPG backend",
    author="John Ye",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

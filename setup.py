import os
import sys
import setuptools
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext


# -----------------------------------------------
# 1. Detect Boost include path cross-platform
# -----------------------------------------------

def detect_boost_include():
    """Return a list of possible Boost include paths."""
    candidates = []

    # User-specified
    if "BOOST_ROOT" in os.environ:
        candidates.append(os.path.join(os.environ["BOOST_ROOT"]))

    # Windows vcpkg
    candidates += [
        r"C:/local/boost_1_87_0",
        r"C:/local/boost_1_86_0",
        r"C:/vcpkg/installed/x64-windows/include",
    ]

    # macOS Homebrew (Apple Silicon and Intel)
    candidates += [
        "/opt/homebrew/include",
        "/usr/local/include",
    ]

    # Linux common locations
    candidates += [
        "/usr/include",
        "/usr/local/include",
    ]

    # Filter only valid paths
    valid = []
    for path in candidates:
        if path is None:
            continue
        hdr = os.path.join(path, "boost", "dynamic_bitset.hpp")
        if os.path.exists(hdr):
            valid.append(path)

    return valid


boost_include_paths = detect_boost_include()

if not boost_include_paths:
    print("WARNING: Boost not found, will likely fail to build.")
else:
    print("Found Boost include paths:", boost_include_paths)


# -----------------------------------------------
# 2. Compiler flags
# -----------------------------------------------

extra_compile_args = []
extra_link_args = []

if sys.platform.startswith("win"):
    extra_compile_args = ["/std:c++20", "/EHsc", "/O2"]
    extra_link_args = []
else:
    extra_compile_args = ["-std=c++20", "-O3"]


# -----------------------------------------------
# 3. Pybind11 Extension
# -----------------------------------------------

ext_modules = [
    Pybind11Extension(
        "scaler.qepg",
        sources=[
            "QEPG/bindings.cpp",
            "QEPG/src/QEPG.cpp",
            "QEPG/src/clifford.cpp",
            "QEPG/src/sampler.cpp",
            "QEPG/src/LERcalculator.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            "QEPG",
            "QEPG/src",
        ] + boost_include_paths,  # <--- Add all valid boost paths
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]


# -----------------------------------------------
# 4. Setup
# -----------------------------------------------

setuptools.setup(
    name="scaler",
    version="0.0.1",
    author="John Ye",
    description="ScaLER + QEPG backend (PyBind11 C++ extension)",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

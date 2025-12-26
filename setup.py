import os
import sys
import subprocess

import setuptools
import pybind11
from pybind11.setup_helpers import Pybind11Extension
from setuptools.command.build_ext import build_ext as _build_ext

# ---------------------------------------------------------------------------
# Helper: build_ext that knows how to compile .cu files with NVCC
# ---------------------------------------------------------------------------


cuda_root = os.environ.get(
    "CUDA_PATH",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)
cuda_lib_dir = os.path.join(cuda_root, "lib", "x64")



libraries = []
library_dirs = []

if sys.platform == "win32":
    # Link against CUDA runtime on Windows
    library_dirs.append(cuda_lib_dir)
    libraries.append("cudart")
else:
    # Typical Linux CUDA layout
    cuda_lib_dir_linux = os.path.join(cuda_root, "lib64")
    library_dirs.append(cuda_lib_dir_linux)
    libraries.append("cudart")



class BuildExtWithCUDA(_build_ext):
    """
    Custom build_ext that:
      - Compiles all .cu sources with NVCC into .obj files.
      - Removes .cu from ext.sources.
      - Adds the resulting .obj files to ext.extra_objects so the normal
        MSVC link step will link them in.
    """

    def build_extensions(self):
        for ext in self.extensions:
            cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
            if not cuda_sources:
                continue

            # Remaining C++ sources (handled by the usual flow)
            cpp_sources = [s for s in ext.sources if not s.endswith(".cu")]
            ext.sources = cpp_sources

            # Where to put intermediate .obj files
            build_temp = os.path.abspath(self.build_temp)
            os.makedirs(build_temp, exist_ok=True)

            extra_objects = getattr(ext, "extra_objects", [])

            # Path to NVCC (adjust if your CUDA version/path differs)
            # You can also set env var NVCC to override.
            nvcc_path = os.environ.get(
                "NVCC",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
            )

            # Common NVCC flags
            if sys.platform == "win32":
                host_flags = "/MD,/O2,/EHsc"
                
                cl_exe = (
                    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
                    r"\VC\Tools\MSVC\14.41.34120\bin\HostX86\x64\cl.exe"
                )

                nvcc_common = [
                    nvcc_path,
                    "-ccbin", cl_exe,           # <<< tell nvcc where cl.exe is
                    "-c",
                    "-Xcompiler=" + host_flags,
                    "-std=c++17",
                    "-DBOOST_NO_CXX14_CONSTEXPR",  # <<< important for Boost on MSVC
                ]
            else:
                nvcc_common = [
                    nvcc_path,
                    "-c",
                    "-std=c++17",
                    "-O3",
                ]

            # Compile each .cu into an .obj
            for src in cuda_sources:
                src_abs = os.path.abspath(src)
                obj_name = os.path.splitext(os.path.basename(src))[0] + ".obj"
                obj_path = os.path.join(build_temp, obj_name)

                # Build include dirs for NVCC
                include_args = []
                for inc in ext.include_dirs:
                    include_args.append("-I" + os.path.abspath(inc))

                cmd = nvcc_common + include_args + [
                    src_abs,
                    "-o",
                    obj_path,
                ]

                print("NVCC:", " ".join(cmd))
                subprocess.check_call(cmd)

                extra_objects.append(obj_path)

            ext.extra_objects = extra_objects

        # After CUDA objects are built and registered in extra_objects,
        # let the normal build_ext do its job for C++ sources + linking.
        super().build_extensions()

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
        "scalerqec.qepg",
        [
            "QEPG/bindings.cpp",
            "QEPG/src/QEPG.cpp",
            "QEPG/src/clifford.cpp",
            "QEPG/src/sampler.cpp",
            "QEPG/src/LERcalculator.cpp",
            "QEPG/src/sampler_gpu.cu",
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,        
        language="c++",
    )
]


setuptools.setup(
    name="scalerqec",
    version="0.0.1",
    description="Scalable Quantum Error Correction testing Tools for logical error rate and software correctness",
    author="John Ye",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtWithCUDA},
    zip_safe=False,
)

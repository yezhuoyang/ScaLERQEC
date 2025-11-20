import setuptools
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys



# pick the right “O3”‑style flag
extra_compile_args = []
extra_link_args = []



if sys.platform == 'win32':
    extra_compile_args=["/std:c++20", "/EHsc", "/O2", "/openmp:llvm"]
    extra_link_args=["/DEBUG"]
else:
    extra_compile_args = ["-std=c++20", "-O3"]


ext_modules=[
    setuptools.Extension(
        'scaler.qepg',
        ['QEPG/bindings.cpp','QEPG/src/QEPG.cpp','QEPG/src/clifford.cpp','QEPG/src/sampler.cpp','QEPG/src/LERcalculator.cpp'],
        include_dirs=[
            pybind11.get_include(),
            'QEPG',
            'QEPG/src',
            'C:/local/boost_1_87_0/',
            'C:/Users/username/OneDrive/Documents/GitHub/vcpkg/installed/x64-windows/include',
            'C:/vcpkg/installed/x64-windows/include',
            'C:/Users/username/AppData/Local/Programs/Python/Python311/Include',
            'C:/Users/username/miniconda3/Include',
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
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
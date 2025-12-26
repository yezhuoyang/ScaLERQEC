"""
ScaLER: Scalable Logical Error Rate Estimation Toolkit
"""



import os
import sys
import ctypes

# Ensure CUDA runtime DLL is loaded before importing the C++/CUDA extension.
if sys.platform == "win32":
    # Prefer CUDA_PATH if defined, otherwise use your v12.8 install.
    cuda_root = os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    )
    cuda_bin = os.path.join(cuda_root, "bin")

    if os.path.isdir(cuda_bin):
        # 1) Add CUDA bin directory to DLL search path
        try:
            os.add_dll_directory(cuda_bin)
        except (AttributeError, FileNotFoundError):
            # Older Python / bad path: ignore and fall back to PATH-only behaviour
            pass

        # 2) Eagerly load cudart64_*.dll by full path to avoid “DLL not found”
        try:
            dll_name = None
            for fname in os.listdir(cuda_bin):
                low = fname.lower()
                if low.startswith("cudart64_") and low.endswith(".dll"):
                    dll_name = fname
                    break

            if dll_name is not None:
                full_path = os.path.join(cuda_bin, dll_name)
                ctypes.CDLL(full_path)
        except Exception:
            # Non-fatal: importing qepg may still succeed if PATH works
            pass


# Expose C++ backend as scaler.qepg
from . import qepg

# Re-export high-level components for easy access
from .Stratified.stratifiedLER import StratifiedLERcalc
from .Stratified.stratifiedScurveLER import StratifiedScurveLERcalc
from .Symbolic.symbolicLER import SymbolicLERcalc
from .Monte.monteLER import MonteLERcalc
from .Clifford.clifford import CliffordCircuit
from .QEC.qeccircuit import StabCode


__all__ = [
    "StratifiedLERcalc",
    "StratifiedScurveLERcalc",
    "symbolicLER",
    "MonteLERcalc",
    "CliffordCircuit",
    "qepg",
    "StabCode"
]

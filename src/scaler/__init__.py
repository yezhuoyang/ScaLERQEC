"""
ScaLER: Scalable Logical Error Rate Estimation Toolkit
"""

# Expose C++ backend as scaler.qepg
from . import qepg

# Re-export high-level components for easy access
from .LERcalculator import LERcalculator
from .stratifiedLERcalc import stratifiedLERcalc
from .stratifiedScurveLER import stratified_Scurve_LERcalc
from .symbolicLER import symbolicLER
from .monteLER import monteLER
from .clifford import CliffordCircuit
from .interface import interface



__all__ = [
    "LERcalculator",
    "stratifiedLERcalc",
    "stratified_Scurve_LERcalc",
    "symbolicLER",
    "monteLER",
    "monteLER",
    "CliffordCircuit",
    "interface",
    "qepg",
]

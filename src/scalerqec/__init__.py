"""
ScaLER: Scalable Logical Error Rate Estimation Toolkit
"""

# Expose C++ backend as scaler.qepg
from . import qepg

# Re-export high-level components for easy access
from .Stratified.stratifiedLERcalc import stratifiedLERcalc
from .Stratified.stratifiedScurveLER import stratified_Scurve_LERcalc
from .Symbolic.symbolicLER import symbolicLER
from .Monte.monteLER import stimLERcalc
from .Clifford.clifford import CliffordCircuit
from .QEC.qeccircuit import QECStab


__all__ = [
    "stratifiedLERcalc",
    "stratified_Scurve_LERcalc",
    "symbolicLER",
    "stimLERcalc",
    "CliffordCircuit",
    "qepg",
    "QECStab"
]

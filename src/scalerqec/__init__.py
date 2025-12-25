"""
ScaLER: Scalable Logical Error Rate Estimation Toolkit
"""

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

"""
ScaLER: Scalable Logical Error Rate Estimation Toolkit
"""

# Expose C++ backend as scaler.qepg
from . import qepg

# Re-export high-level components for easy access
from .stratifiedLERcalc import stratifiedLERcalc
from .stratifiedScurveLER import stratified_Scurve_LERcalc
from .symbolicLER import symbolicLER
from .monteLER import stimLERcalc
from .clifford import CliffordCircuit



__all__ = [
    "stratifiedLERcalc",
    "stratified_Scurve_LERcalc",
    "symbolicLER",
    "stimLERcalc",
    "CliffordCircuit",
    "qepg",
]

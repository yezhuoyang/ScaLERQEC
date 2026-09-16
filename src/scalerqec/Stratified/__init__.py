"""Stratified sampling module for logical error rate (LER) estimation.

This package implements the ScaLER (Scalable Logical Error Rate) algorithm,
which uses stratified sampling to efficiently estimate logical error rates
for quantum error correction codes. Instead of brute-force Monte Carlo
sampling, ScaLER stratifies by error weight *w*, estimates the conditional
logical error probability P_L(w) for each weight, fits an S-curve model,
and integrates to compute the total LER:

    LER = sum_w  P_L(w) * Binom(N, w) * p^w * (1-p)^(N-w)

where *N* is the number of noise locations and *p* is the physical error
rate. This approach dramatically reduces the number of samples required
for large codes.

Main components:

- :class:`Scaler` -- the primary ScaLER algorithm with time-budgeted,
  multi-phase sampling and S-curve model fitting.
- :class:`StratifiedLERcalc` -- legacy stratified sampler without curve
  fitting (sample-budget-based).
- :class:`StratifiedScurveLERcalc` -- legacy stratified sampler with
  S-curve fitting (sample-budget-based).
- S-curve model classes in the :mod:`models` subpackage
  (:class:`OurScurveModel`, :class:`IBMScurveModel`, :class:`ModelFactory`).
"""

# Re-export high-level components for easy access

from .stratifiedLER import StratifiedLERcalc
from .stratifiedScurveLER import StratifiedScurveLERcalc
from .Scaler import Scaler
from .profile import LERProfile, LERCurve

# Export model classes and factory
from .models import (
    ScurveModelBase,
    OurScurveModel,
    IBMScurveModel,
    ModelType,
    ModelFactory,
)

__all__ = [
    "StratifiedLERcalc",
    "StratifiedScurveLERcalc",
    "Scaler",
    "LERProfile",
    "LERCurve",
    # Model classes
    "ScurveModelBase",
    "OurScurveModel",
    "IBMScurveModel",
    "ModelType",
    "ModelFactory",
]

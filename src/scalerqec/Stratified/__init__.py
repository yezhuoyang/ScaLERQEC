"""Stratified sampling module for logical error rate (LER) estimation.

This package implements the ScaLER (Scalable Logical Error Rate) algorithm,
which uses stratified sampling to efficiently estimate logical error rates
for quantum error correction codes. Instead of brute-force Monte Carlo
sampling, ScaLER stratifies by error weight *w*, estimates the conditional
logical error probability P_L(w) for each weight, fits an S-curve model,
and integrates to compute the total LER:

    LER = sum_w  P_L(w) * Binom(N, w) * p^w * (1-p)^(N-w)

where *N* is the number of noise locations and *p* is the physical error
rate. Efficiency depends on the code, noise family, proposal, and accuracy
required. The experimental general-noise profiler retains additional history
likelihoods; a single conditional failure probability per weight is insufficient
for general nonuniform noise.

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

from .adaptive import AccuracyControlledProfile, AccuracyEstimate
from .confidence import GeneralNoiseConfidenceBounds
from .general_noise import GeneralNoiseEstimate, GeneralNoiseProfile, LinearNoiseModel

# Export model classes and factory
from .models import (
    IBMScurveModel,
    ModelFactory,
    ModelType,
    OurScurveModel,
    ScurveModelBase,
)
from .noise_polynomial import LERPolynomial
from .profile import LERCurve, LERProfile
from .Scaler import Scaler
from .stratifiedLER import StratifiedLERcalc
from .stratifiedScurveLER import StratifiedScurveLERcalc
from .uniformized import BernsteinProfile, UniformizedSampler

__all__ = [
    "AccuracyControlledProfile",
    "AccuracyEstimate",
    "BernsteinProfile",
    "GeneralNoiseConfidenceBounds",
    "GeneralNoiseEstimate",
    "GeneralNoiseProfile",
    "IBMScurveModel",
    "LERCurve",
    "LERPolynomial",
    "LERProfile",
    "LinearNoiseModel",
    "ModelFactory",
    "ModelType",
    "OurScurveModel",
    "Scaler",
    "ScurveModelBase",
    "StratifiedLERcalc",
    "StratifiedScurveLERcalc",
    "UniformizedSampler",
]

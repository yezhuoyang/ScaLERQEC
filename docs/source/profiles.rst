Profile once, evaluate many physical error rates
===============================================

For uniform independent single-qubit depolarizing noise and a fixed decoder,
the conditional logical failure probabilities at each fault weight do not
depend on the physical error probability p. ScaLER 1.1 separates this expensive
profiling step from inexpensive binomial reweighting::

   import numpy as np
   from scalerqec.Stratified import Scaler, LERProfile

   scaler = Scaler(time_budget=60)
   profile = scaler.profile_from_file(
       "surface3.stim", codedistance=3, decoder_reference_p=0.001,
   )
   profile.save("surface3-profile.json")

   p = np.geomspace(1e-5, 0.02, 100)
   ler = profile.evaluate(p)         # no compilation, decoding, sampling, or fitting
   ax = profile.plot(p)
   ax.figure.savefig("ler-vs-p.pdf", bbox_inches="tight")

   restored = LERProfile.load("surface3-profile.json")
   print(restored.evaluate(0.0005))   # scalar result; arrays retain their shape

A profile can also be extracted after an existing single-p calculation using
``scaler.get_profile()``. The convenience method
``calculate_LER_curve_from_file(filepath, p_values, codedistance)`` profiles
once and returns a ``LERCurve``. Keep the profile object or save it to reuse
the calculation across processes.

Scientific scope
----------------

The input circuit must be noiseless and have exactly one logical observable,
index 0. ScaLER inserts SID faults before each normalized primitive gate or
measurement, excluding resets. ``REPEAT`` blocks and multi-target instructions
are normalized. Unsupported instructions, explicit noise, classical controls,
inverted measurements, and multiple observables fail with an explanatory error.
General nonuniform or correlated noise requires a different weighting scheme.

``decoder_reference_p`` specifies the physical probability used to construct
the fixed PyMatching decoder; it is not a restriction on the p values that can
be evaluated. A custom ``decode_batch`` decoder can be supplied to ``Scaler``.
The profile does not serialize decoder code and does not retune the decoder
when evaluating another p. A threshold plot made this way describes a family
of fixed decoders. It must not be labeled as a p-retuned decoder experiment.

``codedistance`` is the circuit-level distance for the circuit and decoder,
not necessarily the nominal code distance. It supplies the assumption that
weights up to floor((d-1)/2) have zero failure probability.

All N+1 binomial weights are integrated, without a normal approximation or
five-sigma cutoff. Temporary arrays are chunked. The stored conditional rates
are the coefficients in the binomial/Bernstein representation; conversion to
dense monomial coefficients is unnecessary and often numerically ill-conditioned.

Model diagnostics and uncertainty
---------------------------------

``profile.curve(p)`` returns the total LER, ``modeled_ler`` (the contribution
from weights supplied by the model), and ``modeled_probability_mass`` (their
binomial mass). ``profile.sample_counts``, ``failure_counts`` and
``modeled_weights`` preserve the underlying evidence. Metadata records the
circuit hash, package version, distance assumption, model parameters, decoder
class, and decoder reference probability.

These diagnostics are not confidence intervals. Measured rates have sampling
uncertainty; zero observed failures do not establish zero probability.
Extrapolation has unquantified systematic error that is not bounded by a high
R-squared or by repeatability. A 0.5 S-curve asymptote is not universal at high
fault weights. Evaluation is numerically supported throughout [0,1], but
scientific accuracy must be checked at representative p values with direct
sampling or exact enumeration, especially where model contributions dominate.

See `the paper <https://arxiv.org/pdf/2602.04921>`_, equations 2-4 and section 9.

API
---

.. automodule:: scalerqec.Stratified.profile
   :members:

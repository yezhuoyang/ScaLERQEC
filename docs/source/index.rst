==================================
ScaLERQEC Documentation
==================================

**ScaLERQEC** (Scalable Logical Error Rate Estimation for Quantum Error Correction)
is a Python toolkit for estimating the logical error rate (LER) of stabilizer codes
under realistic noise models. It provides multiple estimation strategies -- from exact
symbolic computation to scalable stratified sampling -- so users can trade off
accuracy, speed, and code distance.

Architecture Overview
=====================

ScaLERQEC is organized into six Python modules and one compiled C++ backend:

**Clifford** -- Circuit representation, STIM parsing, QEPG construction
   Provides :class:`~scalerqec.Clifford.clifford.CliffordCircuit`, the core
   circuit representation.  Handles import from STIM circuit files and
   construction of the Quantum Error Propagation Graph (QEPG) used by
   downstream estimators.

**QEC** -- Stabilizer code definitions, noise models, IR compilation
   Defines :class:`~scalerqec.QEC.qeccircuit.StabCode` for specifying
   stabilizer codes together with their noise parameters.  Compiles a
   high-level code description into the internal representation consumed by
   the estimation engines.

**Monte** -- Monte Carlo LER estimation via stim detector sampling
   Implements :class:`~scalerqec.Monte.monteLER.MonteLERcalc`, a
   straightforward Monte Carlo estimator that draws detector samples from
   ``stim`` and decodes them to estimate the logical error rate.

**Stratified** -- ScaLER algorithm: stratified fault injection with S-curve fitting
   The flagship module.  Contains
   :class:`~scalerqec.Stratified.stratifiedLER.StratifiedLERcalc` and
   :class:`~scalerqec.Stratified.stratifiedScurveLER.StratifiedScurveLERcalc`,
   which partition faults by weight and fit an S-curve to estimate LER
   efficiently even at very low physical error rates.

**Symbolic** -- Exact symbolic LER as a polynomial via dynamic programming
   Provides :class:`~scalerqec.Symbolic.symbolicLER.SymbolicLERcalc`, which
   computes the logical error rate symbolically as a polynomial in the
   physical error rate using dynamic programming over the QEPG.

**util** -- Helper functions (Pauli algebra, binomial weights, formatting)
   Shared utilities used across modules, including Pauli-group algebra,
   binomial weight calculations, and output formatting helpers.

**qepg** (C++ backend) -- Compiled error propagation graph, fast sampling
   A compiled C++ extension that provides the high-performance error
   propagation graph data structure and sampling routines underlying the
   Python estimators.

Getting Started
===============

A minimal example that estimates the logical error rate of a surface code
using the stratified S-curve method::

   from scalerqec import StabCode, StratifiedScurveLERcalc

   # Define a stabilizer code with a noise model
   code = StabCode.from_stim("path/to/surface_code.stim")

   # Run the ScaLER stratified estimator
   calc = StratifiedScurveLERcalc(code)
   result = calc.run()

   print(f"Estimated LER: {result}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   profiles
   api/modules

# QEPG


qepg is a Python extension module (built with pybind11) that exposes core functionality from the C++ QEPG library for sampling detector/observable outcomes and working with a compiled QEPG graph derived from a Stim-style program string.


## QEPG

QEPGGraph is the Python binding of the C++ QEPG::QEPG class.
It represents a compiled error-propagation graph that can be reused across multiple sampling calls.

## Clifford


## LERcalculator

LERcalculator is a C++ namespace whose functions are exposed at the module level in Python.
These functions provide the main sampling and analysis interface.


## Sampler


Sampling utilities returning Python lists or NumPy arrays
Compile-once, reuse-many workflow via QEPGGraph
Support for multi-weight sampling and Monte Carlo sampling
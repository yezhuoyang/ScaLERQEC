"""
Pybind11 bindings for QEPG library
"""
from __future__ import annotations
import collections.abc
import numpy.typing
import typing
__all__: list[str] = ['CliffordCircuit', 'DynamicBitset', 'QEPGGraph', 'Sampler', 'compile_QEPG', 'return_detector_matrix', 'return_samples', 'return_samples_Monte_separate_obs_with_QEPG', 'return_samples_many_weights', 'return_samples_many_weights_numpy', 'return_samples_many_weights_separate_obs', 'return_samples_many_weights_separate_obs_with_QEPG', 'return_samples_numpy', 'return_samples_with_fixed_QEPG', 'return_samples_with_noise_vector']
class CliffordCircuit:
    def __init__(self) -> None:
        ...
    def compile_from_rewrited_stim_string(self, prog_str: str) -> None:
        """
        Compile circuit from Stim string
        """
    def get_num_detector(self) -> int:
        ...
    def get_num_noise(self) -> int:
        ...
    def get_num_qubit(self) -> int:
        ...
class DynamicBitset:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: DynamicBitset) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def size(self) -> int:
        """
        Get the size of the bitset
        """
    def test(self, pos: typing.SupportsInt) -> bool:
        """
        Test if the bit at position pos is set
        """
    def to_list(self) -> list[bool]:
        """
        Convert the bitset to a list of booleans
        """
class QEPGGraph:
    def __init__(self, circuit: CliffordCircuit, num_detector: typing.SupportsInt, num_noise: typing.SupportsInt) -> None:
        ...
    def backward_graph_construction(self) -> None:
        ...
class Sampler:
    def __init__(self, num_total_paulierror: typing.SupportsInt) -> None:
        ...
def compile_QEPG(prog_str: str) -> QEPGGraph:
    """
                compile_QEPG(prog_str: str) → QEPGGraph
                Parse a Stim‐style program string into a QEPGGraph object,
                run its backward_graph_construction() pass, and return it.
    """
def return_detector_matrix(prog_str: str) -> list[list[bool]]:
    """
    Function that returns the detector matrix
    """
def return_samples(prog_str: str, weight: typing.SupportsInt, shots: typing.SupportsInt) -> list[list[bool]]:
    """
    Function that returns samples based on a circuit and parameters
    """
def return_samples_Monte_separate_obs_with_QEPG(graph: QEPGGraph, error_rate: typing.SupportsFloat, shot: typing.SupportsInt) -> tuple[numpy.typing.NDArray[numpy.bool_], numpy.typing.NDArray[numpy.bool_]]:
    """
    Function that returns samples based on a QEPG with monte carlo method
    """
def return_samples_many_weights(prog_str: str, weight: collections.abc.Sequence[typing.SupportsInt], shots: collections.abc.Sequence[typing.SupportsInt]) -> list[list[list[bool]]]:
    """
    Function that returns samples of a list of weights based on a circuit and parameters
    """
def return_samples_many_weights_numpy(prog_str: str, weight: collections.abc.Sequence[typing.SupportsInt], shots: collections.abc.Sequence[typing.SupportsInt]) -> list[numpy.typing.NDArray[numpy.bool_]]:
    """
    Function that returns samples of a list of weights based on a circuit and parameters, it return numpy vector directly
    """
def return_samples_many_weights_separate_obs(prog_str: str, weight: collections.abc.Sequence[typing.SupportsInt], shots: collections.abc.Sequence[typing.SupportsInt]) -> tuple[numpy.typing.NDArray[numpy.bool_], numpy.typing.NDArray[numpy.bool_]]:
    ...
def return_samples_many_weights_separate_obs_with_QEPG(graph: QEPGGraph, weight: collections.abc.Sequence[typing.SupportsInt], shots: collections.abc.Sequence[typing.SupportsInt]) -> tuple[numpy.typing.NDArray[numpy.bool_], numpy.typing.NDArray[numpy.bool_]]:
    ...
def return_samples_numpy(prog_str: str, weight: typing.SupportsInt, shots: typing.SupportsInt) -> numpy.typing.NDArray[numpy.bool_]:
    """
    Function that directly return numpy array
    """
def return_samples_with_fixed_QEPG(graph: QEPGGraph, weight: typing.SupportsInt, shots: typing.SupportsInt) -> list[list[bool]]:
    """
    Function that returns samples based on a QEPG
    """
def return_samples_with_noise_vector(prog_str: str, weight: typing.SupportsInt, shots: typing.SupportsInt) -> tuple[list[list[tuple[int, int]]], list[list[bool]]]:
    ...

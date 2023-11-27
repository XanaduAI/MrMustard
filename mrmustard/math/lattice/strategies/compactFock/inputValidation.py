"""
This module contains helper functions that are used in
diagonal_amps.py, diagonal_grad.py, singleLeftoverMode_amps.py and singleLeftoverMode_grad.py
to validate the input provided by the user.
"""

from typing import Iterable
import numpy as np
from mrmustard.math.lattice.strategies.compactFock.diagonal_amps import (
    fock_representation_diagonal_amps,
)
from mrmustard.math.lattice.strategies.compactFock.diagonal_grad import (
    fock_representation_diagonal_grad,
)
from mrmustard.math.lattice.strategies.compactFock.singleLeftoverMode_amps import (
    fock_representation_1leftoverMode_amps,
)
from mrmustard.math.lattice.strategies.compactFock.singleLeftoverMode_grad import (
    fock_representation_1leftoverMode_grad,
)
from thewalrus._hafnian import input_validation


def hermite_multidimensional_diagonal(A, B, G0, cutoffs, rtol=1e-05, atol=1e-08):
    """
    Validation of user input for mrmustard.math.backend_tensorflow.hermite_renormalized_diagonal
    """
    input_validation(A, atol=atol, rtol=rtol)
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    if isinstance(cutoffs, Iterable):
        cutoffs = tuple(cutoffs)
    else:
        raise ValueError("cutoffs should be array like of length M")
    M = len(cutoffs)
    if A.shape[0] // 2 != M:
        raise ValueError("The matrix A and cutoffs have incompatible dimensions")
    return fock_representation_diagonal_amps(A, B, G0, M, cutoffs)


def grad_hermite_multidimensional_diagonal(A, B, G0, arr0, arr2, arr1010, arr1001, arr1):
    """
    Validation of user input for gradients of mrmustard.math.backend_tensorflow.hermite_renormalized_diagonal
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    M = A.shape[0] // 2
    arr0_dA, arr0_dB = fock_representation_diagonal_grad(
        A, B, M, arr0, arr2, arr1010, arr1001, arr1
    )
    arr0_dG0 = np.array(arr0 / G0).astype(np.complex128)
    return arr0_dG0, arr0_dA, arr0_dB


def hermite_multidimensional_1leftoverMode(A, B, G0, cutoffs, rtol=1e-05, atol=1e-08):
    """
    Validation of user input for mrmustard.math.backend_tensorflow.hermite_renormalized_1leftoverMode
    """
    input_validation(A, atol=atol, rtol=rtol)
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    if isinstance(cutoffs, Iterable):
        cutoffs = tuple(cutoffs)
    else:
        raise ValueError("cutoffs should be array like of length M")
    M = len(cutoffs)
    if A.shape[0] // 2 != M:
        raise ValueError("The matrix A and cutoffs have incompatible dimensions")
    if M <= 1:
        raise ValueError("The number of modes should be greater than 1.")
    return fock_representation_1leftoverMode_amps(A, B, G0, M, cutoffs)


def grad_hermite_multidimensional_1leftoverMode(A, B, G0, arr0, arr2, arr1010, arr1001, arr1):
    """
    Validation of user input for gradients of mrmustard.math.backend_tensorflow.hermite_renormalized_1leftoverMode
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    M = A.shape[0] // 2
    if M <= 1:
        raise ValueError("The number of modes should be greater than 1.")
    arr0_dA, arr0_dB = fock_representation_1leftoverMode_grad(
        A, B, M, arr0, arr2, arr1010, arr1001, arr1
    )
    arr0_dG0 = np.array(arr0 / G0).astype(np.complex128)
    return arr0_dG0, arr0_dA, arr0_dB


def hermite_multidimensional_diagonal_batch(A, B, G0, cutoffs, rtol=1e-05, atol=1e-08):
    """
    Validation of user input for mrmustard.math.backend_tensorflow.hermite_renormalized_diagonal_batch
    """
    input_validation(A, atol=atol, rtol=rtol)
    if len(B.shape) != 2:
        raise ValueError("B should be two dimensional (vector and batch dimension)")
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    if isinstance(cutoffs, Iterable):
        cutoffs = tuple(cutoffs)
    else:
        raise ValueError("cutoffs should be array like of length M")
    M = len(cutoffs)
    if A.shape[0] // 2 != M:
        raise ValueError("The matrix A and cutoffs have incompatible dimensions")
    return fock_representation_diagonal_amps(A, B, G0, M, cutoffs)

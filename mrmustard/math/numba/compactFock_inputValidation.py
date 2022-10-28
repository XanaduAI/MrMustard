import numpy as np
from typing import Iterable
from mrmustard.math.numba.compactFock_diagonal_amps import fock_representation_diagonal_amps
from mrmustard.math.numba.compactFock_diagonal_grad import fock_representation_diagonal_grad
from mrmustard.math.numba.compactFock_1leftoverMode_amps import fock_representation_1leftoverMode_amps
from mrmustard.math.numba.compactFock_1leftoverMode_grad import fock_representation_1leftoverMode_grad
from thewalrus._hafnian import input_validation

def hermite_multidimensional_diagonal(A,B,G0,cutoffs,rtol=1e-05, atol=1e-08):
    input_validation(A, atol=atol, rtol=rtol)
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    # Should I use np.real_if_close on A and B here?
    if isinstance(cutoffs, Iterable):
        cutoffs = tuple(cutoffs)
    else:
        raise ValueError("cutoffs should be array like of length M")
    M = len(cutoffs)
    if A.shape[0]//2 != M:
        raise ValueError("The matrix A and cutoffs have incompatible dimensions")
    return fock_representation_diagonal_amps(A, B, G0, M, cutoffs)

def grad_hermite_multidimensional_diagonal(A,B,G0,arr0,arr2,arr1010,arr1001,arr1):
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    M = A.shape[0] // 2
    arr0_dA,arr0_dB = fock_representation_diagonal_grad(A, B, M,arr0,arr2,arr1010,arr1001,arr1)
    arr0_dG0 = np.array(arr0 / G0).astype(np.complex128)
    return arr0_dG0,arr0_dA,arr0_dB

def hermite_multidimensional_1leftoverMode(A,B,G0,cutoffs,rtol=1e-05, atol=1e-08):
    input_validation(A, atol=atol, rtol=rtol)
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    # Should I use np.real_if_close on A and B here?
    if isinstance(cutoffs, Iterable):
        cutoffs = tuple(cutoffs)
    else:
        raise ValueError("cutoffs should be array like of length M")
    M = len(cutoffs)
    if A.shape[0]//2 != M:
        raise ValueError("The matrix A and cutoffs have incompatible dimensions")
    return fock_representation_1leftoverMode_amps(A, B, G0, M, cutoffs)

def grad_hermite_multidimensional_1leftoverMode(A,B,G0,arr0,arr2,arr1010,arr1001,arr1):
    if A.shape[0] != B.shape[0]:
        raise ValueError("The matrix A and vector B have incompatible dimensions")
    M = A.shape[0] // 2
    arr0_dA,arr0_dB = fock_representation_1leftoverMode_grad(A, B, M,arr0,arr2,arr1010,arr1001,arr1)
    arr0_dG0 = np.array(arr0 / G0).astype(np.complex128)
    return arr0_dG0,arr0_dA,arr0_dB
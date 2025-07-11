"""
This module calculates the diagonal of the Fock representation (i.e. the PNR detection probabilities of all modes)
by applying the recursion relation in a selective manner.
"""

import numba
import numpy as np
from numba import int64, njit
from numba.cpython.unsafe.tuple import tuple_setitem

from mrmustard.math.lattice.strategies.compactFock.helperFunctions import (
    SQRT,
    construct_dict_params,
    repeat_twice,
)


@njit(cache=True)
def use_offDiag_pivot(  # noqa: C901
    A,
    B,
    M,
    cutoffs,
    params,
    d,
    arr0,
    arr2,
    arr1010,
    arr1001,
    arr1,
):  # pragma: no cover
    """
    Apply recurrence relation for pivot of type [a+1,a,b,b,c,c,...] / [a,a,b+1,b,c,c,...] / [a,a,b,b,c+1,c,...]
    Args:
        A, B (array, vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of modes
        cutoffs (tuple): upper bounds for the number of photons in each mode
        params (tuple): (a,b,c,...)
        d (int): mode index in which the considered Fock amplitude is off diagonal
            e.g. [a,a,b+1,b,c,c,...] --> b is off diagonal --> d=1
        arr0, arr2, arr1010, arr1001, arr1 (array, array, array, array, array): submatrices of the fock representation
    Returns:
        (array, array, array, array, array): updated versions of arr0, arr2, arr1010, arr1001, arr1
    """
    pivot = repeat_twice(params)
    pivot[2 * d] += 1
    K_l = SQRT[pivot]
    K_i = SQRT[pivot + 1]
    if B.ndim == 1:
        G_in = np.zeros(2 * M, dtype=np.complex128)
    elif B.ndim == 2:
        G_in = np.zeros((2 * M, B.shape[1]), dtype=np.complex128)

    ########## READ ##########
    GB = arr1[(2 * d, *params)] * B

    # Array0
    G_in[2 * d] = arr0[params]

    # read from Array2
    if params[d] > 0:
        params_adapted = tuple_setitem(params, d, params[d] - 1)
        G_in[2 * d + 1] = arr2[(d, *params_adapted)]

    # read from Array11
    for i in range(d + 1, M):  # i>d
        if params[i] > 0:
            params_adapted = tuple_setitem(params, i, params[i] - 1)
            G_in[2 * i] = arr1001[(d, i - d - 1, *params_adapted)]
            G_in[2 * i + 1] = arr1010[(d, i - d - 1, *params_adapted)]

    ########## WRITE ##########
    if B.ndim == 1:
        G_in = np.multiply(K_l, G_in)
    elif B.ndim == 2:
        G_in = np.multiply(np.expand_dims(K_l, 1), G_in)

    # Array0
    params_adapted = tuple_setitem(params, d, params[d] + 1)
    arr0[params_adapted] = (GB[2 * d + 1] + A[2 * d + 1] @ G_in) / K_i[2 * d + 1]

    # Array2
    if params[d] + 2 < cutoffs[d]:
        arr2[(d, *params)] = (GB[2 * d] + A[2 * d] @ G_in) / K_i[2 * d]

    # Array11
    for i in range(d + 1, M):
        if params[i] + 1 < cutoffs[i]:
            arr1010[(d, i - d - 1, *params)] = (GB[2 * i] + A[2 * i] @ G_in) / K_i[2 * i]
            arr1001[(d, i - d - 1, *params)] = (GB[2 * i + 1] + A[2 * i + 1] @ G_in) / K_i[
                2 * i + 1
            ]

    return arr0, arr2, arr1010, arr1001


@njit(cache=True)
def use_diag_pivot(A, B, M, cutoffs, params, arr0, arr1):  # pragma: no cover
    """
    Apply recurrence relation for pivot of type [a,a,b,b,c,c...]
    Args:
        A, B (array, vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of modes
        cutoffs (tuple): upper bounds for the number of photons in each mode
        params (tuple): (a,b,c,...)
        arr0, arr1 (array, array): submatrices of the fock representation
    Returns:
        (array, array): updated versions of arr0, arr1
    """
    pivot = repeat_twice(params)
    K_l = SQRT[pivot]
    K_i = SQRT[pivot + 1]
    if B.ndim == 1:
        G_in = np.zeros(2 * M, dtype=np.complex128)
    elif B.ndim == 2:
        G_in = np.zeros((2 * M, B.shape[1]), dtype=np.complex128)

    ########## READ ##########
    GB = arr0[params] * B

    # Array1
    for i in range(2 * M):
        if params[i // 2] > 0:
            params_adapted = tuple_setitem(params, i // 2, params[i // 2] - 1)
            G_in[i] = arr1[
                (i + 1 - 2 * (i % 2), *params_adapted)
            ]  # [i+1-2*(i%2) for i in range(6)] = [1,0,3,2,5,4]

    ########## WRITE ##########
    if B.ndim == 1:
        G_in = np.multiply(K_l, G_in)
    elif B.ndim == 2:
        G_in = np.multiply(np.expand_dims(K_l, 1), G_in)

    # Array1
    for i in range(2 * M):
        if params[i // 2] + 1 < cutoffs[i // 2] and (i != 1 or params[0] + 2 < cutoffs[0]):
            # this prevents a few elements from being written that will never be read
            arr1[(i, *params)] = (GB[i] + A[i] @ G_in) / K_i[i]

    return arr1


@njit(cache=True)
def fock_representation_diagonal_amps_NUMBA(
    A,
    B,
    M,
    cutoffs,
    arr0,
    arr2,
    arr1010,
    arr1001,
    arr1,
    tuple_type,
    list_type,
):  # pragma: no cover
    """
    Returns the PNR probabilities of a mixed state according to algorithm 1 of:
    https://doi.org/10.22331/q-2023-08-29-1097
    Args:
        A, B (array, vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of modes
        cutoffs (tuple): upper bounds for the number of photons in each mode
        arr0 (array): submatrix of the fock representation that contains Fock amplitudes of the type [a,a,b,b,c,c...]
            (!) should already contain G0 at position (0,)*M
        arr2 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+2,a,b,b,c,c...] / [a,a,b+2,b,c,c...] / ...
        arr1010 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b+1,b,c,c,...] / [a+1,a,b,b,c+1,c,...] / [a,a,b+1,b,c+1,c,...] / ...
        arr1001 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b,b+1,c,c,...] / [a+1,a,b,b,c,c+1,...] / [a,a,b+1,b,c,c+1,...] / ...
        arr1 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b,b,c,c...] / [a,a+1,b,b,c,c...] / [a,a,b+1,b,c,c...] / ...
        tuple_type, list_type (numba types): numba types that need to be defined outside of numba compiled functions
    Returns:
        array: the fock representation
    """
    dict_params = construct_dict_params(cutoffs, tuple_type, list_type)
    for sum_params in range(sum(cutoffs)):
        for params in dict_params[sum_params]:
            # diagonal pivots: aa,bb,cc,dd,...
            if (cutoffs[0] == 1) or (params[0] < cutoffs[0] - 1):
                arr1 = use_diag_pivot(A, B, M, cutoffs, params, arr0, arr1)
            # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: 00,(b+1)b,cc,dd | 00,00,(c+1)c,dd | ...
            for d in range(M):
                if np.all(np.array(params)[:d] == 0) and (params[d] < cutoffs[d] - 1):
                    arr0, arr2, arr1010, arr1001 = use_offDiag_pivot(
                        A,
                        B,
                        M,
                        cutoffs,
                        params,
                        d,
                        arr0,
                        arr2,
                        arr1010,
                        arr1001,
                        arr1,
                    )
    return arr0, arr2, arr1010, arr1001, arr1


def fock_representation_diagonal_amps(A, B, G0, M, cutoffs):
    """
    First initialise the submatrices of G (of which the shape depends on cutoff and M)
    and some other constants
    (These initialisations currently cannot be done using Numba.)
    Then calculate the fock representation.
    """

    cutoffs = tuple(cutoffs)
    tuple_type = numba.types.UniTuple(int64, M)
    list_type = numba.types.ListType(tuple_type)

    if B.ndim == 1:
        arr0 = np.zeros(cutoffs, dtype=np.complex128)
        arr2 = np.zeros((M, *cutoffs), dtype=np.complex128)
        arr1 = np.zeros((2 * M, *cutoffs), dtype=np.complex128)
        if M == 1:
            arr1010 = np.zeros((1, 1, 1), dtype=np.complex128)
            arr1001 = np.zeros((1, 1, 1), dtype=np.complex128)
        else:
            arr1010 = np.zeros((M, M - 1, *cutoffs), dtype=np.complex128)
            arr1001 = np.zeros((M, M - 1, *cutoffs), dtype=np.complex128)

    elif B.ndim == 2:
        batch_length = B.shape[1]
        arr0 = np.zeros((*cutoffs, batch_length), dtype=np.complex128)
        arr2 = np.zeros((M, *cutoffs, batch_length), dtype=np.complex128)
        arr1 = np.zeros((2 * M, *cutoffs, batch_length), dtype=np.complex128)
        if M == 1:
            arr1010 = np.zeros((1, 1, 1, batch_length), dtype=np.complex128)
            arr1001 = np.zeros((1, 1, 1, batch_length), dtype=np.complex128)
        else:
            arr1010 = np.zeros((M, M - 1, *cutoffs, batch_length), dtype=np.complex128)
            arr1001 = np.zeros((M, M - 1, *cutoffs, batch_length), dtype=np.complex128)

    arr0[(0,) * M] = G0
    return fock_representation_diagonal_amps_NUMBA(
        A,
        B,
        M,
        cutoffs,
        arr0,
        arr2,
        arr1010,
        arr1001,
        arr1,
        tuple_type,
        list_type,
    )

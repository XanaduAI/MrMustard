"""
This module calculates the derivatives of the diagonal of the Fock representation (i.e. the PNR detection probabilities of all modes)
by applying the derivated recursion relation in a selective manner.
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
def calc_dA_dB(i, G_in_dA, G_in_dB, G_in, A, B, K_l, K_i, M, pivot_val, pivot_val_dA, pivot_val_dB):
    """
    Calculate the derivatives of one Fock amplitude w.r.t A and B.
    Args:
        i (int): the element of the multidim index that is increased
        G_in, G_in_dA, G_in_dB (array, array, array): all Fock amplitudes from the 'read' group in the recurrence relation and their derivatives w.r.t. A and B
        A, B (array, vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        K_l, K_i (vector, vector): SQRT[pivot], SQRT[pivot + 1]
        M (int): number of modes
        pivot_val, pivot_val_dA, pivot_val_dB (array, array, array): Fock amplitude at the position of the pivot and its derivatives w.r.t. A and B
    """
    dA = pivot_val_dA * B[i]
    dB = pivot_val_dB * B[i]
    dB[i] += pivot_val
    for l in range(2 * M):
        dA += K_l[l] * A[i, l] * G_in_dA[l]
        dB += K_l[l] * A[i, l] * G_in_dB[l]
        dA[i, l] += G_in[l]
    return dA / K_i[i], dB / K_i[i]


@njit(cache=True)
def use_offDiag_pivot_grad(
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
    arr0_dA,
    arr2_dA,
    arr1010_dA,
    arr1001_dA,
    arr1_dA,
    arr0_dB,
    arr2_dB,
    arr1010_dB,
    arr1001_dB,
    arr1_dB,
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
        arr..._dA, arr..._dB (array, array): derivatives of submatrices w.r.t A and B
    Returns:
        (array, array, array, array, array): updated versions of arr0, arr2, arr1010, arr1001, arr1
    """
    pivot = repeat_twice(params)
    pivot[2 * d] += 1
    K_l = SQRT[pivot]
    K_i = SQRT[pivot + 1]
    G_in = np.zeros(2 * M, dtype=np.complex128)
    G_in_dA = np.zeros((2 * M, *A.shape), dtype=np.complex128)
    G_in_dB = np.zeros((2 * M, *B.shape), dtype=np.complex128)

    ########## READ ##########
    pivot_val = arr1[2 * d][params]
    pivot_val_dA = arr1_dA[2 * d][params]
    pivot_val_dB = arr1_dB[2 * d][params]

    # Array0
    G_in[2 * d] = arr0[params]
    G_in_dA[2 * d] = arr0_dA[params]
    G_in_dB[2 * d] = arr0_dB[params]

    # read from Array2
    if params[d] > 0:
        G_in[2 * d + 1] = arr2[d][tuple_setitem(params, d, params[d] - 1)]
        G_in_dA[2 * d + 1] = arr2_dA[d][tuple_setitem(params, d, params[d] - 1)]
        G_in_dB[2 * d + 1] = arr2_dB[d][tuple_setitem(params, d, params[d] - 1)]

    # read from Array11
    for i in range(d + 1, M):  # i>d
        if params[i] > 0:
            params_adapted = tuple_setitem(params, i, params[i] - 1)
            G_in[2 * i] = arr1001[d][i - d - 1][params_adapted]
            G_in_dA[2 * i] = arr1001_dA[d][i - d - 1][params_adapted]
            G_in_dB[2 * i] = arr1001_dB[d][i - d - 1][params_adapted]
            G_in[2 * i + 1] = arr1010[d][i - d - 1][params_adapted]
            G_in_dA[2 * i + 1] = arr1010_dA[d][i - d - 1][params_adapted]
            G_in_dB[2 * i + 1] = arr1010_dB[d][i - d - 1][params_adapted]

    ########## WRITE ##########
    G_in = np.multiply(K_l, G_in)

    # Array0
    params_adapted = tuple_setitem(params, d, params[d] + 1)
    arr0_dA[params_adapted], arr0_dB[params_adapted] = calc_dA_dB(
        2 * d + 1,
        G_in_dA,
        G_in_dB,
        G_in,
        A,
        B,
        K_l,
        K_i,
        M,
        pivot_val,
        pivot_val_dA,
        pivot_val_dB,
    )

    # Array2
    if params[d] + 2 < cutoffs[d]:
        arr2_dA[d][params], arr2_dB[d][params] = calc_dA_dB(
            2 * d,
            G_in_dA,
            G_in_dB,
            G_in,
            A,
            B,
            K_l,
            K_i,
            M,
            pivot_val,
            pivot_val_dA,
            pivot_val_dB,
        )

    # Array11
    for i in range(d + 1, M):
        if params[i] + 1 < cutoffs[i]:
            (
                arr1010_dA[d][i - d - 1][params],
                arr1010_dB[d][i - d - 1][params],
            ) = calc_dA_dB(
                2 * i,
                G_in_dA,
                G_in_dB,
                G_in,
                A,
                B,
                K_l,
                K_i,
                M,
                pivot_val,
                pivot_val_dA,
                pivot_val_dB,
            )
            (
                arr1001_dA[d][i - d - 1][params],
                arr1001_dB[d][i - d - 1][params],
            ) = calc_dA_dB(
                2 * i + 1,
                G_in_dA,
                G_in_dB,
                G_in,
                A,
                B,
                K_l,
                K_i,
                M,
                pivot_val,
                pivot_val_dA,
                pivot_val_dB,
            )

    return (
        arr0_dA,
        arr2_dA,
        arr1010_dA,
        arr1001_dA,
        arr0_dB,
        arr2_dB,
        arr1010_dB,
        arr1001_dB,
    )


@njit(cache=True)
def use_diag_pivot_grad(A, B, M, cutoffs, params, arr0, arr1, arr0_dA, arr1_dA, arr0_dB, arr1_dB):
    """
    Apply recurrence relation for pivot of type [a,a,b,b,c,c...]
    Args:
        A, B (array, vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of modes
        cutoffs (tuple): upper bounds for the number of photons in each mode
        params (tuple): (a,b,c,...)
        arr0, arr1 (array, array): submatrices of the fock representation
        arr..._dA, arr..._dB (array, array): derivatives of submatrices w.r.t A and B
    Returns:
        (array, array): updated versions of arr0, arr1
    """
    pivot = repeat_twice(params)
    K_l = SQRT[pivot]
    K_i = SQRT[pivot + 1]
    G_in = np.zeros(2 * M, dtype=np.complex128)
    G_in_dA = np.zeros((2 * M, *A.shape), dtype=np.complex128)
    G_in_dB = np.zeros((2 * M, *B.shape), dtype=np.complex128)

    ########## READ ##########
    pivot_val = arr0[params]
    pivot_val_dA = arr0_dA[params]
    pivot_val_dB = arr0_dB[params]

    # Array1
    for i in range(2 * M):
        if params[i // 2] > 0:
            i_staggered = i + 1 - 2 * (i % 2)  # [i+1-2*(i%2) for i in range(6)] == [1,0,3,2,5,4]
            params_adapted = tuple_setitem(params, i // 2, params[i // 2] - 1)
            G_in[i] = arr1[i_staggered][params_adapted]
            G_in_dA[i] = arr1_dA[i_staggered][params_adapted]
            G_in_dB[i] = arr1_dB[i_staggered][params_adapted]

    ########## WRITE ##########
    G_in = np.multiply(K_l, G_in)

    # Array1
    for i in range(2 * M):
        if params[i // 2] + 1 < cutoffs[i // 2] and (i != 1 or params[0] + 2 < cutoffs[0]):
            # this if statement prevents a few elements from being written that will never be read
            arr1_dA[i][params], arr1_dB[i][params] = calc_dA_dB(
                i,
                G_in_dA,
                G_in_dB,
                G_in,
                A,
                B,
                K_l,
                K_i,
                M,
                pivot_val,
                pivot_val_dA,
                pivot_val_dB,
            )

    return arr1_dA, arr1_dB


@njit(cache=True)
def fock_representation_diagonal_grad_NUMBA(
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
):
    """
    Returns the gradients of the PNR probabilities of a mixed state according to algorithm 1 of
    https://doi.org/10.22331/q-2023-08-29-1097
    Args:
        A, B (array, vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of modes
        cutoffs (tuple): upper bounds for the number of photons in each mode
        arr0 (array): submatrix of the fock representation that contains Fock amplitudes of the type [a,a,b,b,c,c...]
        arr2 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+2,a,b,b,c,c...] / [a,a,b+2,b,c,c...] / ...
        arr1010 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b+1,b,c,c,...] / [a+1,a,b,b,c+1,c,...] / [a,a,b+1,b,c+1,c,...] / ...
        arr1001 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b,b+1,c,c,...] / [a+1,a,b,b,c,c+1,...] / [a,a,b+1,b,c,c+1,...] / ...
        arr1 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b,b,c,c...] / [a,a+1,b,b,c,c...] / [a,a,b+1,b,c,c...] / ...
        tuple_type, list_type (Numba types): numba types that need to be defined outside of Numba compiled functions
    Returns:
        array: the derivatives of the fock representation w.r.t. A and B
    """
    arr0_dA = np.zeros(arr0.shape + A.shape, dtype=np.complex128)
    arr2_dA = np.zeros(arr2.shape + A.shape, dtype=np.complex128)
    arr1010_dA = np.zeros(arr1010.shape + A.shape, dtype=np.complex128)
    arr1001_dA = np.zeros(arr1001.shape + A.shape, dtype=np.complex128)
    arr1_dA = np.zeros(arr1.shape + A.shape, dtype=np.complex128)
    arr0_dB = np.zeros(arr0.shape + B.shape, dtype=np.complex128)
    arr2_dB = np.zeros(arr2.shape + B.shape, dtype=np.complex128)
    arr1010_dB = np.zeros(arr1010.shape + B.shape, dtype=np.complex128)
    arr1001_dB = np.zeros(arr1001.shape + B.shape, dtype=np.complex128)
    arr1_dB = np.zeros(arr1.shape + B.shape, dtype=np.complex128)

    dict_params = construct_dict_params(cutoffs, tuple_type, list_type)
    for sum_params in range(sum(cutoffs)):
        for params in dict_params[sum_params]:
            # diagonal pivots: aa,bb,cc,dd,...
            if (cutoffs[0] == 1) or (params[0] < cutoffs[0] - 1):
                arr1_dA, arr1_dB = use_diag_pivot_grad(
                    A,
                    B,
                    M,
                    cutoffs,
                    params,
                    arr0,
                    arr1,
                    arr0_dA,
                    arr1_dA,
                    arr0_dB,
                    arr1_dB,
                )
            # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: 00,(b+1)b,cc,dd | 00,00,(c+1)c,dd | ...
            for d in range(M):
                if np.all(np.array(params)[:d] == 0) and (params[d] < cutoffs[d] - 1):
                    (
                        arr0_dA,
                        arr2_dA,
                        arr1010_dA,
                        arr1001_dA,
                        arr0_dB,
                        arr2_dB,
                        arr1010_dB,
                        arr1001_dB,
                    ) = use_offDiag_pivot_grad(
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
                        arr0_dA,
                        arr2_dA,
                        arr1010_dA,
                        arr1001_dA,
                        arr1_dA,
                        arr0_dB,
                        arr2_dB,
                        arr1010_dB,
                        arr1001_dB,
                        arr1_dB,
                    )
    return arr0_dA, arr0_dB


def fock_representation_diagonal_grad(A, B, M, arr0, arr2, arr1010, arr1001, arr1):
    """
    First initialise some Numba types (needs to be done outside of Numba compiled function)
    Then calculate the fock representation.
    """

    cutoffs = arr0.shape
    tuple_type = numba.types.UniTuple(int64, M)
    list_type = numba.types.ListType(tuple_type)
    return fock_representation_diagonal_grad_NUMBA(
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

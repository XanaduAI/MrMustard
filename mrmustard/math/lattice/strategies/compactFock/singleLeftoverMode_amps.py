"""
This module calculates all possible Fock representations of mode 0,where all other modes are PNR detected.
This is done by applying the recursion relation in a selective manner.
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

# ruff: noqa: RUF005


@njit(cache=True)
def write_block(
    i,
    arr_write,
    write,
    arr_read_pivot,
    read_GB,
    G_in,
    GB,
    A,
    K_i,
    cutoff_leftoverMode,
):  # pragma: no cover
    """
    Apply the recurrence relation to blocks of Fock amplitudes (of shape cutoff_leftoverMode x cutoff_leftoverMode)
    This is the coarse-grained version of applying the recurrence relation of mrmustard.math.compactFock.compactFock_diagonal_amps once.
    """
    m, n = 0, 0
    A_adapted = A[i, 2:]
    G_in_adapted = G_in[0, 0]
    arr_write[(0, 0) + write] = (GB[0, 0, i] + A_adapted @ G_in_adapted) / K_i[i - 2]

    m = 0
    A_adapted = A[i, 1:]
    for n in range(1, cutoff_leftoverMode):
        G_in_adapted = np.hstack(
            (np.array([arr_read_pivot[(0, n - 1) + read_GB] * np.sqrt(n)]), G_in[0, n]),
        )
        arr_write[(0, n) + write] = (GB[0, n, i] + A_adapted @ G_in_adapted) / K_i[i - 2]

    n = 0
    A_adapted = np.hstack((np.array([A[i, 0]]), A[i, 2:]))
    for m in range(1, cutoff_leftoverMode):
        G_in_adapted = np.hstack(
            (np.array([arr_read_pivot[(m - 1, 0) + read_GB] * np.sqrt(m)]), G_in[m, 0]),
        )
        arr_write[(m, 0) + write] = (GB[m, 0, i] + A_adapted @ G_in_adapted) / K_i[i - 2]

    A_adapted = A[i]
    for m in range(1, cutoff_leftoverMode):
        for n in range(1, cutoff_leftoverMode):
            G_in_adapted = np.hstack(
                (
                    np.array(
                        [
                            arr_read_pivot[(m - 1, n) + read_GB] * np.sqrt(m),
                            arr_read_pivot[(m, n - 1) + read_GB] * np.sqrt(n),
                        ],
                    ),
                    G_in[m, n],
                ),
            )
            arr_write[(m, n) + write] = (GB[m, n, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
    return arr_write


@njit(cache=True)
def read_block(
    arr_write,
    idx_write,
    arr_read,
    idx_read_tail,
    cutoff_leftoverMode,
):  # pragma: no cover
    """
    Read the blocks of Fock amplitudes (of shape cutoff_leftoverMode x cutoff_leftoverMode)
    that are required to apply the recurrence relation and write them to G_in
    """
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            arr_write[m, n, idx_write] = arr_read[
                (
                    m,
                    n,
                )
                + idx_read_tail
            ]
    return arr_write


@njit(cache=True)
def use_offDiag_pivot(
    A,
    B,
    M,
    cutoff_leftoverMode,
    cutoffs_tail,
    params,
    d,
    arr0,
    arr2,
    arr1010,
    arr1001,
    arr1,
):
    """
    Apply recurrence relation for pivot of type [a+1,a,b,b,c,c,...] / [a,a,b+1,b,c,c,...] / [a,a,b,b,c+1,c,...]
    Args:
        A, B (array, Vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of detected modes
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
    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)

    ########## READ ##########
    read_GB = (2 * d,) + params
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr1[(m, n) + read_GB] * B

    # Array0
    G_in = read_block(G_in, 2 * d, arr0, params, cutoff_leftoverMode)

    # read from Array2
    if params[d] > 0:
        params_adapted = tuple_setitem(params, d, params[d] - 1)
        G_in = read_block(G_in, 2 * d + 1, arr2, (d,) + params_adapted, cutoff_leftoverMode)

    # read from Array11
    for i in range(d + 1, M):  # i>d
        if params[i] > 0:
            params_adapted = tuple_setitem(params, i, params[i] - 1)
            G_in = read_block(
                G_in,
                2 * i,
                arr1001,
                (d, i - d - 1) + params_adapted,
                cutoff_leftoverMode,
            )
            G_in = read_block(
                G_in,
                2 * i + 1,
                arr1010,
                (d, i - d - 1) + params_adapted,
                cutoff_leftoverMode,
            )

    ########## WRITE ##########
    G_in = np.multiply(K_l, G_in)

    # Array0
    params_adapted = tuple_setitem(params, d, params[d] + 1)
    write = params_adapted
    arr0 = write_block(2 * d + 3, arr0, write, arr1, read_GB, G_in, GB, A, K_i, cutoff_leftoverMode)

    # Array2
    if params[d] + 2 < cutoffs_tail[d]:
        write = (d,) + params
        arr2 = write_block(
            2 * d + 2,
            arr2,
            write,
            arr1,
            read_GB,
            G_in,
            GB,
            A,
            K_i,
            cutoff_leftoverMode,
        )

    # Array11
    for i in range(d + 1, M):
        if params[i] + 1 < cutoffs_tail[i]:
            write = (d, i - d - 1) + params
            arr1010 = write_block(
                2 * i + 2,
                arr1010,
                write,
                arr1,
                read_GB,
                G_in,
                GB,
                A,
                K_i,
                cutoff_leftoverMode,
            )
            arr1001 = write_block(
                2 * i + 3,
                arr1001,
                write,
                arr1,
                read_GB,
                G_in,
                GB,
                A,
                K_i,
                cutoff_leftoverMode,
            )

    return arr0, arr2, arr1010, arr1001


@njit(cache=True)
def use_diag_pivot(A, B, M, cutoff_leftoverMode, cutoffs_tail, params, arr0, arr1):
    """
    Apply recurrence relation for pivot of type [a,a,b,b,c,c...]
    Args:
        A, B (array, Vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of detected modes
        cutoffs (tuple): upper bounds for the number of photons in each mode
        params (tuple): (a,b,c,...)
        arr0, arr1 (array, array): submatrices of the fock representation
    Returns:
        (array, array): updated versions of arr0, arr1
    """
    pivot = repeat_twice(params)
    K_l = SQRT[pivot]
    K_i = SQRT[pivot + 1]
    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)

    ########## READ ##########
    read_GB = params
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr0[(m, n) + read_GB] * B

    # Array1
    for i in range(2 * M):
        if params[i // 2] > 0:
            params_adapted = tuple_setitem(params, i // 2, params[i // 2] - 1)
            G_in = read_block(
                G_in,
                i,
                arr1,
                (i + 1 - 2 * (i % 2),) + params_adapted,
                cutoff_leftoverMode,
            )  # [i+1-2*(i%2) for i in range(6)] == [1,0,3,2,5,4]

    ########## WRITE ##########
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n] = np.multiply(K_l, G_in[m, n])

    # Array1
    for i in range(2 * M):
        if params[i // 2] + 1 < cutoffs_tail[i // 2] and (
            i != 1 or params[0] + 2 < cutoffs_tail[0]
        ):
            # this if statement prevents a few elements from being written that will never be read
            write = (i,) + params
            arr1 = write_block(
                i + 2,
                arr1,
                write,
                arr0,
                read_GB,
                G_in,
                GB,
                A,
                K_i,
                cutoff_leftoverMode,
            )

    return arr1


@njit(cache=True)
def fock_representation_1leftoverMode_amps_NUMBA(
    A,
    B,
    M,
    cutoff_leftoverMode,
    cutoffs_tail,
    arr0,
    arr2,
    arr1010,
    arr1001,
    arr1,
    tuple_type,
    list_type,
    zero_tuple,
):
    """
    Returns the density matrices in the upper, undetected mode of a circuit when all other modes are PNR detected
    according to algorithm 2 of https://doi.org/10.22331/q-2023-08-29-1097
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
        tuple_type, list_type (Numba types): numba types that need to be defined outside of numba compiled functions
    Returns:
        Tensor: the fock representation
    """

    # fill first mode for all PNR detections equal to zero
    for m in range(cutoff_leftoverMode - 1):
        arr0[(m + 1, 0) + zero_tuple] = (
            arr0[(m, 0) + zero_tuple] * B[0] + np.sqrt(m) * A[0, 0] * arr0[(m - 1, 0) + zero_tuple]
        ) / np.sqrt(m + 1)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode - 1):
            arr0[(m, n + 1) + zero_tuple] = (
                arr0[(m, n) + zero_tuple] * B[1]
                + np.sqrt(m) * A[1, 0] * arr0[(m - 1, n) + zero_tuple]
                + np.sqrt(n) * A[1, 1] * arr0[(m, n - 1) + zero_tuple]
            ) / np.sqrt(n + 1)

    dict_params = construct_dict_params(cutoffs_tail, tuple_type, list_type)
    for sum_params in range(sum(cutoffs_tail)):
        for params in dict_params[sum_params]:
            # diagonal pivots: aa,bb,cc,dd,...
            if (cutoffs_tail[0] == 1) or (params[0] < cutoffs_tail[0] - 1):
                arr1 = use_diag_pivot(
                    A,
                    B,
                    M - 1,
                    cutoff_leftoverMode,
                    cutoffs_tail,
                    params,
                    arr0,
                    arr1,
                )
            # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: 00,(b+1)b,cc,dd | 00,00,(c+1)c,dd | ...
            for d in range(M - 1):
                if np.all(np.array(params)[:d] == 0) and (params[d] < cutoffs_tail[d] - 1):
                    arr0, arr2, arr1010, arr1001 = use_offDiag_pivot(
                        A,
                        B,
                        M - 1,
                        cutoff_leftoverMode,
                        cutoffs_tail,
                        params,
                        d,
                        arr0,
                        arr2,
                        arr1010,
                        arr1001,
                        arr1,
                    )
    return arr0, arr2, arr1010, arr1001, arr1


def fock_representation_1leftoverMode_amps(A, B, G0, M, cutoffs):
    """
    First initialise the submatrices of G (of which the shape depends on cutoff and M)
    and some other constants
    (These initialisations currently cannot be done using Numba.)
    Then calculate the fock representation.
    """

    cutoff_leftoverMode = cutoffs[0]
    cutoffs_tail = tuple(cutoffs[1:])
    tuple_type = numba.types.UniTuple(int64, M - 1)
    list_type = numba.types.ListType(tuple_type)
    zero_tuple = (0,) * (M - 1)

    arr0 = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode) + cutoffs_tail, dtype=np.complex128)
    arr0[(0,) * (M + 1)] = G0
    arr2 = np.zeros(
        (cutoff_leftoverMode, cutoff_leftoverMode) + (M - 1,) + cutoffs_tail,
        dtype=np.complex128,
    )
    arr1 = np.zeros(
        (cutoff_leftoverMode, cutoff_leftoverMode) + (2 * (M - 1),) + cutoffs_tail,
        dtype=np.complex128,
    )
    if M == 2:
        arr1010 = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
        arr1001 = np.zeros((1, 1, 1, 1, 1), dtype=np.complex128)
    else:
        arr1010 = np.zeros(
            (cutoff_leftoverMode, cutoff_leftoverMode) + (M - 1, M - 2) + cutoffs_tail,
            dtype=np.complex128,
        )
        arr1001 = np.zeros(
            (cutoff_leftoverMode, cutoff_leftoverMode) + (M - 1, M - 2) + cutoffs_tail,
            dtype=np.complex128,
        )
    return fock_representation_1leftoverMode_amps_NUMBA(
        A,
        B,
        M,
        cutoff_leftoverMode,
        cutoffs_tail,
        arr0,
        arr2,
        arr1010,
        arr1001,
        arr1,
        tuple_type,
        list_type,
        zero_tuple,
    )

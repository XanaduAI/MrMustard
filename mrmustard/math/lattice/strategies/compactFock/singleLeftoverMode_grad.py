"""
This module calculates the derivatives for all possible Fock representations of mode 0, where all other modes are PNR detected.
This is done by applying the derivated recursion relation in a selective manner.
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
def calc_dA_dB(
    m,
    n,
    i,
    arr_read_pivot,
    read_GB,
    G_in_adapted,
    A_adapted,
    B,
    K_i,
    K_l_adapted,
    arr_read_pivot_dA,
    G_in_dA_adapted,
    arr_read_pivot_dB,
    G_in_dB_adapted,
    l_range,
):
    """
    Apply the derivated recurrence relation.
    """
    dA = arr_read_pivot_dA[(m, n, *read_GB)] * B[i]
    dB = arr_read_pivot_dB[(m, n, *read_GB)] * B[i]
    dB[i] += arr_read_pivot[(m, n, *read_GB)]
    for l_prime, l in enumerate(l_range):
        dA += K_l_adapted[l_prime] * A_adapted[l_prime] * G_in_dA_adapted[l_prime]
        dB += K_l_adapted[l_prime] * A_adapted[l_prime] * G_in_dB_adapted[l_prime]
        dA[i, l] += G_in_adapted[l_prime]
    return dA / K_i[i - 2], dB / K_i[i - 2]


@njit(cache=True)
def write_block_grad(
    i,
    write,
    arr_read_pivot,
    read_GB,
    G_in,
    A,
    B,
    K_i,
    K_l,
    cutoff_leftoverMode,
    arr_write_dA,
    arr_read_pivot_dA,
    G_in_dA,
    arr_write_dB,
    arr_read_pivot_dB,
    G_in_dB,
):
    """
    Apply the derivated recurrence relation to blocks of Fock amplitudes (of shape cutoff_leftoverMode x cutoff_leftoverMode)
    This is the coarse-grained version of applying the derivated recurrence relation of mrmustard.math.compactFock.compactFock_diagonal_grad once.
    """
    m, n = 0, 0
    l_range = np.arange(2, A.shape[1])
    A_adapted = A[i, 2:]
    G_in_adapted = G_in[0, 0]
    G_in_dA_adapted = G_in_dA[0, 0]
    G_in_dB_adapted = G_in_dB[0, 0]
    K_l_adapted = K_l
    arr_write_dA[(0, 0, *write)], arr_write_dB[(0, 0, *write)] = calc_dA_dB(
        m,
        n,
        i,
        arr_read_pivot,
        read_GB,
        G_in_adapted,
        A_adapted,
        B,
        K_i,
        K_l_adapted,
        arr_read_pivot_dA,
        G_in_dA_adapted,
        arr_read_pivot_dB,
        G_in_dB_adapted,
        l_range,
    )
    m = 0
    l_range = np.arange(1, A.shape[1])
    A_adapted = A[i, 1:]
    for n in range(1, cutoff_leftoverMode):
        K_l_adapted = np.hstack((np.array([np.sqrt(n)]), K_l))
        G_in_adapted = np.hstack(
            (np.array([arr_read_pivot[(0, n - 1, *read_GB)] * np.sqrt(n)]), G_in[0, n]),
        )
        G_in_dA_adapted = np.concatenate(
            (
                np.expand_dims(arr_read_pivot_dA[(0, n - 1, *read_GB)], axis=0),
                G_in_dA[0, n],
            ),
            axis=0,
        )
        G_in_dB_adapted = np.concatenate(
            (
                np.expand_dims(arr_read_pivot_dB[(0, n - 1, *read_GB)], axis=0),
                G_in_dB[0, n],
            ),
            axis=0,
        )
        arr_write_dA[(0, n, *write)], arr_write_dB[(0, n, *write)] = calc_dA_dB(
            m,
            n,
            i,
            arr_read_pivot,
            read_GB,
            G_in_adapted,
            A_adapted,
            B,
            K_i,
            K_l_adapted,
            arr_read_pivot_dA,
            G_in_dA_adapted,
            arr_read_pivot_dB,
            G_in_dB_adapted,
            l_range,
        )
    n = 0
    l_range = np.arange(1, A.shape[1])
    l_range[0] = 0
    A_adapted = np.hstack((np.array([A[i, 0]]), A[i, 2:]))
    for m in range(1, cutoff_leftoverMode):
        K_l_adapted = np.hstack((np.array([np.sqrt(m)]), K_l))
        G_in_adapted = np.hstack(
            (np.array([arr_read_pivot[(m - 1, 0, *read_GB)] * np.sqrt(m)]), G_in[m, 0]),
        )
        G_in_dA_adapted = np.concatenate(
            (
                np.expand_dims(arr_read_pivot_dA[(m - 1, 0, *read_GB)], axis=0),
                G_in_dA[m, 0],
            ),
            axis=0,
        )
        G_in_dB_adapted = np.concatenate(
            (
                np.expand_dims(arr_read_pivot_dB[(m - 1, 0, *read_GB)], axis=0),
                G_in_dB[m, 0],
            ),
            axis=0,
        )
        arr_write_dA[(m, 0, *write)], arr_write_dB[(m, 0, *write)] = calc_dA_dB(
            m,
            n,
            i,
            arr_read_pivot,
            read_GB,
            G_in_adapted,
            A_adapted,
            B,
            K_i,
            K_l_adapted,
            arr_read_pivot_dA,
            G_in_dA_adapted,
            arr_read_pivot_dB,
            G_in_dB_adapted,
            l_range,
        )
    # m>0,n>0
    l_range = np.arange(A.shape[1])
    A_adapted = A[i]
    for m in range(1, cutoff_leftoverMode):
        for n in range(1, cutoff_leftoverMode):
            K_l_adapted = np.hstack((np.array([np.sqrt(m), np.sqrt(n)]), K_l))
            G_in_adapted = np.hstack(
                (
                    np.array(
                        [
                            arr_read_pivot[(m - 1, n, *read_GB)] * np.sqrt(m),
                            arr_read_pivot[(m, n - 1, *read_GB)] * np.sqrt(n),
                        ],
                    ),
                    G_in[m, n],
                ),
            )
            G_in_dA_adapted = np.concatenate(
                (
                    np.expand_dims(arr_read_pivot_dA[(m - 1, n, *read_GB)], axis=0),
                    np.expand_dims(arr_read_pivot_dA[(m, n - 1, *read_GB)], axis=0),
                    G_in_dA[m, n],
                ),
                axis=0,
            )
            G_in_dB_adapted = np.concatenate(
                (
                    np.expand_dims(arr_read_pivot_dB[(m - 1, n, *read_GB)], axis=0),
                    np.expand_dims(arr_read_pivot_dB[(m, n - 1, *read_GB)], axis=0),
                    G_in_dB[m, n],
                ),
                axis=0,
            )
            arr_write_dA[(m, n, *write)], arr_write_dB[(m, n, *write)] = calc_dA_dB(
                m,
                n,
                i,
                arr_read_pivot,
                read_GB,
                G_in_adapted,
                A_adapted,
                B,
                K_i,
                K_l_adapted,
                arr_read_pivot_dA,
                G_in_dA_adapted,
                arr_read_pivot_dB,
                G_in_dB_adapted,
                l_range,
            )
    return arr_write_dA, arr_write_dB


@njit(cache=True)
def read_block(
    arr_write,
    arr_write_dA,
    arr_write_dB,
    idx_write,
    arr_read,
    arr_read_dA,
    arr_read_dB,
    idx_read_tail,
    cutoff_leftoverMode,
):
    """
    Read the blocks of Fock amplitudes(of shape cutoff_leftoverMode x cutoff_leftoverMode)
    and their derivatives w.r.t A and B and write them to G_in, G_in_dA, G_in_dB
    """
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            arr_write[m, n, idx_write] = arr_read[(m, n, *idx_read_tail)]
            arr_write_dA[m, n, idx_write] = arr_read_dA[(m, n, *idx_read_tail)]
            arr_write_dB[m, n, idx_write] = arr_read_dB[(m, n, *idx_read_tail)]

    return arr_write, arr_write_dA, arr_write_dB


@njit(cache=True)
def use_offDiag_pivot_grad(  # noqa: C901
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
        arr..._dA, arr..._dB (array, array): derivatives of submatrices w.r.t A and B
    Returns:
        (array, array, array, array, array): updated versions of arr0, arr2, arr1010, arr1001, arr1
    """

    pivot = repeat_twice(params)
    pivot[2 * d] += 1
    K_l = SQRT[pivot]
    K_i = SQRT[pivot + 1]
    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)
    G_in_dA = np.zeros(G_in.shape + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros(G_in.shape + B.shape, dtype=np.complex128)

    ########## READ ##########
    read_GB = (2 * d, *params)
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr1[(m, n, *read_GB)] * B

    # Array0
    G_in, G_in_dA, G_in_dB = read_block(
        G_in,
        G_in_dA,
        G_in_dB,
        2 * d,
        arr0,
        arr0_dA,
        arr0_dB,
        params,
        cutoff_leftoverMode,
    )

    # read from Array2
    if params[d] > 0:
        params_adapted = tuple_setitem(params, d, params[d] - 1)
        G_in, G_in_dA, G_in_dB = read_block(
            G_in,
            G_in_dA,
            G_in_dB,
            2 * d + 1,
            arr2,
            arr2_dA,
            arr2_dB,
            (d, *params_adapted),
            cutoff_leftoverMode,
        )

    # read from Array11
    for i in range(d + 1, M):  # i>d
        if params[i] > 0:
            params_adapted = tuple_setitem(params, i, params[i] - 1)
            G_in, G_in_dA, G_in_dB = read_block(
                G_in,
                G_in_dA,
                G_in_dB,
                2 * i,
                arr1001,
                arr1001_dA,
                arr1001_dB,
                (d, i - d - 1, *params_adapted),
                cutoff_leftoverMode,
            )
            G_in, G_in_dA, G_in_dB = read_block(
                G_in,
                G_in_dA,
                G_in_dB,
                2 * i + 1,
                arr1010,
                arr1010_dA,
                arr1010_dB,
                (d, i - d - 1, *params_adapted),
                cutoff_leftoverMode,
            )

    ########## WRITE ##########
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n] = np.multiply(K_l, G_in[m, n])

    # Array0
    write = tuple_setitem(params, d, params[d] + 1)
    arr0_dA, arr0_dB = write_block_grad(
        2 * d + 3,
        write,
        arr1,
        read_GB,
        G_in,
        A,
        B,
        K_i,
        K_l,
        cutoff_leftoverMode,
        arr0_dA,
        arr1_dA,
        G_in_dA,
        arr0_dB,
        arr1_dB,
        G_in_dB,
    )

    # Array2
    if params[d] + 2 < cutoffs_tail[d]:
        write = (d, *params)
        arr2_dA, arr2_dB = write_block_grad(
            2 * d + 2,
            write,
            arr1,
            read_GB,
            G_in,
            A,
            B,
            K_i,
            K_l,
            cutoff_leftoverMode,
            arr2_dA,
            arr1_dA,
            G_in_dA,
            arr2_dB,
            arr1_dB,
            G_in_dB,
        )

    # Array11
    for i in range(d + 1, M):
        if params[i] + 1 < cutoffs_tail[i]:
            write = (d, i - d - 1, *params)
            arr1010_dA, arr1010_dB = write_block_grad(
                2 * i + 2,
                write,
                arr1,
                read_GB,
                G_in,
                A,
                B,
                K_i,
                K_l,
                cutoff_leftoverMode,
                arr1010_dA,
                arr1_dA,
                G_in_dA,
                arr1010_dB,
                arr1_dB,
                G_in_dB,
            )
            arr1001_dA, arr1001_dB = write_block_grad(
                2 * i + 3,
                write,
                arr1,
                read_GB,
                G_in,
                A,
                B,
                K_i,
                K_l,
                cutoff_leftoverMode,
                arr1001_dA,
                arr1_dA,
                G_in_dA,
                arr1001_dB,
                arr1_dB,
                G_in_dB,
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
def use_diag_pivot_grad(
    A,
    B,
    M,
    cutoff_leftoverMode,
    cutoffs_tail,
    params,
    arr0,
    arr1,
    arr0_dA,
    arr1_dA,
    arr0_dB,
    arr1_dB,
):
    """
    Apply recurrence relation for pivot of type [a,a,b,b,c,c...]
    Args:
        A, B (array, Vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of detected modes
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
    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)
    G_in_dA = np.zeros(G_in.shape + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros(G_in.shape + B.shape, dtype=np.complex128)

    ########## READ ##########
    read_GB = params
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr0[(m, n, *read_GB)] * B

    # Array1
    for i in range(2 * M):
        if params[i // 2] > 0:
            params_adapted = tuple_setitem(params, i // 2, params[i // 2] - 1)
            read = (
                i + 1 - 2 * (i % 2),
                *params_adapted,
            )  # [i+1-2*(i%2) for i in range(6)] == [1,0,3,2,5,4]
            G_in, G_in_dA, G_in_dB = read_block(
                G_in,
                G_in_dA,
                G_in_dB,
                i,
                arr1,
                arr1_dA,
                arr1_dB,
                read,
                cutoff_leftoverMode,
            )

    ########## WRITE ##########
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n] = np.multiply(K_l, G_in[m, n])

    # Array1
    for i in range(2 * M):
        if params[i // 2] + 1 < cutoffs_tail[i // 2] and (
            i != 1 or params[0] + 2 < cutoffs_tail[0]
        ):
            # this prevents a few elements from being written that will never be read
            write = (i, *params)
            arr1_dA, arr1_dB = write_block_grad(
                i + 2,
                write,
                arr0,
                read_GB,
                G_in,
                A,
                B,
                K_i,
                K_l,
                cutoff_leftoverMode,
                arr1_dA,
                arr0_dA,
                G_in_dA,
                arr1_dB,
                arr0_dB,
                G_in_dB,
            )

    return arr1_dA, arr1_dB


@njit(cache=True)
def fock_representation_1leftoverMode_grad_NUMBA(
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
    Returns the gradients of the density matrices in the upper, undetected mode of a circuit when all other modes
    are PNR detected (according to algorithm 2 of https://doi.org/10.22331/q-2023-08-29-1097)
    Args:
        A, B (array, Vector): required input for recurrence relation (given by mrmustard.physics.fock_utils.ABC)
        M (int): number of modes
        cutoffs (tuple): upper bounds for the number of photons in each mode
        arr0 (array): submatrix of the fock representation that contains Fock amplitudes of the type [a,a,b,b,c,c...]
        arr2 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+2,a,b,b,c,c...] / [a,a,b+2,b,c,c...] / ...
        arr1010 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b+1,b,c,c,...] / [a+1,a,b,b,c+1,c,...] / [a,a,b+1,b,c+1,c,...] / ...
        arr1001 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b,b+1,c,c,...] / [a+1,a,b,b,c,c+1,...] / [a,a,b+1,b,c,c+1,...] / ...
        arr1 (array): submatrix of the fock representation that contains Fock amplitudes of the types [a+1,a,b,b,c,c...] / [a,a+1,b,b,c,c...] / [a,a,b+1,b,c,c...] / ...
        tuple_type, list_type (numba types): numba types that need to be defined outside of numba compiled functions
    Returns:
        Tensor: the fock representation
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

    # fill first mode for all PNR detections equal to zero
    for m in range(cutoff_leftoverMode - 1):
        arr0_dA[(m + 1, 0, *zero_tuple)] = (
            arr0_dA[(m, 0, *zero_tuple)] * B[0]
            + np.sqrt(m) * A[0, 0] * arr0_dA[(m - 1, 0, *zero_tuple)]
        ) / np.sqrt(m + 1)
        arr0_dA[(m + 1, 0, *zero_tuple)][0, 0] += (
            np.sqrt(m) * arr0[(m - 1, 0, *zero_tuple)]
        ) / np.sqrt(m + 1)
        arr0_dB[(m + 1, 0, *zero_tuple)] = (
            arr0_dB[(m, 0, *zero_tuple)] * B[0]
            + np.sqrt(m) * A[0, 0] * arr0_dB[(m - 1, 0, *zero_tuple)]
        ) / np.sqrt(m + 1)
        arr0_dB[(m + 1, 0, *zero_tuple)][0] += arr0[(m, 0, *zero_tuple)] / np.sqrt(m + 1)

    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode - 1):
            arr0_dA[(m, n + 1, *zero_tuple)] = (
                arr0_dA[(m, n, *zero_tuple)] * B[1]
                + np.sqrt(m) * A[1, 0] * arr0_dA[(m - 1, n, *zero_tuple)]
                + np.sqrt(n) * A[1, 1] * arr0_dA[(m, n - 1, *zero_tuple)]
            ) / np.sqrt(n + 1)
            arr0_dA[(m, n + 1, *zero_tuple)][1, 0] += (
                np.sqrt(m) * arr0[(m - 1, n, *zero_tuple)]
            ) / np.sqrt(n + 1)
            arr0_dA[(m, n + 1, *zero_tuple)][1, 1] += (
                np.sqrt(n) * arr0[(m, n - 1, *zero_tuple)]
            ) / np.sqrt(n + 1)
            arr0_dB[(m, n + 1, *zero_tuple)] = (
                arr0_dB[(m, n, *zero_tuple)] * B[1]
                + np.sqrt(m) * A[1, 0] * arr0_dB[(m - 1, n, *zero_tuple)]
                + np.sqrt(n) * A[1, 1] * arr0_dB[(m, n - 1, *zero_tuple)]
            ) / np.sqrt(n + 1)
            arr0_dB[(m, n + 1, *zero_tuple)][1] += arr0[(m, n, *zero_tuple)] / np.sqrt(n + 1)

    dict_params = construct_dict_params(cutoffs_tail, tuple_type, list_type)
    for sum_params in range(sum(cutoffs_tail)):
        for params in dict_params[sum_params]:
            # diagonal pivots: aa,bb,cc,dd,...
            if (cutoffs_tail[0] == 1) or (params[0] < cutoffs_tail[0] - 1):
                arr1_dA, arr1_dB = use_diag_pivot_grad(
                    A,
                    B,
                    M - 1,
                    cutoff_leftoverMode,
                    cutoffs_tail,
                    params,
                    arr0,
                    arr1,
                    arr0_dA,
                    arr1_dA,
                    arr0_dB,
                    arr1_dB,
                )
            # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: 00,(b+1)b,cc,dd | 00,00,(c+1)c,dd | ...
            for d in range(M - 1):
                if np.all(np.array(params)[:d] == 0) and (params[d] < cutoffs_tail[d] - 1):
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


def fock_representation_1leftoverMode_grad(A, B, M, arr0, arr2, arr1010, arr1001, arr1):
    """
    First initialise the submatrices of G (of which the shape depends on cutoff and M)
    and some other constants
    (These initialisations currently cannot be done using Numba.)
    Then calculate the fock representation.
    """

    cutoffs = tuple(arr0.shape[1:])
    cutoff_leftoverMode = cutoffs[0]
    cutoffs_tail = tuple(cutoffs[1:])
    tuple_type = numba.types.UniTuple(int64, M - 1)
    list_type = numba.types.ListType(tuple_type)
    zero_tuple = (0,) * (M - 1)

    return fock_representation_1leftoverMode_grad_NUMBA(
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

import numpy as np
from numba import njit, typeof, int64
from numba.typed import Dict
from scipy.special import binom
from mrmustard.math.compactFock_helperFunctions import *

@njit
def calc_dA_dB(m, n, i, arr_read_pivot, read_GB, G_in_adapted, A_adapted, B, K_i, K_l_adapted, cutoff,
               cutoff_leftoverMode, arr_read_pivot_dA, G_in_dA_adapted, arr_read_pivot_dB, G_in_dB_adapted, l_range):
    dA = arr_read_pivot_dA[m, n, read_GB] * B[i]
    dB = arr_read_pivot_dB[m, n, read_GB] * B[i]
    dB[i] += arr_read_pivot[m, n, read_GB]
    for l_prime, l in enumerate(l_range):
        dA += K_l_adapted[l_prime] * A_adapted[l_prime] * G_in_dA_adapted[l_prime]
        dB += K_l_adapted[l_prime] * A_adapted[l_prime] * G_in_dB_adapted[l_prime]
        dA[i, l] += G_in_adapted[l_prime]
    return dA / K_i[i - 2], dB / K_i[i - 2]


@njit
def write_block(i, arr_write, write, arr_read_pivot, read_GB, G_in, GB, A, B, K_i, K_l, cutoff, cutoff_leftoverMode,
                arr_write_dA, arr_read_pivot_dA, G_in_dA, arr_write_dB, arr_read_pivot_dB, G_in_dB):
    # m,n = 0,0
    l_range = np.arange(2, A.shape[1])
    A_adapted = A[i, 2:]
    G_in_adapted = G_in[0, 0]
    G_in_dA_adapted = G_in_dA[0, 0]
    G_in_dB_adapted = G_in_dB[0, 0]
    K_l_adapted = K_l
    arr_write[0, 0, write] = (GB[0, 0, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
    arr_write_dA[0, 0, write], arr_write_dB[0, 0, write] = calc_dA_dB(0, 0, i, arr_read_pivot, read_GB, G_in_adapted,
                                                                      A_adapted, B, K_i, K_l_adapted, cutoff,
                                                                      cutoff_leftoverMode, arr_read_pivot_dA,
                                                                      G_in_dA_adapted, arr_read_pivot_dB,
                                                                      G_in_dB_adapted, l_range)

    # m=0
    l_range = np.arange(1, A.shape[1])
    A_adapted = A[i, 1:]
    for n in range(1, cutoff_leftoverMode):
        K_l_adapted = np.hstack((np.array([np.sqrt(n)]), K_l))
        G_in_adapted = np.hstack((np.array([arr_read_pivot[0, n - 1, read_GB] * np.sqrt(n)]), G_in[0, n]))
        G_in_dA_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dA[0, n - 1, read_GB], axis=0), G_in_dA[0, n]),
                                         axis=0)
        G_in_dB_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dB[0, n - 1, read_GB], axis=0), G_in_dB[0, n]),
                                         axis=0)
        arr_write[0, n, write] = (GB[0, n, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
        arr_write_dA[0, n, write], arr_write_dB[0, n, write] = calc_dA_dB(0, n, i, arr_read_pivot, read_GB,
                                                                          G_in_adapted, A_adapted, B, K_i, K_l_adapted,
                                                                          cutoff, cutoff_leftoverMode,
                                                                          arr_read_pivot_dA, G_in_dA_adapted,
                                                                          arr_read_pivot_dB, G_in_dB_adapted, l_range)

    # n=0
    l_range = np.arange(1, A.shape[1])
    l_range[0] = 0
    A_adapted = np.hstack((np.array([A[i, 0]]), A[i, 2:]))
    for m in range(1, cutoff_leftoverMode):
        K_l_adapted = np.hstack((np.array([np.sqrt(m)]), K_l))
        G_in_adapted = np.hstack((np.array([arr_read_pivot[m - 1, 0, read_GB] * np.sqrt(m)]), G_in[m, 0]))
        G_in_dA_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dA[m - 1, 0, read_GB], axis=0), G_in_dA[m, 0]),
                                         axis=0)
        G_in_dB_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dB[m - 1, 0, read_GB], axis=0), G_in_dB[m, 0]),
                                         axis=0)
        arr_write[m, 0, write] = (GB[m, 0, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
        arr_write_dA[m, 0, write], arr_write_dB[m, 0, write] = calc_dA_dB(m, 0, i, arr_read_pivot, read_GB,
                                                                          G_in_adapted, A_adapted, B, K_i, K_l_adapted,
                                                                          cutoff, cutoff_leftoverMode,
                                                                          arr_read_pivot_dA, G_in_dA_adapted,
                                                                          arr_read_pivot_dB, G_in_dB_adapted, l_range)

    # m>0,n>0
    l_range = np.arange(A.shape[1])
    A_adapted = A[i]
    for m in range(1, cutoff_leftoverMode):
        for n in range(1, cutoff_leftoverMode):
            K_l_adapted = np.hstack((np.array([np.sqrt(m), np.sqrt(n)]), K_l))
            G_in_adapted = np.hstack((np.array(
                [arr_read_pivot[m - 1, n, read_GB] * np.sqrt(m), arr_read_pivot[m, n - 1, read_GB] * np.sqrt(n)]),
                                      G_in[m, n]))
            G_in_dA_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dA[m - 1, n, read_GB], axis=0),
                                              np.expand_dims(arr_read_pivot_dA[m, n - 1, read_GB], axis=0),
                                              G_in_dA[m, n]), axis=0)
            G_in_dB_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dB[m - 1, n, read_GB], axis=0),
                                              np.expand_dims(arr_read_pivot_dB[m, n - 1, read_GB], axis=0),
                                              G_in_dB[m, n]), axis=0)
            arr_write[m, n, write] = (GB[m, n, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
            arr_write_dA[m, n, write], arr_write_dB[m, n, write] = calc_dA_dB(m, n, i, arr_read_pivot, read_GB,
                                                                              G_in_adapted, A_adapted, B, K_i,
                                                                              K_l_adapted, cutoff, cutoff_leftoverMode,
                                                                              arr_read_pivot_dA, G_in_dA_adapted,
                                                                              arr_read_pivot_dB, G_in_dB_adapted,
                                                                              l_range)

    return arr_write, arr_write_dA, arr_write_dB


@njit
def use_offDiag_pivot(A, B, M, cutoff, cutoff_leftoverMode, params, d, arr0, arr2, arr11, arr1, arr0_dA, arr2_dA,
                      arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB, strides0, strides2, strides11, strides1):
    pivot = calc_offDiag_pivot(params, d)
    K_l = np.sqrt(pivot)  # automatic conversion to float
    K_i = np.sqrt(pivot + 1)  # automatic conversion to float
    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)  # M is actually M-1 here
    G_in_dA = np.zeros(G_in.shape + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros(G_in.shape + B.shape, dtype=np.complex128)

    read_GB = strides1[1] * 2 * d + strides1[2] * params[d] + add_tuple_tail_Array2(d, params, M, strides1)
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr1[m, n, read_GB] * B

    ########## READ ##########

    # Array0
    read0 = add_tuple_tail_Array0(params, M,
                                  strides0)  # can store this one as I do not need to check boundary conditions for Array0! (Doesn't work for other arrays)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n, 2 * d] = arr0[m, n, read0]
            G_in_dA[m, n, 2 * d] = arr0_dA[m, n, read0]
            G_in_dB[m, n, 2 * d] = arr0_dB[m, n, read0]

    # read from Array2
    if params[d] > 0:  # params[d]-1>=0
        read = strides2[1] * d + strides2[2] * (params[d] - 1) + add_tuple_tail_Array2(d, params, M, strides2)
        for m in range(cutoff_leftoverMode):
            for n in range(cutoff_leftoverMode):
                G_in[m, n, 2 * d + 1] = arr2[m, n, read]
                G_in_dA[m, n, 2 * d + 1] = arr2_dA[m, n, read]
                G_in_dB[m, n, 2 * d + 1] = arr2_dB[m, n, read]

    # read from Array11
    for i in range(d + 1, M):  # i>d
        if params[i] > 0:
            read = strides11[1] * index_above_diagonal(d, i, M) + strides11[2] * params[d] + strides11[3] * (
                        params[i] - 1) + add_tuple_tail_Array11(d, i, params, M, strides11)
            for m in range(cutoff_leftoverMode):
                for n in range(cutoff_leftoverMode):
                    G_in[m, n, 2 * i] = arr11[m, n, read + strides11[0]]  # READ green (1001)
                    G_in_dA[m, n, 2 * i] = arr11_dA[m, n, read + strides11[0]]
                    G_in_dB[m, n, 2 * i] = arr11_dB[m, n, read + strides11[0]]
                    G_in[m, n, 2 * i + 1] = arr11[m, n, read]  # READ red (1010)
                    G_in_dA[m, n, 2 * i + 1] = arr11_dA[m, n, read]
                    G_in_dB[m, n, 2 * i + 1] = arr11_dB[m, n, read]

    for i in range(d):  # i<d
        if params[i] > 0:
            read = strides11[1] * index_above_diagonal(i, d, M) + strides11[2] * (params[i] - 1) + strides11[3] * \
                   params[d] + add_tuple_tail_Array11(i, d, params, M, strides11)
            for m in range(cutoff_leftoverMode):
                for n in range(cutoff_leftoverMode):
                    G_in[m, n, 2 * i] = arr11[m, n, read + strides11[0] * 2]  # READ blue (0110)
                    G_in_dA[m, n, 2 * i] = arr11_dA[m, n, read + strides11[0] * 2]
                    G_in_dB[m, n, 2 * i] = arr11_dB[m, n, read + strides11[0] * 2]
                    G_in[m, n, 2 * i + 1] = arr11[m, n, read]  # READ red (1010)
                    G_in_dA[m, n, 2 * i + 1] = arr11_dA[m, n, read]
                    G_in_dB[m, n, 2 * i + 1] = arr11_dB[m, n, read]

    ########## WRITE ##########

    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n] = np.multiply(K_l, G_in[m, n])

    # Array0
    if d == 0 or np.all(params[:d] == 0):
        write0 = read0 + strides0[2 + d]  # params[d] --> params[d]+1
        arr0, arr0_dA, arr0_dB = write_block(2 * d + 3, arr0, write0, arr1, read_GB, G_in, GB, A, B, K_i, K_l, cutoff,
                                             cutoff_leftoverMode, arr0_dA, arr1_dA, G_in_dA, arr0_dB, arr1_dB, G_in_dB)

    # Array2
    if params[d] + 2 < cutoff:
        write = strides2[1] * d + strides2[2] * params[d] + add_tuple_tail_Array2(d, params, M, strides2)
        arr2, arr2_dA, arr2_dB = write_block(2 * d + 2, arr2, write, arr1, read_GB, G_in, GB, A, B, K_i, K_l, cutoff,
                                             cutoff_leftoverMode, arr2_dA, arr1_dA, G_in_dA, arr2_dB, arr1_dB, G_in_dB)

    # Array11
    for i in range(d + 1, M):
        if params[i] + 1 < cutoff:
            write = strides11[1] * index_above_diagonal(d, i, M) + strides11[2] * params[d] + strides11[3] * params[
                i] + add_tuple_tail_Array11(d, i, params, M, strides11)
            arr11, arr11_dA, arr11_dB = write_block(2 * i + 2, arr11, write, arr1, read_GB, G_in, GB, A, B, K_i, K_l,
                                                    cutoff, cutoff_leftoverMode, arr11_dA, arr1_dA, G_in_dA, arr11_dB,
                                                    arr1_dB, G_in_dB)  # WRITE red (1010)
            arr11, arr11_dA, arr11_dB = write_block(2 * i + 3, arr11, write + strides11[0], arr1, read_GB, G_in, GB, A,
                                                    B, K_i, K_l, cutoff, cutoff_leftoverMode, arr11_dA, arr1_dA,
                                                    G_in_dA, arr11_dB, arr1_dB, G_in_dB)  # WRITE green (1001)

    for i in range(d):
        if params[i] + 1 < cutoff:
            write = strides11[1] * index_above_diagonal(i, d, M) + strides11[2] * params[i] + strides11[3] * params[
                d] + add_tuple_tail_Array11(i, d, params, M, strides11) + strides11[0] * 2
            arr11, arr11_dA, arr11_dB = write_block(2 * i + 3, arr11, write, arr1, read_GB, G_in, GB, A, B, K_i, K_l,
                                                    cutoff, cutoff_leftoverMode, arr11_dA, arr1_dA, G_in_dA, arr11_dB,
                                                    arr1_dB, G_in_dB)  # WRITE blue (0110)

    return arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB


@njit
def use_diag_pivot(A, B, M, cutoff, cutoff_leftoverMode, params, arr0, arr1, staggered_range, arr0_dA, arr1_dA, arr0_dB,
                   arr1_dB, strides0, strides1):
    pivot = calc_diag_pivot(params)
    K_l = np.sqrt(pivot)  # automatic conversion to float
    K_i = np.sqrt(pivot + 1)  # automatic conversion to float

    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)
    G_in_dA = np.zeros(G_in.shape + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros(G_in.shape + B.shape, dtype=np.complex128)

    read_GB = add_tuple_tail_Array0(params, M, strides0)
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr0[m, n, read_GB] * B

    ########## READ ##########
    # Array1
    for i in range(2 * M):
        if params[i // 2] > 0:
            read = strides1[1] * staggered_range[i] + strides1[2] * (params[i // 2] - 1) + add_tuple_tail_Array2(i // 2,
                                                                                                                 params,
                                                                                                                 M,
                                                                                                                 strides1)
            for m in range(cutoff_leftoverMode):
                for n in range(cutoff_leftoverMode):
                    G_in[m, n, i] = arr1[m, n, read]
                    G_in_dA[m, n, i] = arr1_dA[m, n, read]
                    G_in_dB[m, n, i] = arr1_dB[m, n, read]

    ########## WRITE ##########
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n] = np.multiply(K_l, G_in[m, n])

    # Array1
    for i in range(2 * M):
        if params[i // 2] + 1 < cutoff:
            write = strides1[1] * i + strides1[2] * params[i // 2] + add_tuple_tail_Array2(i // 2, params, M, strides1)
            arr1, arr1_dA, arr1_dB = write_block(i + 2, arr1, write, arr0, read_GB, G_in, GB, A, B, K_i, K_l, cutoff,
                                                 cutoff_leftoverMode, arr1_dA, arr0_dA, G_in_dA, arr1_dB, arr0_dB,
                                                 G_in_dB)

    return arr0, arr1


@njit
def fock_representation_compact_NUMBA(A, B, G0, M, cutoff, cutoff_leftoverMode, PARTITIONS, shape0, shape2, shape11,
                                      shape1, shape0_tuple):
    '''
    Returns the Fock representation of a state or Choi state where all modes are detected accept for the first one
    Args:
        A, B, G0 (Matrix, Vector, Scalar): ABC that are used to apply the recurrence relation
        M (int): number of modes
        cutoff (int): upper bound for the number of photons in each mode
        PARTITIONS (dict): a reference to the "global" PARTITIONS dictionary that is used to iterate over all pivots
        arr0 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the type ab,cc,dd,...
        arr2 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the types ab,(c+2)c,dd,... / ab,cc,(d+2)d,... / ...
        arr11 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the types ab,(c+1)c,(d+1)d,... / ab,(c+1)c,d(d+1),... / ab,c(c+1),(d+1)d,...
        arr1 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the types ab,(c+1)c,dd,... / ab,c(c+1),dd,... / ab,cc,(d+1)d,... / ...
        zero_tuple (tuple): tuple of length M+3 containing integer zeros
    Returns:
        Tensor: the fock representation
    '''
    arr0 = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, np.prod(shape0)), dtype=np.complex128)
    arr2 = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, np.prod(shape2)), dtype=np.complex128)
    arr11 = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, np.prod(shape11)), dtype=np.complex128)
    arr1 = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, np.prod(shape1)), dtype=np.complex128)

    arr0_dA = np.zeros(arr0.shape + A.shape, dtype=np.complex128)
    arr2_dA = np.zeros(arr2.shape + A.shape, dtype=np.complex128)
    arr11_dA = np.zeros(arr11.shape + A.shape, dtype=np.complex128)
    arr1_dA = np.zeros(arr1.shape + A.shape, dtype=np.complex128)
    arr0_dB = np.zeros(arr0.shape + B.shape, dtype=np.complex128)
    arr2_dB = np.zeros(arr2.shape + B.shape, dtype=np.complex128)
    arr11_dB = np.zeros(arr11.shape + B.shape, dtype=np.complex128)
    arr1_dB = np.zeros(arr1.shape + B.shape, dtype=np.complex128)

    # strides excluding leftover mode
    strides0 = strides_from_shape(shape0)
    strides2 = strides_from_shape(shape2)
    strides11 = strides_from_shape(shape11)
    strides1 = strides_from_shape(shape1)

    arr0[0, 0, 0] = G0

    # fill first mode for all PNR detections equal to zero
    for m in range(cutoff_leftoverMode - 1):
        arr0[m + 1, 0, 0] = (arr0[m, 0, 0] * B[0] + np.sqrt(m) * A[0, 0] * arr0[m - 1, 0, 0]) / np.sqrt(m + 1)
        arr0_dA[m + 1, 0, 0] = (arr0_dA[m, 0, 0] * B[0] + np.sqrt(m) * A[0, 0] * arr0_dA[m - 1, 0, 0]) / np.sqrt(m + 1)
        arr0_dA[m + 1, 0, 0][0, 0] += (np.sqrt(m) * arr0[m - 1, 0, 0]) / np.sqrt(m + 1)
        arr0_dB[m + 1, 0, 0] = (arr0_dB[m, 0, 0] * B[0] + np.sqrt(m) * A[0, 0] * arr0_dB[m - 1, 0, 0]) / np.sqrt(m + 1)
        arr0_dB[m + 1, 0, 0][0] += arr0[m, 0, 0] / np.sqrt(m + 1)

    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode - 1):
            arr0[m, n + 1, 0] = (arr0[m, n, 0] * B[1] + np.sqrt(m) * A[1, 0] * arr0[m - 1, n, 0] + np.sqrt(n) * A[
                1, 1] * arr0[m, n - 1, 0]) / np.sqrt(n + 1)
            arr0_dA[m, n + 1, 0] = (arr0_dA[m, n, 0] * B[1] + np.sqrt(m) * A[1, 0] * arr0_dA[m - 1, n, 0] + np.sqrt(n) *
                                    A[1, 1] * arr0_dA[m, n - 1, 0]) / np.sqrt(n + 1)
            arr0_dA[m, n + 1, 0][1, 0] += (np.sqrt(m) * arr0[m - 1, n, 0]) / np.sqrt(n + 1)
            arr0_dA[m, n + 1, 0][1, 1] += (np.sqrt(n) * arr0[m, n - 1, 0]) / np.sqrt(n + 1)
            arr0_dB[m, n + 1, 0] = (arr0_dB[m, n, 0] * B[1] + np.sqrt(m) * A[1, 0] * arr0_dB[m - 1, n, 0] + np.sqrt(n) *
                                    A[1, 1] * arr0_dB[m, n - 1, 0]) / np.sqrt(n + 1)
            arr0_dB[m, n + 1, 0][1] += arr0[m, n, 0] / np.sqrt(n + 1)

    # act as if leftover mode is one element in the nested representation and perform algorithm for diagonal case on M-1 modes
    staggered_range = calc_staggered_range_2M(M - 1)
    for count in range(
            (cutoff - 1) * (M - 1)):  # count = (sum_weight(pivot)-1)/2 # Note: sum_weight(pivot) = 2*(a+b+c+...)+1
        for params in get_partitions((M - 1), count, PARTITIONS):
            if np.max(params) < cutoff:
                # diagonal pivots: aa,bb,cc,dd,...
                arr0, arr1 = use_diag_pivot(A, B, M - 1, cutoff, cutoff_leftoverMode, params, arr0, arr1,
                                            staggered_range, arr0_dA, arr1_dA, arr0_dB, arr1_dB, strides0, strides1)

                # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: aa,(b+1)b,cc,dd | ...
                for d in range((M - 1)):  # for over pivot off-diagonals
                    if params[d] < cutoff - 1:
                        arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB = use_offDiag_pivot(
                            A, B, M - 1, cutoff, cutoff_leftoverMode, params, d, arr0, arr2, arr11, arr1, arr0_dA,
                            arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB, strides0, strides2,
                            strides11, strides1)

    shape_leftoverMode = (cutoff_leftoverMode, cutoff_leftoverMode)
    return arr0.reshape(shape_leftoverMode + shape0_tuple)[:, :, 0, 0], arr0_dA.reshape(
        shape_leftoverMode + shape0_tuple + A.shape)[:, :, 0, 0], arr0_dB.reshape(
        shape_leftoverMode + shape0_tuple + B.shape)[:, :, 0, 0]


def fock_representation_compact_1leftoverMode(A, B, G0, M, cutoff, cutoff_leftoverMode):
    '''
    First initialise the submatrices of G (of which the shape depends on cutoff and M)
    and initialise a zero tuple of length M+3.
    (These initialisations currently cannot be done using Numba.)
    Then calculate the fock representation.
    '''
    # shapes without [cutoff_leftoverMode]*2
    shape0 = np.array([1, 1] + [cutoff] * (M - 1), dtype=np.int64)
    shape2 = np.array([1, (M - 1)] + [cutoff - 2] + [cutoff] * (M - 2), dtype=np.int64)
    if M == 2:
        shape11 = np.array([1, 1, 1],
                           dtype=np.int64)  # For M=1 we will never read from/write to arr11, but Numba requires it to have correct dimensions (corresponding to the tuples that are used for multidim indexing (which have length M+2))
    else:
        shape11 = np.array([3] + [(M - 1) * (M - 2) // 2] + [cutoff - 1] * 2 + [cutoff] * (M - 3), dtype=np.int64)
    shape1 = np.array([1, 2 * (M - 1)] + [cutoff - 1] + [cutoff] * (M - 2), dtype=np.int64)
    shape0_tuple = tuple(shape0)
    return fock_representation_compact_NUMBA(A, B, G0, M, cutoff, cutoff_leftoverMode, PARTITIONS, shape0, shape2,
                                             shape11, shape1, shape0_tuple)


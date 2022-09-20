import numpy as np
from numba import njit, typeof, int64
from numba.typed import Dict
from scipy.special import binom
from numba.cpython.unsafe.tuple import tuple_setitem

# Partitions
PARTITIONS = Dict.empty(key_type=typeof((0, 0)), value_type=int64[:, :])

BINOM = np.zeros((60, 60), dtype=np.int64)
for m in range(BINOM.shape[0]):
    for n in range(BINOM.shape[1]):
        BINOM[m, n] = binom(m, n)


@njit
def len_lvl(M, N):
    r"""Returns the size of an M-mode level with N total photons.
    Args:
        M (int) number of modes
        N (int) number of photons in level
    Returns:
        (int) the the size of an M-mode level with N total photons
    """
    return BINOM[M - 1 + N, N]


@njit
def get_partitions(M, N, PARTITIONS):
    r"""Returns an array of partitions (spreading N photons over M modes)
    If the partitions are already computed it returns them from the PARTITIONS dictionary,
    otherwise it fills the PARTITIONS[(M,N)] dictionary entry.
    Args:
        M (int) number of modes
        N (int) number of photons in level
        PARTITIONS (dict) a reference to the "global" PARTITIONS dictionary
    Returns:
        (2d array) the array of pivots for the given level
    """
    if (M, N) in PARTITIONS:
        return PARTITIONS[(M, N)]
    if M == 1:
        return np.array([[N]])
    else:
        T = 0
        pivots = np.zeros((len_lvl(M, N), M), dtype=np.int64)
        for n in range(N + 1):
            pivots[T: T + len_lvl(M - 1, N - n), :1] = n
            pivots[T: T + len_lvl(M - 1, N - n), 1:] = get_partitions(M - 1, N - n, PARTITIONS)
            T += len_lvl(M - 1, N - n)
        PARTITIONS[(M, N)] = pivots
        return pivots


# Simulation
# helper functions to construct tuples that are used for multidimensional indexing of the submatrices of G
@njit
def fill_tuple_tail_Array0(tup, params, M):
    '''
    This function is equivalent to:
        tup = list(tup)
        tup[2:] = params
        return tuple(tup)
    while being compatible with Numba.
    '''
    for t in range(2, M + 2):
        tup = tuple_setitem(tup, t + 2, params[t - 2])
    return tup


@njit
def fill_tuple_tail_Array2(tup, idx0, params, M):
    '''
    This function is equivalent to:
        tup = list(tup)
        tup[3:] = [x for x in range(M) if x!=idx0]
        return tuple(tup)
    while being compatible with Numba.
    '''
    for x in range(M):
        if x < idx0:
            tup = tuple_setitem(tup, x + 5, params[x])
        elif x > idx0:
            tup = tuple_setitem(tup, x + 4, params[x])
    return tup


@njit
def fill_tuple_tail_Array11(tup, idx0, idx1, params, M):
    '''
    This function is equivalent to:
        tup = list(tup)
        tup[4:] = [x for x in range(M) if (x!=idx0 and x!=idx1)]
        return tuple(tup)
    while being compatible with Numba.

    Assumption: idx0<idx1 (i.e. here: [idx0,idx1] == sorted([i,d]))
    '''
    for x in range(M):
        if x < idx0:
            tup = tuple_setitem(tup, x + 6, params[x])
        elif idx0 < x and x < idx1:
            tup = tuple_setitem(tup, x + 5, params[x])
        elif idx1 < x:
            tup = tuple_setitem(tup, x + 4, params[x])
    return tup


# Other helper functions
@njit
def calc_diag_pivot(params):
    '''
    return pivot in original representation of G
    i.e. a,a,b,b,c,c,...
    params [1D array]: [a,b,c,...]
    '''
    pivot = np.zeros(2 * params.shape[0], dtype=np.int64)
    for i, val in enumerate(params):
        pivot[2 * i] = val
        pivot[2 * i + 1] = val
    return pivot


@njit
def calc_offDiag_pivot(params, d):
    '''
    return pivot in original representation of G
    i.e. d=0: a+1,a,b,b,c,c,... | d=1: a,a,b+1,b,c,c,...
    params [1D array]: [a,b,c,...]
    d [int]: index of pivot-offDiagonal
    '''
    pivot = np.zeros(2 * params.shape[0], dtype=np.int64)
    for i, val in enumerate(params):
        pivot[2 * i] = val
        pivot[2 * i + 1] = val
    pivot[2 * d] += 1
    return pivot


@njit
def index_above_diagonal(idx0, idx1, M):  # should memoize these functions
    '''
    Given the indices of an element that is located above the diagonal in an array of shape MxM,
    return a single index that identifies such an element in the following way:
    (Example for M=3)
    [[x,0,1]
     [x,x,2]
     [x,x,x]]
    idx0,idx1=0,1 --> return 0
    idx0,idx1=0,2 --> return 1
    idx0,idx1=1,2 --> return 2
    (Assumption: idx0<idx1)
    '''
    ids = np.cumsum(np.hstack((np.zeros(1, dtype=np.int64), np.arange(2, M, dtype=np.int64)[
                                                            ::-1])))  # desired n for values next to diagonal (e.g. for M=3: ids=[0,2])
    return ids[idx0] + idx1 - idx0 - 1


@njit
def calc_staggered_range_2M(M):
    '''
    Output: np.array([1,0,3,2,5,4,...,2*M-1,2*M-2])
    This array is used to index the fock amplitudes that are read when using a diagonal pivot (i.e. a pivot of type aa,bb,cc,dd,...).
    '''
    A = np.zeros(2 * M, dtype=np.int64)
    for i in range(1, 2 * M, 2):
        A[i - 1] = i
    for i in range(0, 2 * M, 2):
        A[i + 1] = i
    return A


@njit
def mn_tup(m, n, tup):
    tup = tuple_setitem(tup, 0, m)
    return tuple_setitem(tup, 1, n)


@njit
def calc_dA_dB(m, n, i, arr_read_pivot, read_GB, G_in_adapted, A_adapted, B, K_i, K_l_adapted, cutoff,
               cutoff_leftoverMode, arr_read_pivot_dA, G_in_dA_adapted, arr_read_pivot_dB, G_in_dB_adapted, l_range):
    dA = arr_read_pivot_dA[mn_tup(m, n, read_GB)] * B[i]
    dB = arr_read_pivot_dB[mn_tup(m, n, read_GB)] * B[i]
    dB[i] += arr_read_pivot[mn_tup(m, n, read_GB)]
    for l_prime, l in enumerate(l_range):
        dA += K_l_adapted[l_prime] * A_adapted[l_prime] * G_in_dA_adapted[l_prime]
        dB += K_l_adapted[l_prime] * A_adapted[l_prime] * G_in_dB_adapted[l_prime]
        dA[i, l] += G_in_adapted[l_prime]
    return dA / K_i[i - 2], dB / K_i[i - 2]


@njit
def write_block(i, arr_write, write, arr_read_pivot, read_GB, G_in, GB, A, B, K_i, K_l, cutoff, cutoff_leftoverMode,
                arr_write_dA, arr_read_pivot_dA, G_in_dA, arr_write_dB, arr_read_pivot_dB, G_in_dB):
    # m,n = 0,0
    m, n = 0, 0
    l_range = np.arange(2, A.shape[1])
    A_adapted = A[i, 2:]
    G_in_adapted = G_in[0, 0]
    G_in_dA_adapted = G_in_dA[0, 0]
    G_in_dB_adapted = G_in_dB[0, 0]
    K_l_adapted = K_l
    arr_write[mn_tup(0, 0, write)] = (GB[0, 0, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
    arr_write_dA[mn_tup(0, 0, write)], arr_write_dB[mn_tup(0, 0, write)] = calc_dA_dB(m, n, i, arr_read_pivot, read_GB,
                                                                                      G_in_adapted, A_adapted, B, K_i,
                                                                                      K_l_adapted, cutoff,
                                                                                      cutoff_leftoverMode,
                                                                                      arr_read_pivot_dA,
                                                                                      G_in_dA_adapted,
                                                                                      arr_read_pivot_dB,
                                                                                      G_in_dB_adapted, l_range)

    # m=0
    m = 0
    l_range = np.arange(1, A.shape[1])
    A_adapted = A[i, 1:]
    for n in range(1, cutoff_leftoverMode):
        K_l_adapted = np.hstack((np.array([np.sqrt(n)]), K_l))
        G_in_adapted = np.hstack((np.array([arr_read_pivot[mn_tup(0, n - 1, read_GB)] * np.sqrt(n)]), G_in[0, n]))
        G_in_dA_adapted = np.concatenate(
            (np.expand_dims(arr_read_pivot_dA[mn_tup(0, n - 1, read_GB)], axis=0), G_in_dA[0, n]), axis=0)
        G_in_dB_adapted = np.concatenate(
            (np.expand_dims(arr_read_pivot_dB[mn_tup(0, n - 1, read_GB)], axis=0), G_in_dB[0, n]), axis=0)
        arr_write[mn_tup(0, n, write)] = (GB[0, n, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
        arr_write_dA[mn_tup(0, n, write)], arr_write_dB[mn_tup(0, n, write)] = calc_dA_dB(m, n, i, arr_read_pivot,
                                                                                          read_GB, G_in_adapted,
                                                                                          A_adapted, B, K_i,
                                                                                          K_l_adapted, cutoff,
                                                                                          cutoff_leftoverMode,
                                                                                          arr_read_pivot_dA,
                                                                                          G_in_dA_adapted,
                                                                                          arr_read_pivot_dB,
                                                                                          G_in_dB_adapted, l_range)

    # n=0
    n = 0
    l_range = np.arange(1, A.shape[1])
    l_range[0] = 0
    A_adapted = np.hstack((np.array([A[i, 0]]), A[i, 2:]))
    for m in range(1, cutoff_leftoverMode):
        K_l_adapted = np.hstack((np.array([np.sqrt(m)]), K_l))
        G_in_adapted = np.hstack((np.array([arr_read_pivot[mn_tup(m - 1, 0, read_GB)] * np.sqrt(m)]), G_in[m, 0]))
        G_in_dA_adapted = np.concatenate(
            (np.expand_dims(arr_read_pivot_dA[mn_tup(m - 1, 0, read_GB)], axis=0), G_in_dA[m, 0]), axis=0)
        G_in_dB_adapted = np.concatenate(
            (np.expand_dims(arr_read_pivot_dB[mn_tup(m - 1, 0, read_GB)], axis=0), G_in_dB[m, 0]), axis=0)
        arr_write[mn_tup(m, 0, write)] = (GB[m, 0, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
        arr_write_dA[mn_tup(m, 0, write)], arr_write_dB[mn_tup(m, 0, write)] = calc_dA_dB(m, n, i, arr_read_pivot,
                                                                                          read_GB, G_in_adapted,
                                                                                          A_adapted, B, K_i,
                                                                                          K_l_adapted, cutoff,
                                                                                          cutoff_leftoverMode,
                                                                                          arr_read_pivot_dA,
                                                                                          G_in_dA_adapted,
                                                                                          arr_read_pivot_dB,
                                                                                          G_in_dB_adapted, l_range)

    # m>0,n>0
    l_range = np.arange(A.shape[1])
    A_adapted = A[i]
    for m in range(1, cutoff_leftoverMode):
        for n in range(1, cutoff_leftoverMode):
            K_l_adapted = np.hstack((np.array([np.sqrt(m), np.sqrt(n)]), K_l))
            G_in_adapted = np.hstack((np.array([arr_read_pivot[mn_tup(m - 1, n, read_GB)] * np.sqrt(m),
                                                arr_read_pivot[mn_tup(m, n - 1, read_GB)] * np.sqrt(n)]), G_in[m, n]))
            G_in_dA_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dA[mn_tup(m - 1, n, read_GB)], axis=0),
                                              np.expand_dims(arr_read_pivot_dA[mn_tup(m, n - 1, read_GB)], axis=0),
                                              G_in_dA[m, n]), axis=0)
            G_in_dB_adapted = np.concatenate((np.expand_dims(arr_read_pivot_dB[mn_tup(m - 1, n, read_GB)], axis=0),
                                              np.expand_dims(arr_read_pivot_dB[mn_tup(m, n - 1, read_GB)], axis=0),
                                              G_in_dB[m, n]), axis=0)
            arr_write[mn_tup(m, n, write)] = (GB[m, n, i] + A_adapted @ G_in_adapted) / K_i[i - 2]
            arr_write_dA[mn_tup(m, n, write)], arr_write_dB[mn_tup(m, n, write)] = calc_dA_dB(m, n, i, arr_read_pivot,
                                                                                              read_GB, G_in_adapted,
                                                                                              A_adapted, B, K_i,
                                                                                              K_l_adapted, cutoff,
                                                                                              cutoff_leftoverMode,
                                                                                              arr_read_pivot_dA,
                                                                                              G_in_dA_adapted,
                                                                                              arr_read_pivot_dB,
                                                                                              G_in_dB_adapted, l_range)

    return arr_write, arr_write_dA, arr_write_dB


@njit
def use_offDiag_pivot(A, B, M, cutoff, cutoff_leftoverMode, params, d, arr0, arr2, arr11, arr1, zero_tuple, arr0_dA,
                      arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB):
    pivot = calc_offDiag_pivot(params, d)
    K_l = np.sqrt(pivot)  # automatic conversion to float
    K_i = np.sqrt(pivot + 1)  # automatic conversion to float
    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)  # M is actually M-1 here
    G_in_dA = np.zeros(G_in.shape + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros(G_in.shape + B.shape, dtype=np.complex128)

    read_GB = tuple_setitem(zero_tuple, 3, 2 * d)
    read_GB = tuple_setitem(read_GB, 4, params[d])
    read_GB = fill_tuple_tail_Array2(read_GB, d, params, M)
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr1[mn_tup(m, n, read_GB)] * B

    ########## READ ##########

    # Array0
    read0 = fill_tuple_tail_Array0(zero_tuple, params,
                                   M)  # can store this one as I do not need to check boundary conditions for Array0! (Doesn't work for other arrays)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n, 2 * d] = arr0[mn_tup(m, n, read0)]
            G_in_dA[m, n, 2 * d] = arr0_dA[mn_tup(m, n, read0)]
            G_in_dB[m, n, 2 * d] = arr0_dB[mn_tup(m, n, read0)]

    # read from Array2
    if params[d] > 0:  # params[d]-1>=0
        read = zero_tuple
        read = tuple_setitem(read, 3, d)
        read = tuple_setitem(read, 4, params[d] - 1)
        read = fill_tuple_tail_Array2(read, d, params, M)
        for m in range(cutoff_leftoverMode):
            for n in range(cutoff_leftoverMode):
                G_in[m, n, 2 * d + 1] = arr2[mn_tup(m, n, read)]
                G_in_dA[m, n, 2 * d + 1] = arr2_dA[mn_tup(m, n, read)]
                G_in_dB[m, n, 2 * d + 1] = arr2_dB[mn_tup(m, n, read)]

    # read from Array11
    for i in range(d + 1, M):  # i>d
        if params[i] > 0:
            read = zero_tuple
            read = tuple_setitem(read, 3, index_above_diagonal(d, i, M))
            read = tuple_setitem(read, 4, params[d])
            read = tuple_setitem(read, 5, params[i] - 1)
            read = fill_tuple_tail_Array11(read, d, i, params, M)
            for m in range(cutoff_leftoverMode):
                for n in range(cutoff_leftoverMode):
                    G_in[m, n, 2 * i] = arr11[mn_tup(m, n, tuple_setitem(read, 2, 1))]  # READ green (1001)
                    G_in_dA[m, n, 2 * i] = arr11_dA[mn_tup(m, n, tuple_setitem(read, 2, 1))]
                    G_in_dB[m, n, 2 * i] = arr11_dB[mn_tup(m, n, tuple_setitem(read, 2, 1))]
                    G_in[m, n, 2 * i + 1] = arr11[mn_tup(m, n, read)]  # READ red (1010)
                    G_in_dA[m, n, 2 * i + 1] = arr11_dA[mn_tup(m, n, read)]
                    G_in_dB[m, n, 2 * i + 1] = arr11_dB[mn_tup(m, n, read)]

    for i in range(d):  # i<d
        if params[i] > 0:
            read = zero_tuple
            read = tuple_setitem(read, 3, index_above_diagonal(i, d, M))
            read = tuple_setitem(read, 4, params[i] - 1)
            read = tuple_setitem(read, 5, params[d])
            read = fill_tuple_tail_Array11(read, i, d, params, M)
            for m in range(cutoff_leftoverMode):
                for n in range(cutoff_leftoverMode):
                    G_in[m, n, 2 * i] = arr11[mn_tup(m, n, tuple_setitem(read, 2, 2))]  # READ blue (0110)
                    G_in_dA[m, n, 2 * i] = arr11_dA[mn_tup(m, n, tuple_setitem(read, 2, 2))]
                    G_in_dB[m, n, 2 * i] = arr11_dB[mn_tup(m, n, tuple_setitem(read, 2, 2))]
                    G_in[m, n, 2 * i + 1] = arr11[mn_tup(m, n, read)]  # READ red (1010)
                    G_in_dA[m, n, 2 * i + 1] = arr11_dA[mn_tup(m, n, read)]
                    G_in_dB[m, n, 2 * i + 1] = arr11_dB[mn_tup(m, n, read)]

    ########## WRITE ##########

    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n] = np.multiply(K_l, G_in[m, n])

    # Array0
    if d == 0 or np.all(params[:d] == 0):
        write0 = tuple_setitem(read0, d + 4, params[d] + 1)
        arr0, arr0_dA, arr0_dB = write_block(2 * d + 3, arr0, write0, arr1, read_GB, G_in, GB, A, B, K_i, K_l, cutoff,
                                             cutoff_leftoverMode, arr0_dA, arr1_dA, G_in_dA, arr0_dB, arr1_dB, G_in_dB)

    # Array2
    if params[d] + 2 < cutoff:
        write = zero_tuple
        write = tuple_setitem(write, 3, d)
        write = tuple_setitem(write, 4, params[d])
        write = fill_tuple_tail_Array2(write, d, params, M)
        arr2, arr2_dA, arr2_dB = write_block(2 * d + 2, arr2, write, arr1, read_GB, G_in, GB, A, B, K_i, K_l, cutoff,
                                             cutoff_leftoverMode, arr2_dA, arr1_dA, G_in_dA, arr2_dB, arr1_dB, G_in_dB)

    # Array11
    for i in range(d + 1, M):
        if params[i] + 1 < cutoff:
            write = zero_tuple
            write = tuple_setitem(write, 3, index_above_diagonal(d, i, M))
            write = tuple_setitem(write, 4, params[d])
            write = tuple_setitem(write, 5, params[i])
            write = fill_tuple_tail_Array11(write, d, i, params, M)
            arr11, arr11_dA, arr11_dB = write_block(2 * i + 2, arr11, write, arr1, read_GB, G_in, GB, A, B, K_i, K_l,
                                                    cutoff, cutoff_leftoverMode, arr11_dA, arr1_dA, G_in_dA, arr11_dB,
                                                    arr1_dB, G_in_dB)  # WRITE red (1010)
            arr11, arr11_dA, arr11_dB = write_block(2 * i + 3, arr11, tuple_setitem(write, 2, 1), arr1, read_GB, G_in,
                                                    GB, A, B, K_i, K_l, cutoff, cutoff_leftoverMode, arr11_dA, arr1_dA,
                                                    G_in_dA, arr11_dB, arr1_dB, G_in_dB)  # WRITE green (1001)

    for i in range(d):
        if params[i] + 1 < cutoff:
            write = zero_tuple
            write = tuple_setitem(write, 3, index_above_diagonal(i, d, M))
            write = tuple_setitem(write, 4, params[i])
            write = tuple_setitem(write, 5, params[d])
            write = fill_tuple_tail_Array11(write, i, d, params, M)
            arr11, arr11_dA, arr11_dB = write_block(2 * i + 3, arr11, tuple_setitem(write, 2, 2), arr1, read_GB, G_in,
                                                    GB, A, B, K_i, K_l, cutoff, cutoff_leftoverMode, arr11_dA, arr1_dA,
                                                    G_in_dA, arr11_dB, arr1_dB, G_in_dB)  # WRITE blue (0110)

    return arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB


@njit
def use_diag_pivot(A, B, M, cutoff, cutoff_leftoverMode, params, arr0, arr1, zero_tuple, staggered_range, arr0_dA,
                   arr1_dA, arr0_dB, arr1_dB):
    pivot = calc_diag_pivot(params)
    K_l = np.sqrt(pivot)  # automatic conversion to float
    K_i = np.sqrt(pivot + 1)  # automatic conversion to float

    G_in = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, 2 * M), dtype=np.complex128)
    G_in_dA = np.zeros(G_in.shape + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros(G_in.shape + B.shape, dtype=np.complex128)

    read_GB = fill_tuple_tail_Array0(zero_tuple, params, M)
    GB = np.zeros((cutoff_leftoverMode, cutoff_leftoverMode, len(B)), dtype=np.complex128)
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            GB[m, n] = arr0[mn_tup(m, n, read_GB)] * B

    ########## READ ##########
    # Array1
    for i in range(2 * M):
        if params[i // 2] > 0:
            read = zero_tuple
            read = tuple_setitem(read, 3, staggered_range[i])
            read = tuple_setitem(read, 4, params[i // 2] - 1)
            read = fill_tuple_tail_Array2(read, i // 2, params, M)
            for m in range(cutoff_leftoverMode):
                for n in range(cutoff_leftoverMode):
                    G_in[m, n, i] = arr1[mn_tup(m, n, read)]
                    G_in_dA[m, n, i] = arr1_dA[mn_tup(m, n, read)]
                    G_in_dB[m, n, i] = arr1_dB[mn_tup(m, n, read)]

    ########## WRITE ##########
    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode):
            G_in[m, n] = np.multiply(K_l, G_in[m, n])

    # Array1
    for i in range(2 * M):
        if params[i // 2] + 1 < cutoff:
            write = zero_tuple
            write = tuple_setitem(write, 3, i)
            write = tuple_setitem(write, 4, params[i // 2])
            write = fill_tuple_tail_Array2(write, i // 2, params, M)
            arr1, arr1_dA, arr1_dB = write_block(i + 2, arr1, write, arr0, read_GB, G_in, GB, A, B, K_i, K_l, cutoff,
                                                 cutoff_leftoverMode, arr1_dA, arr0_dA, G_in_dA, arr1_dB, arr0_dB,
                                                 G_in_dB)

    return arr0, arr1


@njit
def fock_representation_compact_NUMBA(A, B, G0, M, cutoff, cutoff_leftoverMode, PARTITIONS, arr0, arr2, arr11, arr1,
                                      zero_tuple):
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
    arr0_dA = np.zeros(arr0.shape + A.shape, dtype=np.complex128)
    arr2_dA = np.zeros(arr2.shape + A.shape, dtype=np.complex128)
    arr11_dA = np.zeros(arr11.shape + A.shape, dtype=np.complex128)
    arr1_dA = np.zeros(arr1.shape + A.shape, dtype=np.complex128)
    arr0_dB = np.zeros(arr0.shape + B.shape, dtype=np.complex128)
    arr2_dB = np.zeros(arr2.shape + B.shape, dtype=np.complex128)
    arr11_dB = np.zeros(arr11.shape + B.shape, dtype=np.complex128)
    arr1_dB = np.zeros(arr1.shape + B.shape, dtype=np.complex128)

    arr0[zero_tuple] = G0

    # fill first mode for all PNR detections equal to zero
    for m in range(cutoff_leftoverMode - 1):
        arr0[tuple_setitem(zero_tuple, 0, m + 1)] = (arr0[tuple_setitem(zero_tuple, 0, m)] * B[0] + np.sqrt(m) * A[
            0, 0] * arr0[tuple_setitem(zero_tuple, 0, m - 1)]) / np.sqrt(m + 1)
        arr0_dA[tuple_setitem(zero_tuple, 0, m + 1)] = (arr0_dA[tuple_setitem(zero_tuple, 0, m)] * B[0] + np.sqrt(m) *
                                                        A[0, 0] * arr0_dA[
                                                            tuple_setitem(zero_tuple, 0, m - 1)]) / np.sqrt(m + 1)
        arr0_dA[tuple_setitem(zero_tuple, 0, m + 1)][0, 0] += (np.sqrt(m) * arr0[
            tuple_setitem(zero_tuple, 0, m - 1)]) / np.sqrt(m + 1)
        arr0_dB[tuple_setitem(zero_tuple, 0, m + 1)] = (arr0_dB[tuple_setitem(zero_tuple, 0, m)] * B[0] + np.sqrt(m) *
                                                        A[0, 0] * arr0_dB[
                                                            tuple_setitem(zero_tuple, 0, m - 1)]) / np.sqrt(m + 1)
        arr0_dB[tuple_setitem(zero_tuple, 0, m + 1)][0] += arr0[tuple_setitem(zero_tuple, 0, m)] / np.sqrt(m + 1)

    for m in range(cutoff_leftoverMode):
        for n in range(cutoff_leftoverMode - 1):
            arr0[mn_tup(m, n + 1, zero_tuple)] = (arr0[mn_tup(m, n, zero_tuple)] * B[1] + np.sqrt(m) * A[1, 0] * arr0[
                mn_tup(m - 1, n, zero_tuple)] + np.sqrt(n) * A[1, 1] * arr0[mn_tup(m, n - 1, zero_tuple)]) / np.sqrt(
                n + 1)
            arr0_dA[mn_tup(m, n + 1, zero_tuple)] = (arr0_dA[mn_tup(m, n, zero_tuple)] * B[1] + np.sqrt(m) * A[1, 0] *
                                                     arr0_dA[mn_tup(m - 1, n, zero_tuple)] + np.sqrt(n) * A[1, 1] *
                                                     arr0_dA[mn_tup(m, n - 1, zero_tuple)]) / np.sqrt(n + 1)
            arr0_dA[mn_tup(m, n + 1, zero_tuple)][1, 0] += (np.sqrt(m) * arr0[mn_tup(m - 1, n, zero_tuple)]) / np.sqrt(
                n + 1)
            arr0_dA[mn_tup(m, n + 1, zero_tuple)][1, 1] += (np.sqrt(n) * arr0[mn_tup(m, n - 1, zero_tuple)]) / np.sqrt(
                n + 1)
            arr0_dB[mn_tup(m, n + 1, zero_tuple)] = (arr0_dB[mn_tup(m, n, zero_tuple)] * B[1] + np.sqrt(m) * A[1, 0] *
                                                     arr0_dB[mn_tup(m - 1, n, zero_tuple)] + np.sqrt(n) * A[1, 1] *
                                                     arr0_dB[mn_tup(m, n - 1, zero_tuple)]) / np.sqrt(n + 1)
            arr0_dB[mn_tup(m, n + 1, zero_tuple)][1] += arr0[mn_tup(m, n, zero_tuple)] / np.sqrt(n + 1)

    # act as if leftover mode is one element in the nested representation and perform algorithm for diagonal case on M-1 modes
    staggered_range = calc_staggered_range_2M(M - 1)
    for count in range(
            (cutoff - 1) * (M - 1)):  # count = (sum_weight(pivot)-1)/2 # Note: sum_weight(pivot) = 2*(a+b+c+...)+1
        for params in get_partitions((M - 1), count, PARTITIONS):
            if np.max(params) < cutoff:
                # diagonal pivots: aa,bb,cc,dd,...
                arr0, arr1 = use_diag_pivot(A, B, M - 1, cutoff, cutoff_leftoverMode, params, arr0, arr1, zero_tuple,
                                            staggered_range, arr0_dA, arr1_dA, arr0_dB, arr1_dB)

                # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: aa,(b+1)b,cc,dd | ...
                for d in range((M - 1)):  # for over pivot off-diagonals
                    if params[d] < cutoff - 1:
                        arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB = use_offDiag_pivot(
                            A, B, M - 1, cutoff, cutoff_leftoverMode, params, d, arr0, arr2, arr11, arr1, zero_tuple,
                            arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB)

    return arr0[:, :, 0, 0], arr0_dA[:, :, 0, 0], arr0_dB[:, :, 0, 0]


def fock_representation_compact(A, B, G0, M, cutoff, cutoff_leftoverMode):
    '''
    First initialise the submatrices of G (of which the shape depends on cutoff and M)
    and initialise a zero tuple of length M+3.
    (These initialisations currently cannot be done using Numba.)
    Then calculate the fock representation.
    '''
    arr0 = np.zeros([cutoff_leftoverMode] * 2 + [1, 1] + [cutoff] * (M - 1), dtype=np.complex128)
    arr2 = np.zeros([cutoff_leftoverMode] * 2 + [1, (M - 1)] + [cutoff - 2] + [cutoff] * (M - 2), dtype=np.complex128)
    if M == 2:
        arr11 = np.zeros([1, 1, 1, 1, 1],
                         dtype=np.complex128)  # For M=1 we will never read from/write to arr11, but Numba requires it to have correct dimensions (corresponding to the tuples that are used for multidim indexing (which have length M+2))
    else:
        arr11 = np.zeros(
            [cutoff_leftoverMode] * 2 + [3] + [(M - 1) * (M - 2) // 2] + [cutoff - 1] * 2 + [cutoff] * (M - 3),
            dtype=np.complex128)
    arr1 = np.zeros([cutoff_leftoverMode] * 2 + [1, 2 * (M - 1)] + [cutoff - 1] + [cutoff] * (M - 2),
                    dtype=np.complex128)
    zero_tuple = tuple([0] * (M + 3))
    return fock_representation_compact_NUMBA(A, B, G0, M, cutoff, cutoff_leftoverMode, PARTITIONS, arr0, arr2, arr11,
                                             arr1, zero_tuple)


def hermite_multidimensional_1leftoverMode(A,B,G0,cutoff, cutoff_leftoverMode):
    cutoff_leftoverMode = cutoff_leftoverMode.item() # tf.numpy_function() wraps this into an array, which leads to numba error if we want to iterate range(cutoff_leftoverMode)
    M = A.shape[0]//2
    assert M>1
    G,G_dA,G_dB = fock_representation_compact(A, B, G0, M, cutoff, cutoff_leftoverMode)
    G_dG0 = np.array(G / G0).astype(np.complex128)
    return G,G_dG0,G_dA,G_dB
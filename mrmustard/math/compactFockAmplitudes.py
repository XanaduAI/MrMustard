import numpy as np
from numba import njit, typeof, int64
from numba.typed import Dict
from scipy.special import binom
from numba.cpython.unsafe.tuple import tuple_setitem

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


# helper functions to construct tuples that are used for multidimensional indexing of the submatrices of G
@njit
def add_tuple_tail_Array0(params, M, strides):
    '''
    This function is equivalent to:
        tup = list(tup)
        tup[2:] = params
        return tuple(tup)
    while being compatible with Numba.
    '''
    strideVal = 0
    for t in range(2, M + 2):
        strideVal += strides[t] * params[t - 2]
    return strideVal


@njit
def add_tuple_tail_Array2(idx0, params, M, strides):
    '''
    This function is equivalent to:
        tup = list(tup)
        tup[3:] = [x for x in range(M) if x!=idx0]
        return tuple(tup)
    while being compatible with Numba.
    '''
    strideVal = 0
    for x in range(M):
        if x < idx0:
            strideVal += strides[x + 3] * params[x]
        elif x > idx0:
            strideVal += strides[x + 2] * params[x]
    return strideVal


@njit
def add_tuple_tail_Array11(idx0, idx1, params, M, strides):
    '''
    This function is equivalent to:
        tup = list(tup)
        tup[4:] = [x for x in range(M) if (x!=idx0 and x!=idx1)]
        return tuple(tup)
    while being compatible with Numba.

    Assumption: idx0<idx1 (i.e. here: [idx0,idx1] == sorted([i,d]))
    '''
    strideVal = 0
    for x in range(M):
        if x < idx0:
            strideVal += strides[x + 4] * params[x]
        elif idx0 < x and x < idx1:
            strideVal += strides[x + 3] * params[x]
        elif idx1 < x:
            strideVal += strides[x + 2] * params[x]
    return strideVal


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
def strides_from_shape(shape):
    for idx in range(len(shape)):
        if shape[idx] == 0:
            shape[idx] = 1
    res = np.ones(len(shape), dtype=np.int64)
    for idx in range(len(shape) - 1):
        res[idx + 1] = res[idx] * shape[::-1][idx]
    return res[::-1]


@njit
def calc_dA_dB(i, G_in_dA, G_in_dB, G_in, A, B, read_GB, K_l, K_i, pivot, M, pivot_val, pivot_val_dA, pivot_val_dB):
    dA = pivot_val_dA * B[i]
    dB = pivot_val_dB * B[i]
    dB[i] += pivot_val
    for l in range(2 * M):
        dA += K_l[l] * A[i, l] * G_in_dA[l]
        dB += K_l[l] * A[i, l] * G_in_dB[l]
        dA[i, l] += G_in[l]
    return dA / K_i[i], dB / K_i[i]


@njit
def use_offDiag_pivot(A, B, M, cutoff, params, d, arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB,
                      arr2_dB, arr11_dB, arr1_dB, strides0, strides2, strides11, strides1):
    pivot = calc_offDiag_pivot(params, d)
    K_l = np.sqrt(pivot)  # automatic conversion to float
    K_i = np.sqrt(pivot + 1)  # automatic conversion to float
    G_in = np.zeros(2 * M, dtype=np.complex128)
    G_in_dA = np.zeros((2 * M,) + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros((2 * M,) + B.shape, dtype=np.complex128)

    read_GB = strides1[1] * 2 * d + strides1[2] * params[d] + add_tuple_tail_Array2(d, params, M, strides1)
    pivot_val = arr1[read_GB]
    pivot_val_dA = arr1_dA[read_GB]
    pivot_val_dB = arr1_dB[read_GB]
    GB = arr1[read_GB] * B

    ########## READ ##########

    # Array0
    read0 = add_tuple_tail_Array0(params, M,
                                  strides0)  # can store this one as I do not need to check boundary conditions for Array0! (Doesn't work for other arrays)
    G_in[2 * d] = arr0[read0]
    G_in_dA[2 * d] = arr0_dA[read0]
    G_in_dB[2 * d] = arr0_dB[read0]

    # read from Array2
    if params[d] > 0:  # params[d]-1>=0
        read = strides2[1] * d + strides2[2] * (params[d] - 1) + add_tuple_tail_Array2(d, params, M, strides2)
        G_in[2 * d + 1] = arr2[read]
        G_in_dA[2 * d + 1] = arr2_dA[read]
        G_in_dB[2 * d + 1] = arr2_dB[read]

    # read from Array11
    for i in range(d + 1, M):  # i>d
        if params[i] > 0:
            read = strides11[1] * index_above_diagonal(d, i, M) + strides11[2] * params[d] + strides11[3] * (
                        params[i] - 1) + add_tuple_tail_Array11(d, i, params, M, strides11)
            G_in[2 * i] = arr11[read + strides11[0]]  # READ green (1001)
            G_in_dA[2 * i] = arr11_dA[read + strides11[0]]
            G_in_dB[2 * i] = arr11_dB[read + strides11[0]]
            G_in[2 * i + 1] = arr11[read]  # READ red (1010)
            G_in_dA[2 * i + 1] = arr11_dA[read]
            G_in_dB[2 * i + 1] = arr11_dB[read]

    for i in range(d):  # i<d
        if params[i] > 0:
            read = strides11[1] * index_above_diagonal(i, d, M) + strides11[2] * (params[i] - 1) + strides11[3] * \
                   params[d] + add_tuple_tail_Array11(i, d, params, M, strides11)

            G_in[2 * i] = arr11[read + strides11[0] * 2]  # READ blue (0110)
            G_in_dA[2 * i] = arr11_dA[read + strides11[0] * 2]
            G_in_dB[2 * i] = arr11_dB[read + strides11[0] * 2]

            G_in[2 * i + 1] = arr11[read]  # READ red (1010)
            G_in_dA[2 * i + 1] = arr11_dA[read]
            G_in_dB[2 * i + 1] = arr11_dB[read]

    ########## WRITE ##########

    G_in = np.multiply(K_l, G_in)

    # Array0
    if d == 0 or np.all(params[:d] == 0):
        write0 = read0 + strides0[2 + d]  # params[d] --> params[d]+1
        arr0[write0] = (GB[2 * d + 1] + A[2 * d + 1] @ G_in) / K_i[2 * d + 1]  # I could absorb K_i in A and GB
        arr0_dA[write0], arr0_dB[write0] = calc_dA_dB(2 * d + 1, G_in_dA, G_in_dB, G_in, A, B, read_GB, K_l, K_i, pivot,
                                                      M, pivot_val, pivot_val_dA, pivot_val_dB)

    # Array2
    if params[d] + 2 < cutoff:
        write = strides2[1] * d + strides2[2] * params[d] + add_tuple_tail_Array2(d, params, M, strides2)
        arr2[write] = (GB[2 * d] + A[2 * d] @ G_in) / K_i[2 * d]
        arr2_dA[write], arr2_dB[write] = calc_dA_dB(2 * d, G_in_dA, G_in_dB, G_in, A, B, read_GB, K_l, K_i, pivot, M,
                                                    pivot_val, pivot_val_dA, pivot_val_dB)

    # Array11
    for i in range(d + 1, M):
        if params[i] + 1 < cutoff:
            write = strides11[1] * index_above_diagonal(d, i, M) + strides11[2] * params[d] + strides11[3] * params[
                i] + add_tuple_tail_Array11(d, i, params, M, strides11)

            arr11[write] = (GB[2 * i] + A[2 * i] @ G_in) / K_i[2 * i]  # WRITE red (1010)
            arr11_dA[write], arr11_dB[write] = calc_dA_dB(2 * i, G_in_dA, G_in_dB, G_in, A, B, read_GB, K_l, K_i, pivot,
                                                          M, pivot_val, pivot_val_dA, pivot_val_dB)

            arr11[write + strides11[0] * 1] = (GB[2 * i + 1] + A[2 * i + 1] @ G_in) / K_i[
                2 * i + 1]  # WRITE green (1001)
            arr11_dA[write + strides11[0] * 1], arr11_dB[write + strides11[0] * 1] = calc_dA_dB(2 * i + 1, G_in_dA,
                                                                                                G_in_dB, G_in, A, B,
                                                                                                read_GB, K_l, K_i,
                                                                                                pivot, M, pivot_val,
                                                                                                pivot_val_dA,
                                                                                                pivot_val_dB)

    for i in range(d):
        if params[i] + 1 < cutoff:
            write = strides11[1] * index_above_diagonal(i, d, M) + strides11[2] * params[i] + strides11[3] * params[
                d] + add_tuple_tail_Array11(i, d, params, M, strides11)
            arr11[write + strides11[0] * 2] = (GB[2 * i + 1] + A[2 * i + 1] @ G_in) / K_i[
                2 * i + 1]  # WRITE blue (0110)
            arr11_dA[write + strides11[0] * 2], arr11_dB[write + strides11[0] * 2] = calc_dA_dB(2 * i + 1, G_in_dA,
                                                                                                G_in_dB, G_in, A, B,
                                                                                                read_GB, K_l, K_i,
                                                                                                pivot, M, pivot_val,
                                                                                                pivot_val_dA,
                                                                                                pivot_val_dB)

    return arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB


@njit
def use_diag_pivot(A, B, M, cutoff, params, arr0, arr1, staggered_range, arr0_dA, arr1_dA, arr0_dB, arr1_dB, strides0,
                   strides1):
    pivot = calc_diag_pivot(params)
    K_l = np.sqrt(pivot)  # automatic conversion to float
    K_i = np.sqrt(pivot + 1)  # automatic conversion to float
    G_in = np.zeros(2 * M, dtype=np.complex128)
    G_in_dA = np.zeros((2 * M,) + A.shape, dtype=np.complex128)
    G_in_dB = np.zeros((2 * M,) + B.shape, dtype=np.complex128)

    read_GB = add_tuple_tail_Array0(params, M, strides0)
    pivot_val = arr0[read_GB]
    pivot_val_dA = arr0_dA[read_GB]
    pivot_val_dB = arr0_dB[read_GB]
    GB = arr0[read_GB] * B

    ########## READ ##########
    # Array1
    for i in range(2 * M):
        if params[i // 2] > 0:
            read = strides1[1] * staggered_range[i] + strides1[2] * (params[i // 2] - 1) + add_tuple_tail_Array2(i // 2,
                                                                                                                 params,
                                                                                                                 M,
                                                                                                                 strides1)
            G_in[i] = arr1[read]
            G_in_dA[i] = arr1_dA[read]
            G_in_dB[i] = arr1_dB[read]

    ########## WRITE ##########
    G_in = np.multiply(K_l, G_in)

    # Array1
    for i in range(2 * M):
        if params[i // 2] + 1 < cutoff:
            write = strides1[1] * i + strides1[2] * params[i // 2] + add_tuple_tail_Array2(i // 2, params, M, strides1)
            arr1[write] = (GB[i] + A[i] @ G_in) / K_i[i]
            arr1_dA[write], arr1_dB[write] = calc_dA_dB(i, G_in_dA, G_in_dB, G_in, A, B, read_GB, K_l, K_i, pivot, M,
                                                        pivot_val, pivot_val_dA, pivot_val_dB)
    return arr0, arr1, arr0_dA, arr1_dA, arr0_dB, arr1_dB


@njit
def fock_representation_compact_NUMBA(A, B, G0, M, cutoff, PARTITIONS, shape0, shape2, shape11, shape1, shape0_tuple):
    '''
    Returns the PNR probabilities of a state or Choi state (by using the recurrence relation to calculate a limited number of Fock amplitudes)
    Args:
        A, B, G0 (Matrix, Vector, Scalar): ABC that are used to apply the recurrence relation
        M (int): number of modes
        cutoff (int): upper bound for the number of photons in each mode
        PARTITIONS (dict): a reference to the "global" PARTITIONS dictionary that is used to iterate over all pivots
        arr0 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the type aa,bb,...
        arr2 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the types (a+2)a,bb,... / aa,(b+2)b,... / ...
        arr11 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the types (a+1)a,(b+1)b,... / (a+1)a,b(b+1),... / a(a+1),(b+1)b,...
        arr1 (Matrix): submatrix of the fock representation that contains Fock amplitudes of the types (a+1)a,bb,... / a(a+1),bb,... / aa,(b+1)b,... / ...
        zero_tuple (tuple): tuple of length M+3 containing integer zeros
    Returns:
        Tensor: the fock representation
    '''
    arr0 = np.zeros(np.prod(shape0), dtype=np.complex128)
    arr2 = np.zeros(np.prod(shape2), dtype=np.complex128)
    arr11 = np.zeros(np.prod(shape11), dtype=np.complex128)
    arr1 = np.zeros(np.prod(shape1), dtype=np.complex128)
    strides0 = strides_from_shape(shape0)
    strides2 = strides_from_shape(shape2)
    strides11 = strides_from_shape(shape11)
    strides1 = strides_from_shape(shape1)
    arr0_dA = np.zeros(arr0.shape + A.shape, dtype=np.complex128)
    arr2_dA = np.zeros(arr2.shape + A.shape, dtype=np.complex128)
    arr11_dA = np.zeros(arr11.shape + A.shape, dtype=np.complex128)
    arr1_dA = np.zeros(arr1.shape + A.shape, dtype=np.complex128)
    arr0_dB = np.zeros(arr0.shape + B.shape, dtype=np.complex128)
    arr2_dB = np.zeros(arr2.shape + B.shape, dtype=np.complex128)
    arr11_dB = np.zeros(arr11.shape + B.shape, dtype=np.complex128)
    arr1_dB = np.zeros(arr1.shape + B.shape, dtype=np.complex128)

    arr0[0] = G0
    staggered_range = calc_staggered_range_2M(M)
    for count in range((cutoff - 1) * M):  # count = (sum_weight(pivot)-1)/2 # Note: sum_weight(pivot) = 2*(a+b+c+...)+1
        for params in get_partitions(M, count, PARTITIONS):
            if np.max(params) < cutoff:
                # diagonal pivots: aa,bb,cc,dd,...
                arr0, arr1, arr0_dA, arr1_dA, arr0_dB, arr1_dB = use_diag_pivot(A, B, M, cutoff, params, arr0, arr1,
                                                                                staggered_range, arr0_dA, arr1_dA,
                                                                                arr0_dB, arr1_dB, strides0, strides1)

                # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: aa,(b+1)b,cc,dd | ...
                for d in range(M):  # for over pivot off-diagonals
                    if params[d] < cutoff - 1:
                        arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA, arr0_dB, arr2_dB, arr11_dB, arr1_dB = use_offDiag_pivot(
                            A, B, M, cutoff, params, d, arr0, arr2, arr11, arr1, arr0_dA, arr2_dA, arr11_dA, arr1_dA,
                            arr0_dB, arr2_dB, arr11_dB, arr1_dB, strides0, strides2, strides11, strides1)

    return arr0.reshape(shape0_tuple)[0, 0], arr0_dA.reshape(shape0_tuple + A.shape)[0, 0], \
           arr0_dB.reshape(shape0_tuple + B.shape)[0, 0]


def fock_representation_compact(A, B, G0, M, cutoff):
    '''
    First initialise the submatrices of G (of which the shape depends on cutoff and M)
    and initialise a zero tuple of length M+2.
    (These initialisations currently cannot be done using Numba.)
    Then calculate the fock representation.
    '''

    shape0 = np.array([1, 1] + [cutoff] * M, dtype=np.int64)
    shape2 = np.array([1, M] + [cutoff - 2] + [cutoff] * (M - 1), dtype=np.int64)
    if M == 1:
        shape11 = np.array([1, 1, 1],
                           dtype=np.int64)  # we will never read from/write to arr11 for M=1, but Numba requires it to have correct dimensions(corresponding to the length of tuples that are used for multidim indexing, i.e. M+2)
    else:
        shape11 = np.array([3] + [M * (M - 1) // 2] + [cutoff - 1] * 2 + [cutoff] * (M - 2), dtype=np.int64)
    shape1 = np.array([1, 2 * M] + [cutoff - 1] + [cutoff] * (M - 1), dtype=np.int64)
    shape0_tuple = tuple(shape0)
    return fock_representation_compact_NUMBA(A, B, G0, M, cutoff, PARTITIONS, shape0, shape2, shape11, shape1,
                                             shape0_tuple)


def hermite_multidimensional_diagonal(A,B,G0,cutoff):
    M = A.shape[0]//2
    G,G_dA,G_dB = fock_representation_compact(A, B, G0, M, cutoff)
    G_dG0 = np.array(G / G0).astype(np.complex128)
    return G,G_dG0,G_dA,G_dB
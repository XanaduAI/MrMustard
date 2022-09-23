import numpy as np
from numba import njit, typeof, int64
from numba.typed import Dict
from scipy.special import binom

# 1. Partitions
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

# 2. helper functions to construct tuples that are used for multidimensional indexing of the submatrices of G (strides instead of tuples now)
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


# 3. Other helper functions
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


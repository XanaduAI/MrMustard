import numpy as np
np.set_printoptions(suppress=True, linewidth=250)
from mrmustard.math import Math
math = Math()  # use methods in math if you want them to be differentiable

from numba import njit, typeof, int64
# from numba.cpython.unsafe.tuple import tuple_setitem
from numba.typed import Dict
from scipy.special import binom

@njit
def len_lvl(M, N, BINOM):
    r"""Returns the size of an M-mode level with N total photons.
    Args:
        M (int) number of modes
        N (int) number of photons in level
    Returns:
        (int) the the size of an M-mode level with N total photons
    """
    return BINOM[M-1+N, N]


@njit
def get_partitions(M, N, PARTITIONS, BINOM):
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
    if (M,N) in PARTITIONS:
        return PARTITIONS[(M,N)]
    # recursive formulation:
    # (doesn't matter if it's slowish because we're caching the results)
    if M == 1:
        return np.array([[N]])
    else:
        T = 0
        pivots = np.zeros((len_lvl(M, N, BINOM), M), dtype=np.int64)
        for n in range(N+1):
            pivots[T : T + len_lvl(M-1, N-n, BINOM), :1] = n
            pivots[T : T + len_lvl(M-1, N-n, BINOM), 1:] = get_partitions(M-1, N-n, PARTITIONS)
            T += len_lvl(M-1, N-n, BINOM)
        PARTITIONS[(M,N)] = pivots
        return pivots


# Helper functions

# def calc_diag_pivot(params):
#     '''
#     return pivot in original representation of G
#     i.e. a,a,b,b,c,c,...
#     params [1D array]: [a,b,c,...]
#     '''
#     pivot = np.zeros(2 * params.shape[0], dtype=np.int64)
#     for i, val in enumerate(params):
#         pivot[2 * i] = val
#         pivot[2 * i + 1] = val
#     return pivot


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


def repeat_each_item_twice(lst):
    res = []
    for item in lst:
        res += [item, item]
    return res


def neighbour_plus(idx, arr):
    res = arr.copy()
    res[idx] += 1
    return res


def neighbour_min(idx, arr):
    res = arr.copy()
    res[idx] -= 1
    return res

def calc_dA(i,G_dA,G_in,A,B,read_GB,K_l,K_i,pivot,write):
    # print('write',write,'------------------------')
    dA = np.zeros(A.shape,dtype=np.complex128) # dA = B[i] * G_dA[tuple(read_GB)] # no displacement
    for l in range(2*M):
        read = pivot.copy()
        read[l] -= 1
        if np.min(read)>=0:
            # print('i,l,read:',i,l,read)
            # dA += K_l[l] * A[i,l] * G_dA[tuple(read)]
            for a1 in range(A.shape[0]):
                for a2 in range(A.shape[1]):
                    dA[a1,a2] += K_l[l] * A[i,l] * G_dA[tuple(read)+tuple([a1,a2])]
            # print('G_dA[tuple(read)]',G_dA[tuple(read)])
            # print(dA)
            dA[i,l] += G_in[l]
            # print(dA)
    dA /= K_i[i]
    # print(dA)

    # print('I')
    # print(B[i] * G_dA[tuple(read_GB)] / K_i[i])
    # print('II')
    # for l in range(2*M):
    #     read = pivot.copy()
    #     read[l] -= 1
    #     if np.min(read)>=0:
    #         print('i,l',i,l,K_l[l] * A[i,l] * G_dA[tuple(read)] / K_i[i])
    # print('III')
    # print(G_in / K_i[i])
    #
    # print('Robbe')
    # print(dA)
    # print('TW')
    # print(-dG_dR_ref[tuple(write)])
    return dA


def use_offDiag_pivot(A, B, M, cutoff, params, d, G, G_dA):
    pivot = calc_offDiag_pivot(params, d)
    # print('pivot:', pivot)
    K_l = np.sqrt(pivot)  # automatic conversion to float
    K_i = np.sqrt(pivot + 1)  # automatic conversion to float
    G_in = np.zeros(2 * M, dtype=np.complex128)

    read_GB = pivot
    GB = G[tuple(read_GB)] * B

    ########## READ ##########

    for l in range(2 * M):
        read = neighbour_min(l, pivot)
        if np.min(read) >= 0:
            G_in[l] = G[tuple(read)]

    ########## WRITE ##########

    G_in = np.multiply(K_l, G_in)

    # Array0
    if d == 0 or np.all(params[:d] == 0):
        write = neighbour_plus(2 * d + 1, pivot)
        G[tuple(write)] = (GB[2 * d + 1] + A[2 * d + 1] @ G_in) / K_i[2 * d + 1]  # I could absorb K_i in A and GB
        G_dA[tuple(write)] = calc_dA(2 * d + 1, G_dA, G_in, A, B, read_GB, K_l, K_i, pivot, write)

    # Array2
    if params[d] + 2 < cutoff:
        write = neighbour_plus(2 * d, pivot)
        G[tuple(write)] = (GB[2 * d] + A[2 * d] @ G_in) / K_i[2 * d]
        G_dA[tuple(write)] = calc_dA(2 * d, G_dA, G_in, A, B, read_GB, K_l, K_i, pivot, write)

    # Array11
    for i in range(d + 1, M):  # i>d
        if params[i] + 1 < cutoff:
            write = neighbour_plus(2 * i, pivot)
            G[tuple(write)] = (GB[2 * i] + A[2 * i] @ G_in) / K_i[2 * i]  # WRITE red (1010)
            G_dA[tuple(write)] = calc_dA(2 * i, G_dA, G_in, A, B, read_GB, K_l, K_i, pivot, write)

            write = neighbour_plus(2 * i + 1, pivot)
            G[tuple(write)] = (GB[2 * i + 1] + A[2 * i + 1] @ G_in) / K_i[2 * i + 1]  # WRITE green (1001)
            G_dA[tuple(write)] = calc_dA(2 * i + 1, G_dA, G_in, A, B, read_GB, K_l, K_i, pivot, write)

    for i in range(d):  # i<d
        if params[i] + 1 < cutoff:
            write = neighbour_plus(2 * i + 1, pivot)
            G[tuple(write)] = (GB[2 * i + 1] + A[2 * i + 1] @ G_in) / K_i[2 * i + 1]  # WRITE blue (0110)
            G_dA[tuple(write)] = calc_dA(2 * i + 1, G_dA, G_in, A, B, read_GB, K_l, K_i, pivot, write)

    return G, G_dA

def fock_representation_compact(A,B,G0,M,cutoff):
    PARTITIONS = Dict.empty(key_type=typeof((0, 0)), value_type=int64[:, :])

    BINOM = np.zeros((60, 60), dtype=np.int64)
    for m in range(BINOM.shape[0]):
        for n in range(BINOM.shape[1]):
            BINOM[m, n] = binom(m, n)

    G = np.zeros([cutoff]*(2*M),dtype=np.complex128)
    G_dA = np.zeros(G.shape + A.shape,dtype=np.complex128)

    G[tuple([0]*(2*M))] = G0
    for count in range((cutoff-1)*M): # count = (sum_weight(pivot)-1)/2 # Note: sum_weight(pivot) = 2*(a+b+c+...)+1
        for params in get_partitions(M, count, PARTITIONS, BINOM):
            if np.max(params)<cutoff:
                # diagonal pivots: aa,bb,cc,dd,...
                # G,G_dA = use_diag_pivot(A,B,M,cutoff,params,G,G_dA)

                # off-diagonal pivots: d=0: (a+1)a,bb,cc,dd,... | d=1: aa,(b+1)b,cc,dd | ...
                for d in range(M): # for over pivot off-diagonals
                    if params[d]<cutoff-1:
                        G,G_dA = use_offDiag_pivot(A,B,M,cutoff,params,d,G,G_dA)
    return G,G_dA
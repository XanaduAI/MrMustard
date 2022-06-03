import numpy as np
from numba import njit, prange
from numba.cpython.unsafe.tuple import tuple_setitem
from collections import defaultdict

# UTILS

@njit
def sum_tpl(tpl):
    N = 0
    for t in tpl:
        N += t
    return N

@njit
def up(idx, i):
    'increases idx[i] by 1'
    return tuple_setitem(idx, i, idx[i] + 1) # why not idx[:i] + (idx[i] + 1,) + idx[i+1:] ?

@njit
def down(idx, i):
    'returns a copy of idx, where idx[i] is decreased by 1'
    return tuple_setitem(idx, i, idx[i] - 1)

@njit
def tuple_verify(tpl, cutoffs):
    for i, c in enumerate(cutoffs):
        if tpl[i] > c:
            raise ValueError("The tuple doesn't respect the cutoffs")

# BINOMIAL PATHS

@njit
def reset_tpl(tpl, i, cutoffs):
    r'''Given a tuple `t`, let `S = sum(t_n, 0 <= n <= i)`. This function returns a tuple s_m such that:
        - s_0 = sum(t_0, 0 <= n <= i)
        - s_1, ..., s_i = 0
        - s_i+1 = t_i+1
    The cutoff on s_0 is used if it's smaller than s_0.
    In that case s_0 is set to the cutoff and the rest of S is added to s_1 and so on
    (that's the while loop).
    '''
    S = 0
    for j in range(i+1):
        S += tpl[j]
        tpl = tuple_setitem(tpl, j, 0)
    i = 0
    while S > 0:
        tpl = tuple_setitem(tpl, i, tpl[i]+min(S, cutoffs[i]))
        S -= cutoffs[i]
        i += 1
    return tpl

@njit
def next_tpl(tpl, i):
    r'''Returns the next tuple in increasing numerical order
    by increasing the i+1-th element of tpl and decreasing the i-th.
    Assumes that tpl[i] > 0 and that 0 <= i < len(tpl)-1.
    '''
    tpl = up(tpl, i+1)
    tpl = down(tpl, i)
    return tpl

@njit
def prev_tpl(tpl, i):
    r'''Returns the next tuple in decreasing numerical order
    by decreasing the i+1-th element of tpl and increasing the i-th.
    Assumes that tpl[i+1] > 0 and that 0 <= i < len(tpl)-1.'''
    tpl = down(tpl, i+1)
    tpl = up(tpl, i)
    return tpl




@njit
def _index_generator(tpl, cutoffs): # would be great to simplify the logic
    i = 0
    reset = False
    yield tpl
    while i < len(tpl)-1:
        # print(i, tpl)
        if tpl[i] == 0 or tpl[i+1] >= cutoffs[i+1]:
            reset = True # mark it for resetting
            i += 1 # try to increment next index
        elif reset:
            tpl = next_tpl(tpl, i)
            tpl = reset_tpl(tpl, i, cutoffs)
            reset = False
            i = 0
            yield tpl
        else:
            tpl = next_tpl(tpl, i)
            yield tpl
            


def index_generator_factory(tpl, cutoffs=None):
    if cutoffs is None:
        cutoffs = (sum_tpl(tpl),)*len(tpl)
    tuple_verify(tpl, cutoffs)
    return _index_generator(tpl, cutoffs)



####
@njit
def largest_index(tup):
    return tup.index(max(tup))

@njit
def smallest_index(tup):
    return tup.index(min(tup))

@njit
def get_pivot(tpl):
    idx = largest_index(tpl)
    return tuple_setitem(tpl[:], idx, tpl[idx]-1)

@njit
def get_lower_tuples(pivot):
    for i,p in enumerate(pivot):
        if p > 0:
            yield tuple_setitem(pivot, i, p-1)

@njit
def get_upper_tuples_with_cutoffs(pivot, cutoffs):
    for i,p in enumerate(pivot):
        if p < cutoffs[i]-1:
            yield tuple_setitem(pivot, i, p+1)

@njit
def get_upper_tuples(pivot):
    for i,p in enumerate(pivot):
        yield tuple_setitem(pivot, i, p+1)



def indices_for_tpl(tpl):
    pivot = get_pivot(tpl)
    lt = get_lower_tuples(pivot)
    return pivot, lt


INDICES = defaultdict(set)

def save_to_dict(pivot, lt, INDICES):
    N = np.sum(pivot)
    INDICES[N].add(pivot)
    for t in lt:
        INDICES[N-1].add(t)
    return INDICES

# def probs(cov, means, cutoffs):
#     A,B,C = ABC(cov, means)
#     M = len(cutoffs)
#     fat_probs = np.zeros((5,)*M + cutoffs, dtype = np.complex128)

#     for pivot in np.ndindex(cutoffs):
#         fat_probs = consume_pivot(pivot, fat_probs)  # step 1: use tpl as pivot index
#         for i in range(M):  # step 2: use up(pivot, i) as pivot indices
#             fat_probs = consume_pivot(up(pivot, i), fat_probs)

#     return fat_probs[(2,)*M]

from mrmustard.physics.fock import ABC


def fast_probs(state, cutoffs):
    A, B, C = ABC(state.cov, state.means, full=True)
    rho = np.zeros([c+2 for c in cutoffs+cutoffs], dtype=np.complex128)
    rho[(0,)*(2*len(cutoffs))] = C
    return probs_naive(A.numpy(), B.numpy(), rho, cutoffs)

@njit
def probs_naive(A, B, rho, cutoffs):
    M = len(cutoffs)
    for pivot in np.ndindex(cutoffs):
        consume_pivot(pivot, rho, A, B, cutoffs)  # step 1: use tpl as pivot index
        for i in range(len(pivot)):  # step 2: use up(pivot, i) as pivot indices
            consume_pivot(up(pivot, i)+pivot, rho, A, B, cutoffs+cutoffs)
            consume_pivot(pivot+up(pivot, i), rho, A, B, cutoffs+cutoffs)
    return rho



# def consume_pivot(pivot, fat_probs):
#     M = len(pivot)
#     center = (2,)*M
#     for i in range(M):
#         fat_probs[up(center, i) + pivot] += fat_probs[center+pivot] * B[i] / SQRT[pivot[i]+1]
#         fat_probs[down(center, i) + pivot] += fat_probs[center+pivot] * B[i+M] / SQRT[pivot[i]+1]
#         for j in range(M):
#             fat_probs[up(center, i) + pivot] += A[i,j] * SQRT[pivot[j]] * fat_probs[down(center, j) + down(pivot, i)]

SQRT = np.sqrt(np.arange(1, 1000))

@njit
def consume_pivot(pivot, G, A, B, cutoffs):
    r"""
    Fills at most M new amplitudes of G[N+1], where M is the dimension of A and b and N = sum(pivot).
    It's sparse in the sense that it runs through the nonzero values of A and b (indices in Aidx and bidx).
    Arguments:
        A, b: A matrix and b vector from the recursive representation
        Aidx, bidx: tuples of indices of the nonzero values of A and b
        G: reference to the current dictionary of amplitudes
        UP, LO: references to vectors that will hold the upper and lower indices for the current pivot
        skip: the number of upper amplitudes to skip
        pivot: the vector of indices that we use as pivot
        pivot_idx: the index of the pivot in the array G[N]
    returns:
        the squared norm of the amplitudes that we computed from this pivot
    """
    for i in range(len(pivot)):
        u = tuple_setitem(pivot, i, pivot[i]+1)
        G[u] = 0.0
        G[u] += B[i] * G[pivot]
        for j in prange(len(pivot)):
            G[u] += SQRT[pivot[j]] * A[i, j] * G[down(pivot,j)]
        G[u] /= SQRT[pivot[i]+1]

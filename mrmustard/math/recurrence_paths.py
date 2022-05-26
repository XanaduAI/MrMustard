import numpy as np
from numba import njit
from numba.cpython.unsafe.tuple import tuple_setitem
from collections import defaultdict

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
def tuple_verify(tpl, cutoffs):
    for i, c in enumerate(cutoffs):
        if tpl[i] > c:
            raise ValueError("The tuple doesn't respect the cutoffs")

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



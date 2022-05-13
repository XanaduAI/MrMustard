import numpy as np
from numba import njit
from numba.cpython.unsafe.tuple import tuple_setitem
from collections import defaultdict

@njit
def up(idx, i):
    return tuple_setitem(idx, i, idx[i] + 1)

@njit
def down(idx, i):
    return tuple_setitem(idx, i, idx[i] - 1)

@njit
def reset_tpl(tpl, i, cutoffs):
    N = 0
    for j in range(i+1):
        N += tpl[j]
        tpl = tuple_setitem(tpl, j, 0)
    i = 0
    while N > 0:
        tpl = tuple_setitem(tpl, i, tpl[i]+min(N, cutoffs[i]))
        N -= cutoffs[i]
        i += 1
        
    return tpl

@njit
def next_tpl(tpl, i):
    tpl = up(tpl, i+1)
    tpl = down(tpl, i)
    return tpl

@njit
def tuple_verify(tpl, cutoffs):
    for i, c in enumerate(cutoffs):
        if tpl[i] > c:
            raise ValueError("The tuple doesn't respect the cutoffs")

@njit
def _index_generator(tpl, cutoffs):
    i = 0
    reset = False
    yield tpl
    while i < len(tpl)-1:
        # print(i, tpl)
        if tpl[i] == 0 or tpl[i+1] >= cutoffs[i+1]:
            reset = True
            i += 1
        elif reset:
            tpl = next_tpl(tpl, i)
            tpl = reset_tpl(tpl, i, cutoffs)
            reset = False
            i = 0
            yield tpl
        else:
            tpl = next_tpl(tpl, i)
            yield tpl
            

def index_generator(tpl, cutoffs=None):
    if cutoffs is None:
        cutoffs = (sum(tpl),)*len(tpl)
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
            t = pivot[:]
            yield tuple_setitem(t, i, p-1)



def indices_for_tpl(tpl):
    pivot = get_pivot(tpl)
    lt = get_lower_tuples(pivot)
    return pivot, lt


INDICES = defaultdict(defaultdict(set))

def save_to_dict(pivot, lt, INDICES):
    N = np.sum(pivot)
    INDICES[N].add(pivot)
    for t in lt:
        INDICES[N-1].add(t)
    return INDICES



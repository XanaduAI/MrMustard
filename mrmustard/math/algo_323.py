import numpy as np
from numba.cpython.unsafe.tuple import tuple_setitem
from numba import njit
from collections import defaultdict
from copy import deepcopy

# @njit
def largest(tup):
    return np.argmax(np.array(tup))

# @njit
def smallest_nonzero(tup):
    # e.g. if tup is (0,0,0,7,0,1), returns 3
    smallest = tup[0]
    position = 0
    for i, val in enumerate(tup):
        if val < smallest and val > 0:
            smallest = val
            position = i
    return position

# @njit
def get_pivot(tup, strategy):
    m = strategy(tup)
    tuple_setitem(tup, m, tup[m]-1)
    return tup

# @njit
def get_lower_tuples(pivot):
    for i,p in enumerate(pivot):
        if p > 0:
            tuple_setitem(pivot, i, p-1)
            yield tup.copy()
            tuple_setitem(pivot, i, p)


@njit
def largest(vec):
    return np.argmax(vec)

@njit
def smallest_nonzero(vec):
    # e.g. if tup is (0,0,0,7,0,1), returns 3
    smallest = vec[0]
    position = 0
    for i, val in enumerate(vec):
        if val < smallest and val > 0:
            smallest = val
            position = i
    return position

@njit
def get_pivot(vec, strategy):
    m = strategy(vec)
    vec[m] -= 1
    return vec

@njit
def get_lower_tuples(pivot):
    for i,p in enumerate(pivot):
        if p > 0:
            pivot[i] -= 1
            yield vec
            pivot[i] += 1






AMPS = defaultdict(set)  # will use tuples as indices

@njit
def find_indices(vec) -> (set, set):
    N = np.sum(vec)
    pivot = get_pivot(vec, largest) 
    AMPS_N1 = {}
    AMPS_N2 = {}
    AMPS_N1.add(pivot.copy())
    for t in get_lower_tuples(pivot):
        AMPS_N2.add(t.copy())
    return AMPS_N1, AMPS_N2

def find_all_indices(vec, AMPS):
    N = np.sum(vec)
    AMPS[N].add(vec.copy())
    for n in range(N, 0, -1):
        for t in AMPS[n]:
            AMPS_N1, AMPS_N2 = find_indices(t)
            AMPS[n-1].update(AMPS_N1)
            AMPS[n-2].update(AMPS_N2)
    return AMPS

def find_all_diagonals(ds, AMPS):
    for idx in np.ndindex(tuple(np.array(ds)+1)):
        tup = tuple(x for d in idx for x in [d,d])
        find_all_indices(tup, AMPS)
    return AMPS
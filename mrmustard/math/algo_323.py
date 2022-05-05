import numpy as np
from numba.cpython.unsafe.tuple import tuple_setitem
from collections import defaultdict
 
def largest(tup):
    return np.argmax(tup)

def smallest_nonzero(tup):
    # e.g. if tup is (0,0,0,7,0,1), returns 3
    for i, val in enumerate(tup):
        if val > 0:
            return i

def get_pivot(tup, strategy):
    m = strategy(tup)
    return tup[:m] + (tup[m]-1,) + tup[m+1:]

def get_lower_tuples(pivot):
    for i,p in enumerate(pivot):
        if p > 0:
            yield pivot[:i] + (p-1,) + pivot[i+1:]

AMPS = defaultdict(set)  # will use tuples as indices


# e.g. we want to reach the index (6,6)

def find_indices(tup, AMPS, strategy=largest):
    N = np.sum(tup)
    pivot = get_pivot(tup, strategy)
    AMPS[N-1].add(pivot)
    for t in get_lower_tuples(pivot):
        AMPS[N-2].add(t)

def find_all_indices(tup, AMPS=None):
    if AMPS is None:
        AMPS = defaultdict(set)
    N = np.sum(tup)
    AMPS[N].add(tup)
    for n in range(N, 0, -1):
        for t in AMPS[n]:
            find_indices(t, AMPS)
    return AMPS

def find_all_diagonals(ds, AMPS=None):
    if AMPS is None:
        AMPS = defaultdict(set)
    for idx in np.ndindex(tuple(np.array(ds)+1)):
        tup = tuple(x for d in idx for x in [d,d])
        find_all_indices(tup, AMPS)
    return AMPS
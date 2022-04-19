from numba.typed import Dict
from numba import njit, typeof
import numpy as np
from numba.types import int64

from scipy.special import binom

BINOM = np.zeros((60,60),dtype=np.int64)
for m in range(BINOM.shape[0]):
    for n in range(BINOM.shape[1]):
        BINOM[m,n] = binom(m,n)
        
SQRT = np.sqrt(np.arange(1000))
PIVOTS = Dict.empty(key_type=typeof((0,0)), value_type=int64[:,:])
SKIPS = Dict.empty(key_type=typeof((0,0)), value_type=int64[:])

@njit
def num_amps(M, N):
    return BINOM[M-1+N, N]

@njit
def all_pivots(M, N, PIVOTS):
    if (M,N) in PIVOTS:
        return PIVOTS[(M,N)]
    T = 0
    if M == 1:
        return np.array([[N]])
    pivots = np.zeros((num_amps(M, N), M), dtype=np.int64)
    for n in range(N+1):
        pivots[T : T + num_amps(M-1, N-n), :1] = n
        pivots[T : T + num_amps(M-1, N-n), 1:] = all_pivots(M-1, N-n, PIVOTS)
        T += num_amps(M-1, N-n)
    PIVOTS[(M,N)] = pivots
    return pivots

@njit
def all_skips(M, N, SKIPS):
    if (M,N) in SKIPS:
        return SKIPS[(M,N)]
    T = 0
    skips = np.zeros(num_amps(M, N), dtype=np.int64)
    for m in range(M):
        skips[T : T + BINOM[m+N-1, N-1]] = m
        T += BINOM[m+N-1, N-1]
    SKIPS[(M,N)] = skips
    return skips

@njit
def index(vec):
    total = 0
    M = len(vec)
    N = np.sum(vec)
    for m,t in enumerate(vec):
        L = M-m-1
        total += BINOM[L+N,L] - BINOM[L+N-t,L]
        N -= t
    return total

@njit
def upper(vec, skip=0):
    UP = np.zeros(len(vec)-skip, dtype=np.int64)
    for l in range(len(vec)-skip):
        vec[l] += 1
        UP[l] = index(vec)
        vec[l] -= 1
    return UP

@njit
def lower(vec):
    LO = np.zeros(len(vec), dtype=np.int64)
    for l in range(len(vec)):
        if vec[l] > 0:
            vec[l] -= 1
            LO[l] = index(vec)
            vec[l] += 1
    return LO

@njit
def consume_one_pivot(A, b, Aidx, bidx, G, UP, DOWN, pivot, pivot_weight, pivot_idx, b_nonzero):
    r"""
    Fills at most M new amplitudes into G, where M is the dimension of the indices of A (or b).
    It's sparse in the sense that it runs through the nonzero values of A and b (indices in Aidx and bidx).
    Arguments:
        A, b: A matrix and b vector from the recursive representation
        Aidx, bidx: tuples of indices of the nonzero values of A and b
        G: the current dictionary of amplitudes
        UP, DOWN: lists of the upper and lower indices that we reach from the pivot
        pivot: the vector of indices that we use as pivot
        pivot_weight: the weight of the pivot (sum of the elements of pivot)
        pivot_idx: the index of the pivot in the array G[pivot_weight]
        b_nonzero: if False, b is assumed to be zero
        f: a function to fold alongside the computation of the amplitudes
        F: the initial value of f
    """
    amplitudes = np.array([0.0] * len(UP), dtype=np.complex128)
    if b_nonzero:
        for m in bidx:
            if m > len(UP):
                break
            amplitudes[m] += b[m] * G[pivot_weight][pivot_idx] / SQRT[pivot[m]+1]
    for m,n in Aidx: # TODO: A is symmetric, so we can iterate over e.g. the upper triangle only
        if m < len(UP):  # TODO: check the order of the indices and in case it's in m order break
            amplitudes[m] -= SQRT[pivot[n]] * A[m, n] * G[pivot_weight-1][DOWN[n]] / SQRT[pivot[m]+1]
    for i,u in enumerate(UP):
        G[pivot_weight+1][u] = amplitudes[i]
    return np.linalg.norm(amplitudes)**2

@njit
def consume_one_pivot_vjp(A, b, Aidx, bidx, G, UP, DOWN, pivot, pivot_weight, pivot_idx, b_nonzero, f, F, dL_dA, dL_db, dL_dG):
    r"""
    Computes the vector-jacobian product dL_dG @ dG_dA and dL_dG @ dG_db.
    """
    # dL_dA = dL_dG @ dG_dA (1d @ 3d)
    # dL_db = dL_dG @ dG_db (1d @ 2d)
    # i.e. we have to sum along the G_up vector only dimension
    if b_nonzero:
        for i in bidx:
            dL_db[i] += dL_dG[pivot_weight+1][UP[i]] * G[pivot_weight][pivot_idx] / SQRT[pivot[i]+1]
    for i,j in Aidx:
        dL_dA[i,j] -= dL_dG[pivot_weight+1][UP[i]] * SQRT[pivot[j]] * G[pivot_weight-1][DOWN[j]] / SQRT[pivot[i]+1]
    return dL_dA, dL_db


# from rich.table import Table
# from rich import print as rprint
# @njit
# def fill_weight_plus_one(A, b, Aidx, bidx, G, weight, b_nonzero, f, F):
#     r""" Fills all the amplitudes with index of weight N+1 in G, using the diverge
#      algorithm. At the same time, the function f is called for each pivot of weight 'weight',
#      and its result is folded as F_i+1 = f(F_i, G[N+1]_i+1).

#     Args:
#         A, b: A matrix and b vector from the recursive representation
#         Aidx, bidx: indices of the non-zero entries in A and b
#         G: the dictionary of amplitudes (weight -> vectorized amplitudes)
#         weight: the weight of the pivot
#         b_nonzero: whether b is nonzero
#         f: a function to fold alongside the computation of the amplitudes
#         F: the initial value of f
#     """
#     M = A.shape[-1]
#     skip = 0
#     i = 0 
#     r = False
#     pivot = np.zeros(M, dtype=np.int64)  # e.g. (0,0,N) -> (0,1,N-1) -> (0,2,N-2) -> ... -> (N,0,0)
#     pivot[-1] = weight
#     pivot_idx = 0 
#     # table = Table(title=f"M={M}, weight={weight}")
#     # table.add_column("Pivot", justify="center")
#     # table.add_column("Upper 1d", justify="left")
#     # table.add_column("Lower 1d", justify="left")
#     while pivot[0] < weight:
#         # join the three lists of integers (pivot, upper(pivot), lower(pivot)), and align the columns
#         # so that the next time they are printed, they are aligned with the column names:
#         # table.add_row(str(pivot), str(upper(pivot, skip)), str(lower(pivot)))
#         F = consume_one_pivot(A, b, Aidx, bidx, G, upper(pivot,skip), lower(pivot), pivot, weight, pivot_idx, b_nonzero, f, F)
#         if i == skip:
#             skip += 1
#         pivot,i,r = next_pivot(pivot,i,r)
#         pivot_idx += 1
#     F = consume_one_pivot(A, b, Aidx, bidx, G, upper(pivot,skip), lower(pivot), pivot, weight, pivot_idx, b_nonzero, f, F)
#     # table.add_row(str(pivot), str(upper(pivot, skip)), str(lower(pivot)))
#     # print the table
#     # rprint(table)
#     return F

# @njit  # TODO should we lru_cache this function?
# def all_pivots2(M,N):
#     r""" Returns a list of all the possible pivots of weight N and length M.
#     """
#     pivots = np.zeros((BINOM[M+N-1, N], M), dtype=np.int64)
#     idxs = np.zeros(len(pivots), dtype=np.int64)
#     skips = np.zeros(len(pivots), dtype=np.int64)
#     pivots[0,-1] = N
#     r = False
#     for i in range(len(pivots)):
#         pivots[i+1], idxs[i+1], r = next_pivot(pivots[i].copy(), idxs[i], r)
#         if idxs[i] == skips[i]:
#             skips[i] += 1
#         else:
#             skips[i] = idxs[i]
#     return skips, pivots


from numba import prange

@njit(parallel=False)
def fill_weight_plus_one(A, b, Aidx, bidx, G, weight, b_nonzero, PIVOTS, SKIPS):
    r""" Fills all the amplitudes with index of weight N+1 in G, using the diverge
     algorithm. At the same time, the function f is called for each pivot of weight 'weight',
     and its result is folded as F_i+1 = f(F_i, G[N+1]_i+1).

    Args:
        A, b: A matrix and b vector from the recursive representation
        Aidx, bidx: indices of the non-zero entries in A and b
        G: the dictionary of amplitudes (weight -> vectorized amplitudes)
        weight: the weight of the pivot
        b_nonzero: whether b is nonzero
        f: a function to fold alongside the computation of the amplitudes
        F: the initial value of f
    """
    M = A.shape[-1]
    F = 0.0
    skips = all_skips(M, weight, SKIPS)
    pivots = all_pivots(M, weight, PIVOTS)
    for idx, pivot in enumerate(pivots):
        F += consume_one_pivot(A, b, Aidx, bidx, G, upper(pivot, skips[idx]), lower(pivot), pivot, weight, idx, b_nonzero)
        # F = F + f
    return F

# @njit(parallel=True)
# def fill_weight_plus_one_parallel(A, b, Aidx, bidx, G, weight, b_nonzero):
#     r""" Fills all the amplitudes with index of weight N+1 in G, using the diverge
#      algorithm. At the same time, the function f is called for each pivot of weight 'weight',
#      and its result is folded as F_i+1 = f(F_i, G[N+1]_i+1).

#     Args:
#         A, b: A matrix and b vector from the recursive representation
#         Aidx, bidx: indices of the non-zero entries in A and b
#         G: the dictionary of amplitudes (weight -> vectorized amplitudes)
#         weight: the weight of the pivot
#         b_nonzero: whether b is nonzero
#         f: a function to fold alongside the computation of the amplitudes
#         F: the initial value of f
#     """
#     M = A.shape[-1]
#     F = 0.0
#     skips = all_skips(M, weight)
#     pivots = all_pivots(M, weight)
#     for idx in prange(len(pivots)):
#         pivot = pivots[idx]
#         F += consume_one_pivot(A, b, Aidx, bidx, G, upper(pivot, skips[idx]), lower(pivot), pivot, weight, idx, b_nonzero)
#     return F

@njit(parallel=True)
def fill_weight_plus_one_parallel(A, b, Aidx, bidx, G, weight, b_nonzero, PIVOTS, SKIPS):
    r""" Fills all the amplitudes with index of weight N+1 in G, using the diverge
     algorithm. At the same time, the function f is called for each pivot of weight 'weight',
     and its result is folded as F_i+1 = f(F_i, G[N+1]_i+1).

    Args:
        A, b: A matrix and b vector from the recursive representation
        Aidx, bidx: indices of the non-zero entries in A and b
        G: the dictionary of amplitudes (weight -> vectorized amplitudes)
        weight: the weight of the pivot
        b_nonzero: whether b is nonzero
        f: a function to fold alongside the computation of the amplitudes
        F: the initial value of f
    """
    M = A.shape[-1]
    skips = all_skips(M, weight, SKIPS)
    pivots = all_pivots(M, weight, PIVOTS)
    F = 0.0#np.zeros(len(pivots), dtype=np.float64)
    for idx in prange(len(pivots)):
        pivot = pivots[idx].copy()
        F += consume_one_pivot(A, b, Aidx, bidx, G, upper(pivot, skips[idx]), lower(pivot), pivot, weight, idx, b_nonzero)
    return F#np.sum(F)


def fill_all_fold_norm(A, b, C, min_norm=0.99, parallel=False):
    r""" Fills all the amplitudes in G, using the sparse fold
    algorithm while accumulating the norm. It stops when the norm reaches min_norm.

    Args:
        A, b, C: A matrix, b vector and C scalar from the recursive representation
        min_norm: the minimum norm to reach
    """
    # 1. Sparse indices
    Aidx = np.transpose(np.nonzero(A))
    bidx = np.nonzero(b)

    # 2. Some constants
    b_nonzero = len(bidx[0]) > 0
    weight = 1-int(b_nonzero)  # odd levels are zero so we build 0,2,4,...
    M = A.shape[-1]

    # 3. Initialize norm and the dictionary of amplitudes
    norm_squared = np.abs(C)**2
    from numba.typed import Dict # value type is array complex128
    from numba.types import int64
    from numba import typeof
    G = Dict.empty(key_type=int64, value_type=typeof(np.array([C])))
    G[0] = np.array([C])

    # 4. Fill the rest of the amplitudes and accumulate the norm
    while np.sqrt(norm_squared) < min_norm:
        # print(weight, np.sqrt(norm_squared))
        G[weight + 1 - b_nonzero] = np.zeros(BINOM[M + weight - b_nonzero, weight + 1 - b_nonzero], dtype=np.complex128)
        if parallel:
            norm_ = fill_weight_plus_one_parallel(A, b, Aidx, bidx, G, weight, b_nonzero, PIVOTS, SKIPS)
        else:
            norm_ = fill_weight_plus_one(A, b, Aidx, bidx, G, weight, b_nonzero, PIVOTS, SKIPS)
        # print('norm_=',norm_,f)
        norm_squared += norm_
        weight += 2 - b_nonzero
    
    return G, norm_squared


################################################################################
#                                                                              #
#                               old code                                       #
#                                                                              #
################################################################################
@njit
def next_pivot(vec, i=0, reset=False):
    r"""Computes the next pivot (vector of integers) given the current pivot and
    an index specifying which of the integers is going to be decreased next.
    It recursively works its way through all the tuples of integers with constant sum.
    The results are in numerical order, i.e. it's always the case that
    next_pivot(vec) > vec if we interpret the integers in the vectors as digits.

    Warning 1: doesn't stop after (sum(vec),0,0,...,0)
    
    Usage:
    next_pivot((0,0,3), 0)  # use np.array()
    > (0,1,2),0
    next_pivot((0,1,2), 0)
    > (0,2,1),0
    next_pivot((0,2,1), 0)
    > (0,3,0),1
    next_pivot((0,3,0), 1)
    > (1,0,2),0
    """
    vec[-i-1] -= 1
    vec[-i-2] += 1
    if reset:
        vec[-1] = np.sum(vec[-1-i:-1])
        vec[-1-i:-1] = 0
        if vec[-1] == 0:
            i += 1
            reset = True
        else:
            i = 0
            reset = False
    elif vec[-i-1] == 0:
        reset = True
        i += 1
    return vec, i, reset


def norm_fill(A, b, C, min_norm=0.99):
    norm = np.abs(C)**2
    weight = int(no_b)
    no_b = np.linalg.norm(b)< 1e-15
    G_lo = np.zeros(0, dtype=np.complex128)
    G = np.array([C])
    yield G, norm, weight
    while np.sqrt(norm) < min_norm:
        if no_b:
            G_lo, G = G, np.zeros(0, dtype=np.complex128)
        G, G_lo, norm_ = fill_1up(A, b, G, G_lo, weight, no_b)
        norm += norm_
        weight += 1 + int(no_b)
        yield G, norm, weight


def norm_fill_vjp(A, b, C, Gdict, dL_dGdict):
    weight = int(no_b)
    no_b = np.linalg.norm(b)< 1e-15
    dL_dA = np.zeros(A.shape, dtype=np.complex128)
    dL_db = np.zeros(b.shape, dtype=np.complex128)
    dL_dC = np.sum(dL_dG_up)
    for i,G_lo in enumerate(Glist):
        G = Glist[i+1]
        if no_b:
            G_lo = G # if no_b only G_lo is used
        fill_1up_vjp(A, b, G, G_lo, dL_dA, dL_db, dL_dG_up, weight, no_b)
        weight += 1 + int(no_b)
    return dL_dA, dL_db

import tensorflow as tf

@tf.custom_gradient
def norm_fill_tf(A, b, C, min_norm=0.99):
    Glist = [(weight, G) for G, _, weight in tf.py_function(norm_fill, [A, b, C, min_norm], [tf.complex128, tf.complex128, tf.float64])]
    def grad(dL_dG_up):
        return tf.py_function(norm_fill_vjp, [A, b, C, Glist, dL_dG_up], [tf.complex128, tf.complex128])
    return G, grad

### TUPLE VERSION (OLD)
from numba.cpython.unsafe.tuple import tuple_setitem

@njit
def next_tpl(tpl, i, reset=False):
    r"""Computes the next tuple of indices given the current tuple
    and the current index.
    """
    if tpl[i] == 0:
        return next_tpl(tpl, i+1, reset=True)
    else:
        tpl = tuple_setitem(tpl, i, tpl[i] - 1)
        tpl = tuple_setitem(tpl, i+1, tpl[i+1] + 1)
    if reset:
        _sum = 0
        for j in range(1, i+1):
            _sum += tpl[j]
            tpl = tuple_setitem(tpl, j, 0)
        tpl = tuple_setitem(tpl, 0, _sum)
        i = 0
    return tpl, i

@njit
def fill_one(A, b, G, tpl):
    i = 0
    for i, val in enumerate(tpl):
        if val > 0:
            break
    ki = dec(tpl, i)
    u = b[i] * G[ki]
    for l, kl in remove(ki):
        u -= SQRT[ki[l]] * A[i, l] * G[kl]
    G[tpl] = u / SQRT[tpl[i]]
    return G[tpl]

@njit
def fill_all(A, b, G, photons, tpl):
    # tpl is the first index to fill
    t=0
    fill_one(tpl, A, b, G)
    while tpl[-1] < photons:
        tpl, t = next_tpl(tpl, t)
        fill_one(A, b, G, tpl)
    return G

from typing import Callable

@njit
def fill_all_fold_f(A, b, G, photons, tpl, f:Callable, f0):
    # tpl is the first index to fill
    t=0
    g = fill_one(tpl, A, b, G)
    fval = f(f0, g, tpl)
    while tpl[-1] < photons:
        tpl, t = next_tpl(tpl, t)
        g = fill_one(A, b, G, tpl)
        fval = f(fval, g, tpl)
    return G, fval

# e.g. inner product with another state
def inprod(other):
    def f(fval, g, tpl):
        return fval + g * other[tpl]
    return f

def fast_inner_prod (A, b, C, other):
    G = np.zeros((cutoff,)*len(A)//2, dtype=np.complex128)
    G[(0,)*M] = C
    return fill_all_fold_f(A, b, G, photons, tpl, inprod(other), 0)[1]

def hermite_multidimensional_n(A, b, C, photons):
    r"""Numba implementation of the multidimensional Hermite polynomials
    up to n photons evaluated at A, b.
    """
    G = np.zeros((cutoff,)*len(A)//2, dtype=np.complex128)
    G[(0,)*M] = np.conj(C)
    P = range(2, n+1, 2) if np.allclose(b, 0) else range(1, n+1)
    for photons in P:
        fill_all(A, b, G, photons, tuple([photons]+[0]*(G.ndim-1)))
    return G

# gradients (vjp)

@njit
def fill_one_vjp(A, b, G, tpl, dL_dA, dL_db, dL_dG):
    # dL_dA = dL_dG @ dG_dA (M-d @ (M+3)-d)
    # dL_db = dL_dG @ dG_db (M-d @ (M+2)-d)
    # i.e. we have to sum along the dL_dG dimensions
    i = 0
    for i, val in enumerate(tpl):
        if val > 0:
            break
    ki = dec(tpl, i)
    dL_db[i] += dL_dG[ki] * G[ki]
    for l, kl in remove(ki):
        dL_dA[i, l] -= SQRT[ki[l]] * G[kl] * dL_dG[ki] / SQRT[tpl[i]]

@njit
def fill_all_vjp(A, b, G, photons, tpl, dL_dA, dL_db, dL_dG):
    t=0
    fill_one_vjp(A, b, G, tpl, dL_dA, dL_db, dL_dG)
    while tpl[-1] < photons:
        tpl, t = next_tpl(tpl, t)
        dL_dA, dL_db = fill_one_vjp(A, b, G, tpl, dL_dG)


def hermite_multidimensional_n_vjp(A, b, C, photons, dL_dG):
    r"""Gradient of the multidimensional Hermite polynomials
    """
    dL_dA = np.zeros(A.shape, dtype=np.complex128)
    dL_db = np.zeros(b.shape, dtype=np.complex128)
    dL_dC = np.sum(dL_dG)
    zeros = [0]*(G.ndim-1)
    if np.isclose(np.linalg.norm(b), 0):
        P = range(2, photons+1, 2)
    else:
        P = range(1, photons+1)
    for photons in P:
        fill_all_vjp(A, b, G, tuple([photons]+zeros), dL_dA, dL_db, dL_dG)
    return dL_dA, dL_db, dL_dC


from collections import defaultdict

class BinomialG:
    def __init__(self, modes):
        if modes < 1:
            raise ValueError("modes must be >= 1")
        self.multiplets = dict()
        self.modes = modes
        
    def __getitem__(self, tpl):
        if len(tpl) > self.modes:
            raise IndexError("too may indices")
        if len(tpl) == self.modes: # a single amplitude
            return self.multiplets[index(tpl)]
        # otherwise we return a new BinomialG with only the relevant modes and amplitudes
        N = np.sum(tpl)
        G = BinomialG(self.modes - len(tpl))
        for n in range(N, max(self.multiplets)+1):
            start = tpl_index(tpl + (0,)*(self.modes-len(tpl)-1) + (n,))
            end = tpl_index(tpl + (n,) + (0,)*(self.modes-len(tpl)-1))
            G.multiplets[n-N] = self.multiplets[n][start:end+1]
        return G




def map_f(A, b, f: Callable):
    r"""Map a function f over the coefficients of a Hermite polynomial.
    """
    return hermite_multidimensional_n(A, b, f(C), photons)

def fold_f(A, b, f: Callable, f0):
    r"""Fold a function f over the coefficients of a Hermite polynomial.
    """
    G = np.zeros((cutoff,)*len(A)//2, dtype=np.complex128) # or something...
    return fill_all_fold_f(A, b, G, photons, tpl, f, f0)[1]



@njit
def project_fill_rest(A, b, C, G:dict, proj_tup, renormalize=True, f_fold = None):
    r"""Produces (p1, p2, ..., pn, x, y, ...) where p1, p2, ..., pn are the
    coefficients of the projection of the Hermite polynomial.
    """
    M = len(A) # number of indices (number of modes if pure state)
      # total number of photons -> vectorized amplitudes in lex. order
    for tpl in np.ndindex(proj_tup):
        N = np.sum(tpl)
        if N not in G:
            G[N] = np.zeros(BINOM[M+N-1,N], dtype=np.complex128)
        fill_one(A, b, G[np.sum(tpl)], tpl+(0,)*(M-len(tpl)))
    # now proj_tpl + (0,)*(M-len(proj_tup)) is the index of the vacuum amplitude of the projection


# OK, so this is a bit of a mess. Let's keep the vector methods instead of the tuple index ones.

# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import numpy as np
from numba import njit, typeof, prange
from numba.types import complex128
from numba.typed import Dict, List
from scipy.special import binom

# precomputing binomials and square roots

BINOM = np.zeros((60,60),dtype=np.int64)
for m in range(BINOM.shape[0]):
    for n in range(BINOM.shape[1]):
        BINOM[m,n] = binom(m,n)
        
SQRT = np.sqrt(np.arange(1000))

# These are just for an attempt at parallelizing numba



@njit
def len_lvl(M, N):
    r"""Returns the size of an M-mode level with N total photons.
    Args:
        M (int) number of modes
        N (int) number of photons in level
    Returns:
        (int) the the size of an M-mode level with N total photons
    """
    return BINOM[M-1+N, N]


@njit
def lvl_pivots(M, N, PIVOTS):
    r"""Returns an array of pivots for the given M and N (num modes and total photons).
    If the pivots are already computed it returns them from the PIVOTS dictionary,
    otherwise it fills the PIVOTS[(M,N)] dictionary entry.
    Args:
        M (int) number of modes
        N (int) number of photons in level
        PIVOTS (dict) a reference to the "global" PIVOTS dictionary
    Returns:
        (2d array) the array of pivots for the given level
    """
    if (M,N) in PIVOTS:
        return PIVOTS[(M,N)]
    # recursive formulation:
    # (doesn't matter if it's slowish because we're caching the results)
    if M == 1:
        return (N,)
    else:
        T = 0
        pivots = np.zeros((len_lvl(M, N), M), dtype=np.int64)
        for n in range(N+1):
            pivots[T : T + len_lvl(M-1, N-n), :1] = n
            pivots[T : T + len_lvl(M-1, N-n), 1:] = lvl_pivots(M-1, N-n, PIVOTS)
            T += len_lvl(M-1, N-n)
        PIVOTS[(M,N)] = pivots
        return pivots


@njit
def lvl_skips(M, N, SKIPS):
    r"""Returns the vector of skips for a given M and N (num modes and total photons).
    If the skips are already computed it returns them from the SKIPS dictionary,
    otherwise it fills the SKIPS[(M,N)] dictionary entry.
    What are the skips? When computing an upper level, we go through a loop
    on zip(PIVOTS[(M,N)], SKIPS[(M,N)]) which gives pairs (p, s) so that we skip the
    last `s` upper amplitudes when consuming the pivot `p`.
    Args:
        M (int) number of modes
        N (int) number of photons in level
        SKIPS (dict) a reference to the "global" SKIPS dictionary
    Returns:
        (1d array) the vector of skips for the given level
    """
    if (M,N) in SKIPS:
        return SKIPS[(M,N)]
    T = 0
    skips = np.zeros(len_lvl(M, N), dtype=np.int64)
    for m in range(M):
        skips[T : T + len_lvl(m+1, N-1)] = m
        T += len_lvl(m+1, N-1)
    SKIPS[(M,N)] = skips
    return skips


@njit
def index(pivot):
    r"""Returns the binomial index (along 1-dim vector) of a given pivot.
    E.g. index(np.array([1,0,1])) = 3 (comes after 002, 011, 020).
    Args:
        pivot (1d array): the array of M integers that identifies a single amplitude
    Returns:
        (int) the index of `pivot` in the 1-dim level array.
    """
    idx = 0
    M = len(pivot)
    N = np.sum(pivot)
    for m,t in enumerate(pivot):
        idx += len_lvl(M-m, N) - len_lvl(M-m, N-t)
        N -= t
    return idx


@njit
def upper(pivot, UP, skip):
    r"""Returns the binomial indices (along 1D vector of amplitudes with N+1 photons)
    of the amplitudes that we can compute when we consume `pivot`.
    Note: UP is passed by reference, so that it can be modified and not reallocated each time.
    Args:
        pivot (1-dim array of ints): the pivot of length M
        UP (1-dim array of ints): the array to fill with binomial indices
    Returns:
        UP: the filled array
    """
    for l in range(len(UP) - skip):
        pivot[l] += 1
        UP[l] = index(pivot)
        pivot[l] -= 1
    return UP


@njit
def lower(pivot, LO):
    r"""Returns the binomial indices (along 1D vector of amplitudes with N-1 photons)
    of the amplitudes that we need to read when we consume `pivot`.
    Note: LO is passed by reference, so that it can be modified and not reallocated each time.
    Args:
        pivot (np.array): the pivot of length M
        LO: the array to fill with binomial indices
    Returns:
        LO: the filled array
    """
    for l in range(len(pivot)):
        if pivot[l] > 0:
            # the if is because we cannot lower an index that is already 0.
            # Note that LO[l] in that case is a spurious value, but it will be ignored
            pivot[l] -= 1
            LO[l] = index(pivot)
            pivot[l] += 1
    return LO


@njit
def consume_one_pivot(A, b, Aidx, bidx, G, UP, LO, skip, pivot, pivot_idx):
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
    N = np.sum(pivot)
    norm_squared = 0.0
    UP = upper(pivot, UP, skip)
    LO = lower(pivot, LO)
    for m in range(len(UP)-skip):
        amplitude = 0.0
        if m in bidx:
            amplitude += b[m] * G[N][pivot_idx]
        for n in Aidx[m]: # TODO: A is symmetric, so we could iterate over e.g. the upper triangle only and then x2
            amplitude += SQRT[pivot[n]] * A[m, n] * G[N-1][LO[n]]
        amplitude /= SQRT[pivot[m]+1]
        G[N+1][UP[m]] = amplitude
        norm_squared += np.abs(amplitude)**2
    return norm_squared


# @njit
# def consume_one_pivot_vjp(A, b, Aidx, bidx, G, UP, LO, skip, pivot, pivot_idx, dL_dA, dL_db, dL_dG):
#     r"""
#     Computes the vector-jacobian product dL_dG @ dG_dA and dL_dG @ dG_db.
#     """
#     N = np.sum(pivot)
#     UP = upper(pivot, UP, skip)
#     LO = lower(pivot, LO)
#     for m in range(len(UP)-skip):
#         if m in bidx:
#             dL_db[m] += dL_dG[N+1][UP[m]] * G[N][pivot_idx] * SQRT[pivot[m]+1]
#         for n in Aidx[m]:
#             dL_dA[m,n] += 1/2 * dL_dG[N+1][UP[m]] * SQRT[pivot[n]+1-np.int(m==n)] * SQRT[pivot[m]+1] * G[N-1][LO[n]]
#     return dL_dA, dL_db


@njit
def fill_N_plus_one(A, b, Aidx, bidx, G, N, PIVOTS, SKIPS):
    r""" Fills all the amplitudes with index of weight N+1 in G, using the diverge algorithm.

    Args:
        A, b: A matrix and b vector from the recursive representation
        Aidx, bidx: indices of the non-zero entries in A and b
        G: the dictionary of amplitudes (N -> vectorized amplitudes)
        N: the weight of the pivots
    """
    M = A.shape[-1]
    norm_squared = 0.0
    pivots = lvl_pivots(M, N, PIVOTS)
    skips = lvl_skips(M, N, SKIPS)
    UP = np.zeros(M, dtype=np.int64)
    LO = np.zeros(M, dtype=np.int64)
    for i, pivot in enumerate(pivots):
        norm_squared += consume_one_pivot(A, b, Aidx, bidx, G, UP, LO, skips[i], pivot, i)
    return norm_squared


# @njit(parallel=False)
# def fill_N_plus_one_vjp(A, b, Aidx, bidx, G, N, PIVOTS, SKIPS, dL_dA, dL_db, dL_dG):
#     r""" Computes the vector-jacobian product dL_dG @ dG_dA and dL_dG @ dG_db.
#     """
#     M = A.shape[-1]
#     UP = np.zeros(M, dtype=np.int64)
#     LO = np.zeros(M, dtype=np.int64)
#     pivots = lvl_pivots(M, N, PIVOTS)
#     skips = lvl_skips(M, N, SKIPS)
#     for i, pivot in enumerate(pivots):
#         dL_dA, dL_db = consume_one_pivot_vjp(A, b, Aidx, bidx, G, UP, LO, skips[i], pivot, i, dL_dA, dL_db, dL_dG)
#     return dL_dA, dL_db





def fill_all_fold_norm(A, b, C, min_norm=0.99):
    r""" Fills all the amplitudes in G, using the sparse fold
    algorithm while accumulating the norm. It stops when the norm reaches min_norm.

    Args:
        A, b, C: A matrix, b vector and C scalar from the recursive representation
        min_norm: the minimum norm to reach
    """

    # 2. Some constants
    b_nonzero = len(bidx) > 0
    N = 0 if b_nonzero else 1  # if b is zero then odd levels are zero, so we skip building level 1 and jump by two
    M = A.shape[-1]

    # 3. Initialize norm and the dictionary of amplitudes
    norm_squared = np.abs(C)**2
    G = Dict.empty(key_type=typeof((0,)*M), value_type=complex128)
    G[(0,)*M] = C

    # 4. Fill the rest of the amplitudes and accumulate the norm
    while np.sqrt(norm_squared) < min_norm:
        norm_squared += fill_N_plus_one(A, b, G, N, PIVOTS, SKIPS)
        N += 1 if b_nonzero else 2 # go to the next pivots
    return G, norm_squared


# Integration

class BinomialG:
    def __init__(self, G):
        self.multiplets = G
        try:
            self.modes = len(G[1])
        except KeyError:
            self.modes = int((np.sqrt(8*len(G[2]) + 1) - 1) / 2)
        except KeyError:
            raise ValueError("G[1] and G[2] don't exist")
        
    def __getitem__(self, tpl):
        if type(tpl) is int:
            tpl = (tpl,)
        if len(tpl) > self.modes:
            raise IndexError("too may indices")
        N = np.sum(tpl)
        if len(tpl) == self.modes: # a single amplitude
            return self.multiplets[N][index(np.array(tpl))]
        # otherwise we return a new BinomialG with only the relevant modes and amplitudes
        G = dict()
        for n in self.multiplets.keys():#range(N, max(self.multiplets)+1):
            if n > N:
                start = index(np.array(tpl + (0,)*(self.modes-len(tpl)-1) + (n,)))
                end = index(np.array(tpl + (n,) + (0,)*(self.modes-len(tpl)-1)))
                G[n-N] = self.multiplets[n][start:end+1]
        return BinomialG(G)

import tensorflow as tf

@tf.custom_gradient
def G_state(A, b, C, norm=0.99):
    G, norm_squared = fill_all_fold_norm(A, b, C, min_norm=norm, parallel=False)

    def grad(dL_dG):
        dL_dA = np.zeros(A.shape, dtype=np.complex128)
        dL_db = np.zeros(b.shape, dtype=np.complex128)
        dL_dC = np.zeros(1, dtype=np.complex128)
        M = A.shape[-1]
        UP = np.zeros(M, dtype=np.int64)
        LO = np.zeros(M, dtype=np.int64)
        for N in G.keys():
            pivots = lvl_pivots(M, N, PIVOTS)
            skips = lvl_skips(M, N, SKIPS)
            for i, pivot in enumerate(pivots):
                dL_dA, dL_db, dL_dC = consume_one_pivot_vjp(A, b, Aidx, bidx, G, UP, LO, skips[i], pivot, pivot_idx, dL_dA, dL_db, dL_dC, dL_dG)
        return np.conj(dL_dA), np.conj(dL_db), np.conj(dL_dC)

    return G, grad
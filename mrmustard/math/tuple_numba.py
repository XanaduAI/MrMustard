

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

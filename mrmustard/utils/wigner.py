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

"""This module contains the calculation of the Wigner function."""

import numpy as np
from numba import njit

from mrmustard import settings


__all__ = ["wigner_discretized"]

# ~~~~~~~
# Helpers
# ~~~~~~~

@njit
def make_grid(qvec, pvec, hbar):
    Q = np.outer(qvec, np.ones_like(pvec))
    P = np.outer(np.ones_like(qvec), pvec)
    return Q, P, (Q + P * 1.0j) / np.sqrt(2 * hbar)

@njit
def wig_laguerre_val(L, x, c):
    """
    this is evaluation of polynomial series inspired by hermval from numpy.    
    Returns polynomial series
    \sum_n b_n LL_n^L,
    where
    LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]    
    The evaluation uses Clenshaw recursion
    """

    if len(c) == 1:
        y0 = c[0]
        y1 = 0
    elif len(c) == 2:
        y0 = c[0]
        y1 = c[1]
    else:
        k = len(c)
        y0 = c[-2]
        y1 = c[-1]
        for i in range(3, len(c) + 1):
            k -= 1
            y0,    y1 = c[-i] - y1 * (float((k - 1)*(L + k - 1))/((L+k)*k))**0.5, \
            y0 - y1 * ((L + 2*k -1) - x) * ((L+k)*k)**-0.5
            
    return y0 - y1 * ((L + 1) - x) * (L + 1)**-0.5

# ~~~~~~~
# Methods
# ~~~~~~~

def wigner_discretized(rho, qvec, pvec, method="iterative"):
    r"""Calculates the discretized Wigner function for a single mode.

    Adapted from `strawberryfields <https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/states.py#L725>`

    Args:
        rho (complex array): the density matrix of the state in Fock representation
        qvec (array): array of discretized :math:`q` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        hbar (optional float): the value of `\hbar`, defaults to ``settings.HBAR``.

    Returns:
        tuple(array, array, array): array containing the discretized Wigner function, and the Q and
            P coordinates (in meshgrid form) in which the function is calculated
    """
    hbar = settings.HBAR
    if method == "cleanshaw":
        return wigner_discretized_cleanshaw(rho, qvec, pvec, hbar)
    elif method == "iterative":
        return wigner_discretized_iterative(rho, qvec, pvec, hbar)
    
    raise ValueError(f"Method `{method}` not supported. Please select one of"
                     "the supported methods, namely 'cleanshaw' and 'iterative'")

@njit
def wigner_discretized_cleanshaw(rho, qvec, pvec, hbar):
    cutoff = rho.shape[0]
    Q, P, grid = make_grid(qvec, pvec, hbar)
    
    A = 2*grid
    B = np.abs(A)
    B *= B

    w0 = (2*rho[0,-1])*np.ones_like(A)
    
    rho2 = rho * (2*np.ones((cutoff, cutoff)) - np.diag(np.ones(cutoff)))
    L = cutoff - 1 
    while L > 0:
        L -= 1
        #here c_L = _wig_laguerre_val(L, B, np.diag(rho, L))
        w0 = wig_laguerre_val(L, B, np.diag(rho2, L)) + w0 * A * (L+1)**-0.5

    return w0.real * np.exp(-B*0.5) * (hbar*0.5 / np.pi), Q, P

@njit
def wigner_discretized_iterative(rho, qvec, pvec, hbar):
    cutoff = rho.shape[-1]
    Q, P, grid = make_grid(qvec, pvec, hbar)
    Wmat = np.zeros((2, cutoff) + grid.shape, dtype=np.complex128)

    # W = rho(0,0)W(|0><0|)
    Wmat[0, 0] = np.exp(-2.0 * np.abs(grid) ** 2) / np.pi
    W = np.real(rho[0, 0]) * np.real(Wmat[0, 0])

    for n in range(1, cutoff):
        Wmat[0, n] = (2.0 * grid * Wmat[0, n - 1]) / np.sqrt(n)

        # W += rho(0,n)W(|0><n|) + rho(n,0)W(|n><0|)
        W += 2 * np.real(rho[0, n] * Wmat[0, n])

    for m in range(1, cutoff):
        Wmat[1, m] = (2 * np.conj(grid) * Wmat[0, m] - np.sqrt(m) * Wmat[0, m - 1]) / np.sqrt(m)

        # W = rho(m, m)W(|m><m|)
        W += np.real(rho[m, m] * Wmat[1, m])

        for n in range(m + 1, cutoff):
            Wmat[1, n] = (2 * grid * Wmat[1, n - 1] - np.sqrt(m) * Wmat[0, n - 1]) / np.sqrt(n)

            # W += rho(m,n)W(|m><n|) + rho(n,m)W(|n><m|)
            W += 2 * np.real(rho[m, n] * Wmat[1, n])
        Wmat[0] = Wmat[1]

    return W / hbar, Q, P
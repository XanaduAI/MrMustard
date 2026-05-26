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

from mrmustard import math, settings

__all__ = ["wigner_discretized"]


# ~~~~~~~
# Helpers
# ~~~~~~~


@njit(cache=True)
def make_grid(q_vec, p_vec, hbar):  # pragma: no cover
    r"""Returns two coordinate matrices `Q` and `P` from coordinate vectors
    `q_vec` and `p_vec`, along with the grid over which Wigner functions can be
    discretized.
    """
    Q = np.outer(q_vec, np.ones_like(p_vec))
    P = np.outer(np.ones_like(q_vec), p_vec)
    return Q, P, (Q + P * 1.0j) / np.sqrt(2 * hbar)


@njit(cache=True)
def _wig_laguerre_val(L, x, diag):  # pragma: no cover
    """Returns the coefficient `c_L = sum_n rho_{n,L+n} Z_n^L` used
    by `_wigner_discretized_clenshaw`. The evaluation uses the Clenshaw recursion.
    """
    if len(diag) == 2:
        y0 = np.array([[diag[0]]], dtype=np.complex128)
        y1 = np.array([[diag[1]]], dtype=np.complex128)
    else:
        k = len(diag)
        y0 = np.array([[diag[-2]]], dtype=np.complex128)
        y1 = np.array([[diag[-1]]], dtype=np.complex128)
        for i in range(3, len(diag) + 1):
            k -= 1
            temp_y0 = y0
            y0 = diag[-i] - y1 * (float((k - 1) * (L + k - 1)) / ((L + k) * k)) ** 0.5
            y1 = temp_y0 - y1 * ((L + 2 * k - 1) - x) * ((L + k) * k) ** -0.5

    return y0 - y1 * ((L + 1) - x) * (L + 1) ** -0.5


# ~~~~~~~
# Methods
# ~~~~~~~


def wigner_discretized(rho, q_vec, p_vec):
    r"""Calculates the discretized Wigner function for a single mode using Clenshaw recursion.

    Uses Clenshaw summations for the Wigner coefficients :math:`W_{mn}` in
    :math:`W = \sum_{mn} W_{mn} |m\rangle\langle n|`, adapted from the routine in
    QuTiP <http://qutip.org/docs/4.0.2/apidoc/functions.html?highlight=wigner#qutip.wigner.wigner>`_,
    which is released under the BSD license:

    Copyright (C) 2011 and later, P.D. Nation, J.R. Johansson,
    A.J.G. Pitchford, C. Granade, and A.L. Grimsmo. All rights reserved.

    Args:
        rho (complex array): the density matrix of the state in Fock representation
        q_vec (array): array of discretized :math:`q` quadrature values
        p_vec (array): array of discretized :math:`p` quadrature values

    Returns:
        tuple(array, array, array): array containing the discretized Wigner function, and the Q and
            P coordinates (in meshgrid form) in which the function is calculated
    """
    hbar = settings.HBAR
    rho = math.asnumpy(rho)
    return _wigner_discretized_clenshaw(rho, q_vec, p_vec, hbar)


@njit(cache=True)
def _wigner_discretized_clenshaw(rho, q_vec, p_vec, hbar):  # pragma: no cover
    r"""Calculates the Wigner function as a sum of Laguerre polynomials.

    Namely,
    :math:`W = C(x) \sum_L c_L (2x)^L / sqrt(L!)`, where:

    * :math:`x = (q + ip)`, for ``q`` and ``p`` in ``q_vec`` and ``p_vec``
      respectively
    * :math:`C(x) = e^{-x**2/(2\pi)}`
    * :math:`L` is the dimension of ``rho``
    * :math:`c_L = \sum_n \rho_{n,L+n} Z_n^L`
    * :math:`Z_n^L = (-1)^n sqrt(L!n!/(L+n)!) Lag(n,L,x)`
    * :math:`LaguerreL(n,L,x)`
    """
    cutoff = len(rho)
    Q, P, grid = make_grid(q_vec, p_vec, hbar)

    A = 2 * grid
    B = np.abs(A)
    B *= B

    w0 = (2 * rho[0, -1]) * np.ones_like(A)
    rho2 = rho * (2 * np.ones((cutoff, cutoff)) - np.diag(np.ones(cutoff)))

    L = cutoff - 1
    for j in range(1, cutoff):
        c_L = _wig_laguerre_val(L - j, B, np.diag(rho2, L - j))
        w0 = c_L + w0 * A * (L - j + 1) ** -0.5

    return w0.real * np.exp(-B * 0.5) / np.pi / hbar, Q, P

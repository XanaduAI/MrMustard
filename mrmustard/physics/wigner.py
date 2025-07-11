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
    r"""Calculates the discretized Wigner function for a single mode.

    The supported discretization methods are:

    * ``iterative`` (default): Uses an iterative method to calculate the Wigner
    coefficients :math:`W_{mn}` in :math:`W = \sum_{mn} W_{mn} |m\rangle\langle n|`.
    This method is recommended for systems with low numbers of excitations (``n\leq50``).
    * ``clenshaw``: Uses Clenshaw summations to improve the performance for systems
    with large numbers of excitations (``n\leq50``).

    The discretization method can be changed by moodifying the `Settings` object.

    .. code::

        >>> settings.DISCRETIZATION_METHOD  # default method
        "iterative"

        >>> settings.DISCRETIZATION_METHOD = "clenshaw"  # change method

    These methods are adapted versions of the 'iterative' and 'clenshaw' methods of the
    Wigner function discretization routine provided in
    QuTiP <http://qutip.org/docs/4.0.2/apidoc/functions.html?highlight=wigner#qutip.wigner.wigner>`_,
    which is released under the BSD license, with the following copyright notice:

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
    method = settings.DISCRETIZATION_METHOD

    rho = math.asnumpy(rho)
    if method == "iterative":
        return _wigner_discretized_iterative(rho, q_vec, p_vec, hbar)
    return _wigner_discretized_clenshaw(rho, q_vec, p_vec, hbar)


@njit(cache=True)
def _wigner_discretized_clenshaw(rho, q_vec, p_vec, hbar):  # pragma: no cover
    r"""Calculates the Wigner function as
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


@njit(cache=True)
def _wigner_discretized_iterative(rho, q_vec, p_vec, hbar):  # pragma: no cover
    cutoff = len(rho)
    Q, P, grid = make_grid(q_vec, p_vec, hbar)
    Wmat = np.zeros((2, cutoff, *grid.shape), dtype=np.complex128)

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

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


def wigner_discretized(rho, qvec, pvec):
    r"""Calculates the discretized Wigner function for a single mode.

    Adapted from `strawberryfields <https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/states.py#L725>`

    Args:
        rho (complex array): the density matrix of the state in Fock representation
        qvec (array): array of discretized :math:`q` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values

    Retunrs:
        tuple(array, array, array): array containing the discretized Wigner function, and the Q and
            P coordinates (in meshgrid form) in which the function is calculated
    """
    hbar = settings.HBAR
    return _wigner_discretized(rho, qvec, pvec, hbar)

@njit
def _wigner_discretized(rho, qvec, pvec, hbar):
    r""" Calculates the discretized Wigner function for a given value of hbar.
    """
    Q = np.outer(qvec, np.ones_like(pvec))
    P = np.outer(np.ones_like(qvec), pvec)

    cutoff = rho.shape[-1]
    A = (Q + P * 1.0j) / np.sqrt(2 * hbar)

    Wmat = np.zeros((2, cutoff) + A.shape, dtype=np.complex128)

    # Wigner function for |0><0|
    Wmat[0, 0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi
    W = np.real(rho[0, 0]) * np.real(Wmat[0, 0])

    for n in range(1, cutoff):
        Wmat[0, n] = (2.0 * A * Wmat[0, n - 1]) / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wmat[0, n])

    for m in range(1, cutoff):
        # Wigner function for |m><m|
        Wmat[1, m] = (2 * np.conj(A) * Wmat[0, m] - np.sqrt(m) * Wmat[0, m - 1]) / np.sqrt(m)
        W += np.real(rho[m, m] * Wmat[1, m])

        for n in range(m + 1, cutoff):
            # Wigner function for |m><n|
            Wmat[1, n] = (2 * A * Wmat[1, n - 1] - np.sqrt(m) * Wmat[0, n - 1]) / np.sqrt(n)
            W += 2 * np.real(rho[m, n] * Wmat[1, n])
        Wmat[0] = Wmat[1]

    return W / hbar, Q, P

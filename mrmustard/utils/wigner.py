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

from numba import njit
import numpy as np
from mrmustard import settings


@njit
def wigner_discretized(rho, qvec, pvec, hbar=settings.HBAR):
    r"""Calculates the discretized Wigner function

    Adapted from `strawberryfields <https://github.com/XanaduAI/strawberryfields/blob/master/strawberryfields/backends/states.py#L725>`

    Args:
        rho (complex array): the density matrix of the state in Fock representation
        xvec (array): array of discretized :math:`x` quadrature values
        pvec (array): array of discretized :math:`p` quadrature values
        hbar (float): the value of ``\hbar``

    Retunrs:
        tuple(array, array, array): array containing the discretized Wigner function, and the P and
            Q coordinates in meshgrid form
    """

    Q = np.outer(pvec, np.ones_like(qvec))
    P = np.outer(np.ones_like(pvec), qvec)

    cutoff = rho.shape[-1]
    A = (Q + P * 1.0j) / (2 * np.sqrt(hbar / 2))

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

    return W.transpose() / hbar, P, Q

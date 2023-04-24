# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numba import jit


@jit(nopython=True)
def displacement(cutoffs, r, phi, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the matrix elements of the displacement gate using a recurrence relation.
    Uses the log of the matrix elements to avoid numerical issues and then takes the exponential.

    Args:
        r (float): displacement magnitude
        phi (float): displacement angle
        cutoffs (tuple[int, int]): Fock ladder output-input cutoffs
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: matrix representing the displacement operation.
    """
    N, M = cutoffs
    flipped = False
    if N < M:
        N, M = M, N
        flipped = True
    D = np.zeros((N, M), dtype=dtype)
    rng = np.arange(max(*cutoffs))
    rng[0] = 1
    log_k_fac = np.cumsum(np.log(rng))
    for n_minus_m in range(N):
        m_max = min(M, N - n_minus_m)
        logL = np.log(laguerre(r**2.0, m_max, n_minus_m))
        for m in range(m_max):
            n = n_minus_m + m
            sign = 2 * (not (flipped and n > m and n_minus_m % 2)) - 1
            conj = 2 * (not (flipped and n > m)) - 1
            D[n, m] = sign * np.exp(
                +0.5 * (log_k_fac[m] - log_k_fac[n])
                + n_minus_m * np.log(r)
                - (r**2.0) / 2.0
                + conj * 1j * phi * n_minus_m
                + logL[m]
            )
            if n < M:
                D[m, n] = (-1.0) ** n_minus_m * np.conj(D[n, m])
    return D if not flipped else np.transpose(D)


@jit(nopython=True, cache=True)
def laguerre(x, N, alpha, dtype=np.complex128):  # pragma: no cover
    r"""Returns the N first generalized Laguerre polynomials evaluated at x.

    Args:
        x (float): point at which to evaluate the polynomials
        N (int): maximum Laguerre polynomial to calculate
        alpha (float): continuous parameter for the generalized Laguerre polynomials
    """
    L = np.zeros(N, dtype=dtype)
    L[0] = 1.0
    if N > 1:
        for m in range(0, N - 1):
            L[m + 1] = ((2 * m + 1 + alpha - x) * L[m] - (m + alpha) * L[m - 1]) / (m + 1)
    return L


@jit(nopython=True)
def grad_displacement(T, r, phi):  # pragma: no cover
    r"""Calculates the gradients of the displacement gate with respect to the displacement magnitude and angle.

    Args:
        T (array[complex]): array representing the gate
        r (float): displacement magnitude
        phi (float): displacement angle

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the displacement gate with respect to r and phi
    """
    cutoff = T.shape[0]
    dtype = T.dtype
    ei = np.exp(1j * phi)
    eic = np.exp(-1j * phi)
    alpha = r * ei
    alphac = r * eic
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    grad_r = np.zeros((cutoff, cutoff), dtype=dtype)
    grad_phi = np.zeros((cutoff, cutoff), dtype=dtype)

    for m in range(cutoff):
        for n in range(cutoff):
            grad_r[m, n] = -r * T[m, n] + sqrt[m] * ei * T[m - 1, n] - sqrt[n] * eic * T[m, n - 1]
            grad_phi[m, n] = (
                sqrt[m] * 1j * alpha * T[m - 1, n] + sqrt[n] * 1j * alphac * T[m, n - 1]
            )

    return grad_r, grad_phi

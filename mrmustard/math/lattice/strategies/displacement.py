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

"This module contains strategies for calculating the matrix elements of the displacement gate."

import numpy as np
from numba import njit

__all__ = ["displacement", "grad_displacement", "jacobian_displacement", "laguerre"]


@njit(cache=True)
def displacement(cutoffs, alpha, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the matrix elements of the displacement gate using a recurrence relation.
    Uses the log of the matrix elements to avoid numerical issues and then takes the exponential.

    Args:
        cutoffs (tuple[int, int]): Fock ladder output-input cutoffs
        alpha (complex): displacement magnitude and angle
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: matrix representing the displacement operation.
    """
    r = np.abs(alpha)
    phi = np.angle(alpha)
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
                + logL[m],
            )
            if n < M:
                D[m, n] = (-1.0) ** n_minus_m * np.conj(D[n, m])
    return D if not flipped else np.transpose(D)


@njit(cache=True)
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
        for m in range(N - 1):
            L[m + 1] = ((2 * m + 1 + alpha - x) * L[m] - (m + alpha) * L[m - 1]) / (m + 1)
    return L


@njit(cache=True)
def grad_displacement(T, r, phi):  # pragma: no cover
    r"""Calculates the gradient of the displacement gate with respect to the magnitude and angle of the displacement.

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


@njit(cache=True)
def jacobian_displacement(D, alpha):  # pragma: no cover
    r"""Calculates the jacobian of the displacement gate with respect to the complex displacement
    alpha and its conjugate. Both are needed for backprop, as the displacement gate is not a
    holomorphic function, as the Fock amplitudes depend on both alpha and on its conjugate.
    Each jacobian in this case has the same shape as the array D, as the displacement is a scalar.

    Args:
        D (array[complex]): the D(alpha) gate in Fock representation. Batch dimensions are allowed.
        alpha (complex): parameter of D(alpha)

    Returns:
        2 array[complex]: The jacobian of the displacement gate with respect to alpha and alphaconj
    """
    shape = D.shape[-2:]
    alphac = np.conj(alpha)
    sqrt = np.sqrt(np.arange(shape[0] + shape[1], dtype=D.dtype))
    jac_alpha = np.zeros(D.shape, dtype=D.dtype)  # i.e. dD_dalpha for all m,n
    jac_alphac = np.zeros(D.shape, dtype=D.dtype)  # i.e. dD_dalphac for all m,n
    for m in range(shape[0]):
        for n in range(shape[1]):
            jac_alpha[m, n] = -0.5 * alphac * D[m, n] + sqrt[m] * D[m - 1, n]
            jac_alphac[m, n] = -0.5 * alpha * D[m, n] - sqrt[n] * D[m, n - 1]
    return jac_alpha, jac_alphac

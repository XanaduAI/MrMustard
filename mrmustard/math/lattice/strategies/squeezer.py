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

"This module contains strategies for calculating the matrix elements of the squeezing gate."

import numpy as np
from numba import njit

from mrmustard.math.lattice import steps
from mrmustard.utils.typing import ComplexTensor

SQRT = np.sqrt(np.arange(100000))

__all__ = ["squeezed", "squeezed_vjp", "squeezer", "squeezer_vjp"]


@njit(cache=True)
def squeezer(
    shape: tuple[int, int],
    r: float,
    theta: float,
    dtype=np.complex128,
):  # pragma: no cover
    r"""Calculates the matrix elements of the squeezing gate using a recurrence relation.
    (See eq. 50-52 in https://arxiv.org/abs/2004.11002)
        Args:
            shape (tuple[int, int]): Fock cutoffs for the output and input indices.
            r (float): squeezing magnitude
            theta (float): squeezing angle
            dtype (data type): data type used for the calculation.

        Returns:
            array (ComplexMatrix): matrix representing the squeezing gate.
    """
    M, N = shape
    S = np.zeros(shape, dtype=dtype)

    eitheta_tanhr = np.exp(1j * theta) * np.tanh(r)
    eitheta_tanhr_conj = np.conj(eitheta_tanhr)
    sechr = 1.0 / np.cosh(r)

    S[0, 0] = np.sqrt(sechr)
    for m in range(2, M, 2):
        S[m, 0] = -SQRT[m - 1] / SQRT[m] * eitheta_tanhr * S[m - 2, 0]

    for m in range(M):
        for n in range(2 - (m % 2), N, 2):
            # for n in range(1, N):
            if (m + n) % 2 == 0:
                S[m, n] = (
                    SQRT[n - 1] / SQRT[n] * eitheta_tanhr_conj * S[m, n - 2]
                    + SQRT[m] / SQRT[n] * sechr * S[m - 1, n - 1]
                )
    return S


@njit(cache=True)
def squeezer_vjp(
    G: ComplexTensor,
    dLdG: ComplexTensor,
    r: float,
    phi: float,
) -> tuple[float, float]:  # pragma: no cover
    r"""Squeezing gradients with respect to r and theta.
    This function could return dL/dA, dL/db, dL/dc like its vanilla counterpart,
    but it is more efficient to include this chain rule step in the numba function, since we can.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor
        r (float): squeezing magnitude
        phi (float): squeezing angle

    Returns:
        tuple[float, float]: dL/dr, dL/phi
    """
    M, N = G.shape

    # init gradients
    dA = np.zeros((2, 2), dtype=np.complex128)  # dGdA at an index (of G)
    _ = np.zeros(2, dtype=np.complex128)
    dLdA = np.zeros_like(dA)

    # first column
    for m in range(2, M, 2):
        dA, _ = steps.vanilla_step_grad(G, (m, 0), dA, _)
        dLdA += dA * dLdG[m, 0]

    # rest of the matrix
    for m in range(M):
        for n in range(1, N):
            if (m + n) % 2 == 0:
                dA, _ = steps.vanilla_step_grad(G, (m, n), dA, _)
                dLdA += dA * dLdG[m, n]

    dLdC = np.sum(G * dLdG)  # np.sqrt(np.cosh(r)) cancels out with 1 / np.sqrt(np.cosh(r)) later
    # chain rule
    d_sech = -np.tanh(r) / np.cosh(r)
    d_tanh = 1.0 / np.cosh(r) ** 2
    tanh = np.tanh(r)
    exp = np.exp(1j * phi)
    exp_conj = np.exp(-1j * phi)

    dLdr = 2 * np.real(
        -dLdA[0, 0] * exp * d_tanh
        + dLdA[0, 1] * d_sech
        + dLdA[1, 1] * exp_conj * d_tanh
        - np.conj(dLdC) * 0.5 * tanh,  # / np.sqrt(np.cosh(r))
    )
    dLdphi = 2 * np.real(-dLdA[0, 0] * 1j * exp * tanh - dLdA[1, 1] * 1j * exp_conj * tanh)

    return dLdr, dLdphi


@njit(cache=True)
def squeezed(cutoff: int, r: float, theta: float, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the matrix elements of the single-mode squeezed state using recurrence relations.

    Args:
        cutoff (int): Fock cutoff for the ket
        r (float): squeezing magnitude
        theta (float): squeezing angle
        dtype (data type): data type used for the calculation.

    Returns:
        array (ComplexMatrix): matrix representing the squeezing gate.
    """
    S = np.zeros(cutoff, dtype=dtype)
    eitheta_tanhr = np.exp(1j * theta) * np.tanh(r)
    S[0] = np.sqrt(1.0 / np.cosh(r))

    for m in range(2, cutoff, 2):
        S[m] = SQRT[m - 1] / SQRT[m] * eitheta_tanhr * S[m - 2]

    return S


@njit(cache=True)
def squeezed_vjp(
    G: ComplexTensor,
    dLdG: ComplexTensor,
    r: float,
    phi: float,
) -> tuple[float, float]:  # pragma: no cover
    r"""Squeezed state gradients with respect to r and theta.
    This function could return dL/dA, dL/db, dL/dc like its vanilla counterpart,
    but it is more efficient to include this chain rule step in the numba function, since we can.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor
        r (float): squeezing magnitude
        phi (float): squeezing angle

    Returns:
        tuple[float, float]: dL/dr, dL/phi
    """
    M = G.shape[0]

    # init gradients
    dA = np.zeros((1, 1), dtype=np.complex128)
    _ = np.zeros(1, dtype=np.complex128)
    dLdA = np.zeros_like(dA)

    # first column
    for m in range(2, M, 2):
        dA, _ = steps.vanilla_step_grad(G, (m,), dA, _)
        dLdA += dA * dLdG[m]

    # chain rule
    tanh = np.tanh(r)
    d_tanh = 1.0 / np.cosh(r) ** 2
    exp = np.exp(1j * phi)

    dLdC = np.sum(G * dLdG)  # np.sqrt(np.cosh(r)) cancels out with 1 / np.sqrt(np.cosh(r)) later

    dLdr = 2 * np.real(-dLdA[0, 0] * exp * d_tanh - np.conj(dLdC) * 0.5 * tanh)
    dLdphi = 2 * np.real(-dLdA[0, 0] * 1j * exp * tanh)

    return dLdr, dLdphi

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
from numba import njit

from mrmustard.math.lattice import steps
from mrmustard.typing import ComplexTensor


@njit
def squeezer(cutoffs: tuple[int, int], r: float, theta: float, dtype=np.complex128):
    r"""Calculates the matrix elements of the squeezing gate using a recurrence relation.

    Args:
        cutoffs (tuple[int, int]): Fock ladder cutoffs in phase space for the output and input modes.
        r (float): squeezing magnitude
        theta (float): squeezing angle
        dtype (data type): Specifies the data type used for the calculation.

    Returns:
        array (ComplexMatrix): matrix representing the squeezing gate.
    """
    S = np.zeros(cutoffs, dtype=dtype)
    sqrt = np.sqrt(np.arange(max(cutoffs), dtype=dtype))

    eitheta_tanhr = np.exp(1j * theta) * np.tanh(r)
    sechr = 1.0 / np.cosh(r)
    R = np.array(
        [
            [-eitheta_tanhr, sechr],
            [sechr, np.conj(eitheta_tanhr)],
        ]
    )

    S[0, 0] = np.sqrt(sechr)
    for m in range(2, cutoffs[0], 2):
        S[m, 0] = sqrt[m - 1] / sqrt[m] * R[0, 0] * S[m - 2, 0]

    for m in range(0, cutoffs[0]):
        for n in range(1, cutoffs[1]):
            if (m + n) % 2 == 0:
                S[m, n] = (
                    sqrt[n - 1] / sqrt[n] * R[1, 1] * S[m, n - 2]
                    + sqrt[m] / sqrt[n] * R[0, 1] * S[m - 1, n - 1]
                )
    return S


@njit
def squeezer_vjp(
    G: ComplexTensor,
    dLdG: ComplexTensor,
    r: float,
    theta: float,
) -> tuple[float, float]:
    r"""Squeezing gradients with respect to r and theta.
    This function could return dL/dA, dL/db, dL/dc like its vanilla counterpart,
    but it is more efficient to include this chain rule step in the numba function, since we can.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor
        r (float): squeezing magnitude
        theta (float): squeezing angle

    Returns:
        tuple[float, float]: dL/dr, dL/dtheta
    """
    D = G.ndim
    M, N = G.shape

    # init gradients
    dA = np.zeros((D, D), dtype=np.complex128)
    db = np.zeros(D, dtype=np.complex128)
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # rank 2
    for m in range(M):
        for n in range(N):
            if (m + n) % 2 == 0:
                dA, db = steps.vanilla_step_grad(G, (m, n), dA, db)
                dLdA += dA * dLdG[m, n]
                dLdb += db * dLdG[m, n]

    eitheta_tanhr = np.exp(1j * theta) * np.tanh(r)
    sechr = 1.0 / np.cosh(r)
    R = np.array(
        [
            [-eitheta_tanhr, sechr],
            [sechr, np.conj(eitheta_tanhr)],
        ]
    )

    dLdr = 2 * np.real(
        np.conj(-eitheta_tanhr) * dLdA[0, 0] + np.conj(R[1, 1]) * dLdA[1, 1] + R[0, 1] * dLdA[0, 1]
    )
    dLdtheta = 2 * np.real(
        np.conj(R[0, 0]) * dLdA[0, 0] + np.conj(R[1, 1]) * dLdA[1, 1] - R[0, 1] * dLdA[0, 1]
    )

    return dLdr, dLdtheta

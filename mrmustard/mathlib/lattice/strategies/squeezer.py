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
from numba import njit, prange

from mrmustard.mathlib.lattice import steps
from mrmustard.utils.typing import ComplexTensor

__all__ = [
    "squeezed_vjp",
    "squeezed_vjp_batched",
    "squeezer_vjp",
    "squeezer_vjp_batched",
]


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


@njit(parallel=True, cache=True)
def squeezer_vjp_batched(
    G: ComplexTensor,
    dLdG: ComplexTensor,
    r: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    r"""Batched squeezing gradients with respect to r and theta.
    This function processes multiple parameter sets in parallel by flattening
    batch dimensions and using prange for parallel iteration.
    Args:
        G (np.ndarray): Tensor result of the forward pass with shape (*batch, M, N)
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor (*batch, M, N)
        r (np.ndarray): squeezing magnitude with shape (*batch,)
        phi (np.ndarray): squeezing angle with shape (*batch,)
    Returns:
        tuple[np.ndarray, np.ndarray]: dL/dr and dL/dphi with shape (*batch,)
    """
    # Get batch shape and flatten
    batch_shape = G.shape[:-2]
    M, N = G.shape[-2:]
    batch_size = np.prod(np.array(batch_shape))

    # Flatten batch dimensions
    G_flat = G.reshape(batch_size, M, N)
    dLdG_flat = dLdG.reshape(batch_size, M, N)
    r_flat = r.flatten()
    phi_flat = phi.flatten()

    # Allocate output arrays
    dLdr_flat = np.zeros(batch_size, dtype=np.float64)
    dLdphi_flat = np.zeros(batch_size, dtype=np.float64)

    # Parallel iteration over batch
    for batch_idx in prange(batch_size):
        dLdr_flat[batch_idx], dLdphi_flat[batch_idx] = squeezer_vjp(
            G_flat[batch_idx],
            dLdG_flat[batch_idx],
            r_flat[batch_idx],
            phi_flat[batch_idx],
        )

    # Reshape to original batch shape
    dLdr = dLdr_flat.reshape(batch_shape)
    dLdphi = dLdphi_flat.reshape(batch_shape)

    return dLdr, dLdphi


@njit(parallel=True, cache=True)
def squeezed_vjp_batched(
    G: ComplexTensor,
    dLdG: ComplexTensor,
    r: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    r"""Batched squeezed state gradients with respect to r and theta.
    This function processes multiple parameter sets in parallel by flattening
    batch dimensions and using prange for parallel iteration.
    Args:
        G (np.ndarray): Tensor result of the forward pass with shape (*batch, M)
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor (*batch, M)
        r (np.ndarray): squeezing magnitude with shape (*batch,)
        phi (np.ndarray): squeezing angle with shape (*batch,)
    Returns:
        tuple[np.ndarray, np.ndarray]: dL/dr and dL/dphi with shape (*batch,)
    """
    # Get batch shape and flatten
    batch_shape = G.shape[:-1]
    M = G.shape[-1]
    batch_size = np.prod(np.array(batch_shape))

    # Flatten batch dimensions
    G_flat = G.reshape(batch_size, M)
    dLdG_flat = dLdG.reshape(batch_size, M)
    r_flat = r.flatten()
    phi_flat = phi.flatten()

    # Allocate output arrays
    dLdr_flat = np.zeros(batch_size, dtype=np.float64)
    dLdphi_flat = np.zeros(batch_size, dtype=np.float64)

    # Parallel iteration over batch
    for batch_idx in prange(batch_size):
        dLdr_flat[batch_idx], dLdphi_flat[batch_idx] = squeezed_vjp(
            G_flat[batch_idx],
            dLdG_flat[batch_idx],
            r_flat[batch_idx],
            phi_flat[batch_idx],
        )

    # Reshape to original batch shape
    dLdr = dLdr_flat.reshape(batch_shape)
    dLdphi = dLdphi_flat.reshape(batch_shape)

    return dLdr, dLdphi

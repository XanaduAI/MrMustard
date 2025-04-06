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

"""Batched vanilla strategies for Fock representation calculation."""

import numpy as np
from numba import njit, prange

from mrmustard.math.lattice import steps
from mrmustard.utils.typing import ComplexTensor

from .core import vanilla, stable


@njit(parallel=True)
def stable_b_batch(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Batched version of the stable vanilla algorithm for calculating the fock representation of a Gaussian tensor.
    See the documentation of ``stable`` for more details about the non-batched version.

    Args:
        shape (tuple[int, ...]): shape of the output tensor excluding the batch dimension, which is inferred from ``b``
        A (np.ndarray): A matrix of the Bargmann representation
        b (np.ndarray): batched b vector of the Bargmann representation with batch on the first dimension
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``(batch,) + shape``
    """

    B = b.shape[0]
    G = np.zeros((B,) + shape, dtype=np.complex128)
    for k in prange(B):
        G[k] = stable(shape, A, b[k], c)
    return G


@njit
def vanilla_b_batch(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy for batched ``b``, with batched dimension on the
    first index.

    Fills the tensor by iterating over all indices in the order given by ``np.ndindex``.

    Args:
        shape (tuple[int, ...]): shape of the output tensor without the batch dimension
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): batched B vector of the Fock-Bargmann representation, the batch dimension is on the first index
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    batch_shape = (b.shape[0],)
    G = np.zeros(batch_shape + shape, dtype=np.complex128)
    path = np.ndindex(shape)
    G[(slice(None),) + next(path)] = c
    for index in path:
        G[(slice(None),) + index] = steps.vanilla_step_batch(G, A, b, index)
    return G


@njit(parallel=True)
def vanilla_full_batch(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Batched version of the vanilla algorithm for calculating the fock representation of a
    Gaussian tensor. This implementation assumes that the batch dimension is on the first
    axis of A, b, and c and it's the same for all of them.
    See the documentation of ``vanilla`` for more details about the non-batched version.

    Args:
        shape (tuple[int, ...]): shape of the output tensor excluding the batch dimension
        A (np.ndarray): batched A matrix of the Bargmann representation
        b (np.ndarray): batched b vector of the Bargmann representation
        c (complex): batched vacuum amplitudes

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``(batch,) + shape``
    """
    batch_size = b.shape[0]
    G = np.zeros((batch_size,) + shape, dtype=np.complex128)
    for k in prange(batch_size):
        G[k] = vanilla(shape, A[k], b[k], c[k])
    return G


@njit(parallel=True)
def stable_full_batch(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Batched version of the stable vanilla algorithm for calculating the fock representation of a
    Gaussian tensor. This implementation assumes that the batch dimension is on the first
    axis of A, b, and c and it's the same for all of them.
    See the documentation ``stable`` for more details about the non-batched version.

    Args:
        shape (tuple[int, ...]): shape of the output tensor excluding the batch dimension
        A (np.ndarray): batched A matrix of the Bargmann representation
        b (np.ndarray): batched b vector of the Bargmann representation
        c (complex): batched vacuum amplitudes

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``(batch,) + shape``
    """
    batch_size = b.shape[0]
    G = np.zeros((batch_size,) + shape, dtype=np.complex128)
    for k in prange(batch_size):
        G[k] = stable(shape, A[k], b[k], c[k])
    return G

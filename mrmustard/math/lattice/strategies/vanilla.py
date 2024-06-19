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

from mrmustard.math.lattice import paths, steps
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector
from .flat_indices import first_available_pivot, lower_neighbors, shape_to_strides

__all__ = ["vanilla", "vanilla_batch", "vanilla_jacobian", "vanilla_vjp"]

SQRT = np.sqrt(np.arange(1000))


@njit
def vanilla(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy.

    Flattens the tensors, then fills it by iterating over all indices in the order
    given by ``np.ndindex``. Finally, it reshapes the tensor before returning.

    Args:
        shape (tuple[int, ...]): shape of the output tensor
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    # calculate the strides
    strides = shape_to_strides(np.array(shape))

    # init flat output tensor
    ret = np.array([0 + 0j] * np.prod(np.array(shape)))

    # initialize the indeces.
    # ``index`` is the index of the flattened output tensor, while
    # ``index_u_iter`` iterates through the unravelled counterparts of
    # ``index``.
    index = 0
    index_u_iter = np.ndindex(shape)
    next(index_u_iter)

    # write vacuum amplitude /
    ret[0] = c

    # iterate over the rest of the indices
    for index_u in index_u_iter:
        # update index
        index += 1

        # calculate pivot's contribution
        i, pivot = first_available_pivot(index, strides)
        value_at_index = b[i] * ret[pivot]

        # add the contribution of pivot's lower's neighbours
        ns = lower_neighbors(pivot, strides, i)
        (j0, n0) = next(ns)
        value_at_index += A[i, j0] * np.sqrt(index_u[j0] - 1) * ret[n0]
        for j, n in ns:
            value_at_index += A[i, j] * np.sqrt(index_u[j]) * ret[n]
        ret[index] = value_at_index / np.sqrt(index_u[i])

    return ret.reshape(shape)


@njit
def vanilla_batch(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy for batched ``b``, with batched dimension on the
    last index.

    Fills the tensor by iterating over all indices in the order given by ``np.ndindex``.

    Args:
        shape (tuple[int, ...]): shape of the output tensor with the batch dimension on the last term
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): batched B vector of the Fock-Bargmann representation, the batch dimension is on the last index
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """

    # init output tensor
    G = np.zeros(shape, dtype=np.complex128)

    # initialize path iterator
    path = np.ndindex(shape[:-1])  # We know the last dimension is the batch one

    # write vacuum amplitude
    G[next(path)] = c

    # iterate over the rest of the indices
    for index in path:
        G[index] = steps.vanilla_step_batch(G, A, b, index)

    return G


@njit
def vanilla_jacobian(
    G, A, b, c
) -> tuple[ComplexTensor, ComplexTensor, ComplexTensor]:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy gradient. Returns dG/dA, dG/db, dG/dc.
    Notice that G is a holomorphic function of A, b, c. This means that there is only
    one gradient to care about for each parameter (i.e. not dG/dA.conj() etc).
    """

    # init output tensors
    dGdA = np.zeros(G.shape + A.shape, dtype=np.complex128)
    dGdb = np.zeros(G.shape + b.shape, dtype=np.complex128)
    dGdc = G / c

    # initialize path iterator
    path = paths.ndindex_path(G.shape)

    # skip first index
    next(path)

    # iterate over the rest of the indices
    for index in path:
        dGdA, dGdb = steps.vanilla_step_jacobian(G, A, b, index, dGdA, dGdb)

    return dGdA, dGdb, dGdc


@njit
def vanilla_vjp(
    G, c, dLdG
) -> tuple[ComplexMatrix, ComplexVector, complex]:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy gradient. Returns dL/dA, dL/db, dL/dc.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        c (complex): vacuum amplitude
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor

    Returns:
        tuple[np.ndarray, np.ndarray, complex]: dL/dA, dL/db, dL/dc
    """
    shape = G.shape

    # calculate the strides
    strides = shape_to_strides(np.array(shape))

    # linearize G
    G_lin = G.flatten()

    # init gradients
    D = G.ndim
    dA = np.zeros((D, D), dtype=np.complex128)  # component of dL/dA
    db = np.zeros(D, dtype=np.complex128)  # component of dL/db
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # initialize the indices.
    # ``index`` is the index of the flattened output tensor, while
    # ``index_u_iter`` iterates through the unravelled counterparts of
    # ``index``.
    index = 0
    index_u_iter = np.ndindex(shape)
    next(index_u_iter)

    for index_u in index_u_iter:
        index += 1

        ns = lower_neighbors(index, strides, 0)

        for i, _ in enumerate(db):
            _, n = next(ns)
            db[i] = np.sqrt(index_u[i]) * G_lin[n]
            dA[i, i] = (
                0.5 * np.sqrt(index_u[i] * (index_u[i] - 1)) * G_lin[n - strides[i]]
            )
            for j in range(i + 1, len(db)):
                dA[i, j] = np.sqrt(index_u[i] * index_u[j]) * G_lin[n - strides[j]]

        dLdA += dA * dLdG[index_u]
        dLdb += db * dLdG[index_u]

    dLdc = np.sum(G_lin.reshape(shape) * dLdG) / c

    return dLdA, dLdb, dLdc


@njit
def autoshape(A, b, c, max_prob, max_shape=100) -> int:
    r"""Strategy to compute the shape of the Fock represenation of a Gaussian DM
    such that its trace is above a certain bound given as ``max_prob``.
    This is effectively a 1-mode adaptation of Robbe's diagonal strategy, with early stopping.

    Args:
        A (np.ndarray): A 1x1 matrix of the Fock-Bargmann representation
        b (np.ndarray): B 1-dim vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude
        max_prob (float): the probability value to stop at
        max_shape (int): stop if loop runs beyond this value

    Details:

    Here's how it works. We maintain two buffers: buf3 and buf2, of size 2x3 and 2x2 respectively.
    The buffers contain the following elements of the density matrix:

                     ┌────┐                           ┌────┐
                     │buf3│                           │buf2│
                     └────┘                           └────┘
            ┌────┬────┬────┬────┬────┐      ┌────┬────┬────┬────┬────┐
            │  c │    │ b3 │    │    │      │    │ b2 │    │    │    │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │    │ b3 │    │ b3 │    │      │ b2 │    │ b2 │    │    │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │ b3 │    │ b3 │    │ b3 │      │    │ b2 │    │ b2 │    │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │    │ b3 │    │ b3 │    │      │    │    │ b2 │    │ b2 │
            ├────┼────┼────┼────┼────┤      ├────┼────┼────┼────┼────┤
            │    │    │ b3 │    │etc │      │    │    │    │ b2 │etc │
            └────┴────┴────┴────┴────┘      └────┴────┴────┴────┴────┘

    Note that the buffers don't have shape (n,3) and (n,2) because they only need to keep
    two consecutive groups of elements, because the recursion only needs the previous two groups.
    By using indices mod 2, the data needed at each iteration is in the columns not being updated.

    The updates roughly look like this:

                    ┌───────┐                     ┌───────┐
                    │k even │                     │ k odd │
                    └───────┘                     └───────┘
                 ┌──────┬──────┐               ┌──────┬──────┐
                 │  A───▶      │               │      ◀───A  │
        ┌────┐   │      │      │               │      │      │
        │buf2│   ├──────┼──────┤               ├──────┼──────┤
        └────┘   │  A───▶      │               │      ◀───A  │
                 │      │      │               │      │      │
                 └───┬──┴───▲──┘               └───▲──┴───┬──┘
                     └──b──┐│                     ┌┼──b───┘
                           ││                     ││
                     ┌──b──┼┘                     │└──b───┐
                 ┌───┴──┬──▼───┐               ┌──▼───┬───┴──┐
        ┌────┐   │  A───▶      │               │      ◀───A  │
        │buf3│   │      │      │               │      │      │
        └────┘   ├──────┼──────┤               ├──────┼──────┤
                 │  A───▶      │               │      ◀───A  │
                 │      │      │               │      │      │
                 ├──────┼──────┤               ├──────┼──────┤
                 │  A───▶      │               │      ◀───A  │
                 │      │      │               │      │      │
                 └──────┴──────┘               └──────┴──────┘

    A and b mean that the contribution is multiplied by some element of A or b.
    There are also diagonal A-arrows between the columns of the same buffer, but they are not shown here.
    For pivot in (k,k) buf2 is updated, for pivots in (k+1,k) and (k,k+1) buf3 is updated.
    """
    buf2 = np.zeros(
        (2, 2), dtype=np.complex128
    )  # this is transposed with respect to the diagram
    buf3 = np.zeros(
        (2, 3), dtype=np.complex128
    )  # this is transposed with respect to the diagram
    buf3[0, 1] = c  # vacuum probability at (0,0)
    norm = np.abs(c)
    k = 0
    while norm < max_norm and k < max_shape:
        # pivot at (k, k), write (k+1, k) and (k, k+1)
        buf2[(k + 1) % 2] = (b * buf3[k % 2, 1] + A @ buf2[k % 2] * SQRT[k]) / SQRT[
            k + 1
        ]
        # pivot at (k+1, k), write (k+2, k) and (k+1, k+1)
        buf3[(k + 1) % 2, :2] = (
            b * buf2[(k + 1) % 2, 0]
            + A @ (buf3[k % 2, :2] * np.array([SQRT[k + 1], SQRT[k]]))
        ) / np.array([SQRT[k + 2], SQRT[k + 1]])
        # pivot at (k, k+1), write (k, k+2) only
        buf3[(k + 1) % 2, 2] = (
            b[1] * buf2[(k + 1) % 2, 1]
            + A[1] @ (buf3[k % 2, 1:] * np.array([SQRT[k], SQRT[k + 1]]))
        ) / SQRT[k + 2]
        norm += np.abs(buf3[(k + 1) % 2, 1])
        k += 1
    return k

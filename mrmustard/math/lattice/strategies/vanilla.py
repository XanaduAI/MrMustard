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
from numba import njit, prange

from mrmustard.math.lattice import paths, steps
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector

__all__ = [
    "vanilla",
    "vanilla_stable",
    "vanilla_stable_batch",
    "vanilla_batch",
    "vanilla_jacobian",
    "vanilla_vjp",
    "autoshape_numba",
]

SQRT = np.sqrt(np.arange(1000))


@njit
def vanilla(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Vanilla algorithm for calculating the fock representation of a Gaussian tensor.
    This implementation works on flattened tensors and reshapes the tensor before returning.

    The vanilla algorithm implements the flattened version of the following recursion which
    calculates the Fock amplitude at index :math:`k` using a pivot at index :math:`k - 1_i`
    and its neighbours at indices :math:`k - 1_i - 1_j`:

    .. math::

         G_{k} = \frac{1}{\sqrt{k_i}} \left[b_{i} G_{k-1_i} + \sum_j A_{ij} \sqrt{k_j - \delta_{ij}} G_{k-1_i-1_j} \right]

    where :math:`1_i` is the vector of zeros with a 1 at index :math:`i`, and :math:`\delta_{ij}` is the Kronecker delta.
    In this formula :math:`k` is the vector of indices indexing into the Fock lattice.
    In the implementation the indices are flattened into a single integer index.
    This simplifies the bounds check when calculating the index of the pivot :math:`k-1_i`,
    which need to be done only until the index is smaller than the maximum stride.

    see https://quantum-journal.org/papers/q-2020-11-30-366/ and
    https://arxiv.org/abs/2209.06069 for more details.

    Args:
        shape (tuple[int, ...]): shape of the output tensor
        A (np.ndarray): A matrix of the Bargmann representation
        b (np.ndarray): b vector of the Bargmann representation
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    # numba doesn't like tuples
    shape_arr = np.array(shape)
    D = b.shape[0]

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(D - 1, 0, -1):
        strides[i - 1] = strides[i] * shape_arr[i]

    # init flat output tensor
    G = np.zeros(np.prod(shape_arr), dtype=np.complex128)

    # initialize the n-dim index
    nd_index = np.ndindex(shape)

    # write vacuum amplitude and skip corresponding n-dim index
    G[0] = c
    next(nd_index)

    # Iterate over the indices smaller than max(strides) with pivot bound check.
    # The check is needed only if the flat index is smaller than the largest stride.
    # Afterwards it will be safe to get the pivot by subtracting the first (largest) stride.
    for flat_index in range(1, strides[0]):
        index = next(nd_index)

        # calculate (flat) pivot
        for i, s in enumerate(strides):
            pivot = flat_index - s
            if pivot >= 0:  # if pivot not outside array
                break

        # contribution from pivot
        value_at_index = b[i] * G[pivot]

        # contributions from pivot's lower neighbours
        # note the first is when j=i which needs a -1 in the sqrt from delta_ij
        value_at_index += A[i, i] * SQRT[index[i] - 1] * G[pivot - strides[i]]
        for j in range(i + 1, D):
            value_at_index += A[i, j] * SQRT[index[j]] * G[pivot - strides[j]]
        G[flat_index] = value_at_index / SQRT[index[i]]

    # Iterate over the rest of the indices.
    # Now i can always be 0 (largest stride), and we don't need bounds check
    for flat_index in range(strides[0], len(G)):
        index = next(nd_index)

        # pivot can be calculated without bounds check
        pivot = flat_index - strides[0]

        # contribution from pivot
        value_at_index = b[0] * G[pivot]

        # contribution from pivot's lower neighbours
        # note the first is when j=0 which needs a -1 in the sqrt from delta_0j
        value_at_index += A[0, 0] * SQRT[index[0] - 1] * G[pivot - strides[0]]
        for j in range(1, D):
            value_at_index += A[0, j] * SQRT[index[j]] * G[pivot - strides[j]]
        G[flat_index] = value_at_index / SQRT[index[0]]

    return G.reshape(shape)


@njit
def vanilla_stable(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Stable version of the vanilla algorithm for calculating the fock representation of a Gaussian tensor.
    This implementation works on flattened tensors and reshapes the tensor before returning.

    The vanilla algorithm implements the flattened version of the following recursion which
    calculates the Fock amplitude at index :math:`k` using all available pivots at index :math:`k - 1_i`
    for all :math:`i`, and their neighbours at indices :math:`k - 1_i - 1_j`:

    .. math::

         G_{k} = \frac{1}{N}\sum_i\frac{1}{\sqrt{k_i}} \left[b_{i} G_{k-1_i} + \sum_j A_{ij} \sqrt{k_j - \delta_{ij}} G_{k-1_i-1_j} \right]

    where :math:`N` is the number of valid pivots for the :math:`k`-th lattice point, :math:`1_i` is
    the vector of zeros with a 1 at index :math:`i`, and :math:`\delta_{ij}` is the Kronecker delta.
    In the implementation the indices are flattened into a single integer index, which makes the
    computations of the pivots more efficient.

    see https://quantum-journal.org/papers/q-2020-11-30-366/ and
    https://arxiv.org/abs/2209.06069 for more details.

    Args:
        shape (tuple[int, ...]): shape of the output tensor
        A (np.ndarray): A matrix of the Bargmann representation
        b (np.ndarray): b vector of the Bargmann representation
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    shape_arr = np.array(shape)
    D = b.shape[0]

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(D - 1, 0, -1):
        strides[i - 1] = strides[i] * shape[i]

    # initialize flat output array and write vacuum amplitude
    G = np.zeros(np.prod(shape_arr), dtype=np.complex128)
    G[0] = c

    # initialize flat index and n-dim iterator
    idx = 0
    nd_iterator = np.ndindex(shape)
    next(nd_iterator)  # skip (0,0,...,0) as we already wrote it

    for nd_idx in nd_iterator:
        idx += 1
        num_pivots = 0
        vals = 0
        for i in range(D):
            if nd_idx[i] == 0:
                continue  # pivot would be out of bounds
            num_pivots += 1
            pivot = idx - strides[i]

            # contribution from i-th pivot
            val = b[i] * G[pivot]

            # contributions from lower neighbours of the pivot
            # note we split lower neighbours in 3 parts (j<i, j==i, i<j)
            # so that delta_ij is just 1 for j==i, and 0 elsewhere.
            for j in range(i):
                val += A[i, j] * SQRT[nd_idx[j]] * G[pivot - strides[j]]

            val += A[i, i] * SQRT[nd_idx[i] - 1] * G[pivot - strides[i]]

            for j in range(i + 1, D):
                val += A[i, j] * SQRT[nd_idx[j]] * G[pivot - strides[j]]

            # accumulate the contribution from the i-th pivot
            vals += val / SQRT[nd_idx[i]]

        # write the average
        G[idx] = vals / num_pivots

    return G.reshape(shape)


@njit(parallel=True)
def vanilla_stable_batch(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    """Batched version of the stable vanilla algorithm for calculating the fock representation of a Gaussian tensor.
    See the documentation of ``vanilla_stable`` for more details about the non-batched version.

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
        G[k] = vanilla_stable(shape, A, b[k], c)
    return G


@njit
def vanilla_batch(shape: tuple[int, ...], A, b, c) -> ComplexTensor:  # pragma: no cover
    r"""Vanilla Fock-Bargmann strategy for batched ``b``, with batched dimension on the
    first index.

    Fills the tensor by iterating over all indices in the order given by ``np.ndindex``.

    Args:
        shape (tuple[int, ...]): shape of the output tensor with the batch dimension on the first term
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): batched B vector of the Fock-Bargmann representation, the batch dimension is on the first index
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """

    # init output tensor
    G = np.zeros(shape, dtype=np.complex128)

    # initialize path iterator
    path = np.ndindex(shape[1:])  # We know the first dimension is the batch one

    # write vacuum amplitude
    G[(slice(None),) + next(path)] = c

    # iterate over the rest of the indices
    for index in path:
        G[(slice(None),) + index] = steps.vanilla_step_batch(G, A, b, index)

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
def vanilla_vjp(G, c, dLdG) -> tuple[ComplexMatrix, ComplexVector, complex]:  # pragma: no cover
    r"""Vanilla vjp function. Returns dL/dA, dL/db, dL/dc.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        c (complex): vacuum amplitude
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor

    Returns:
        tuple[np.ndarray, np.ndarray, complex]: dL/dA, dL/db, dL/dc
    """
    # numba doesn't like tuples
    shape_arr = np.array(G.shape)

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(len(shape_arr) - 1, 0, -1):
        strides[i - 1] = strides[i] * shape_arr[i]

    # linearize G
    G_lin = G.flatten()

    # init gradients
    D = len(shape_arr)
    dA = np.zeros((D, D), dtype=np.complex128)  # component of dL/dA
    db = np.zeros(D, dtype=np.complex128)  # component of dL/db
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # initialize the n-dim index
    flat_index = 0
    nd_index = np.ndindex(G.shape)
    next(nd_index)

    # numba doesn't like tuples
    shape_arr = np.array(G.shape)

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(len(shape_arr) - 1, 0, -1):
        strides[i - 1] = strides[i] * shape_arr[i]

    # iterate over the indices (no need to split the loop in two parts)
    for index_u in nd_index:
        flat_index += 1

        # contributions from lower neighbours
        for i in range(D):
            pivot = flat_index - strides[i]
            db[i] = SQRT[index_u[i]] * G_lin[pivot]
            dA[i, i] = 0.5 * SQRT[index_u[i]] * SQRT[index_u[i] - 1] * G_lin[pivot - strides[i]]
            for j in range(i + 1, D):
                dA[i, j] = SQRT[index_u[i]] * SQRT[index_u[j]] * G_lin[pivot - strides[j]]

        dLdA += dA * dLdG[index_u]
        dLdb += db * dLdG[index_u]

    dLdc = np.sum(G * dLdG) / c

    return dLdA, dLdb, dLdc


@njit
def autoshape_numba(A, b, c, max_prob, max_shape) -> int:  # pragma: no cover
    r"""Strategy to compute the shape of the Fock representation of a Gaussian DM
    such that its trace is above a certain bound given as ``max_prob``.
    This is an adaptation of Robbe's diagonal strategy, with early stopping.
    Details in https://quantum-journal.org/papers/q-2023-08-29-1097/.

    Args:
        A (np.ndarray): 2Mx2M matrix of the Bargmann ansatz
        b (np.ndarray): 2M-dim vector of the Bargmann ansatz
        c (float): vacuum amplitude
        max_prob (float): the probability value to stop at (default 0.999)
        max_shape (int): max value before stopping (default 100)

    **Details:**

    Here's how it works. First we get the reduced density matrix at the given mode. Then we
    maintain two buffers that contain values around the diagonal: buf3 and buf2, of size 2x3
    and 2x2 respectively. The buffers contain the following elements of the density matrix:

    .. code-block::

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

    .. code-block::

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

    The rules for updating are in https://quantum-journal.org/papers/q-2023-08-29-1097/
    """
    # reduced DMs
    M = len(b) // 2
    shape = np.ones(M, dtype=np.int64)
    A = A.reshape((2, M, 2, M)).transpose((1, 3, 0, 2))  # (M,M,2,2)
    b = b.reshape((2, M)).transpose()  # (M,2)
    zero = np.zeros((M - 1, M - 1), dtype=np.complex128)
    id = np.eye(M - 1, dtype=np.complex128)
    X = np.vstack((np.hstack((zero, id)), np.hstack((id, zero))))
    for m in range(M):
        idx_m = np.array([m])
        idx_n = np.delete(np.arange(M), m)
        A_mm = np.ascontiguousarray(A[idx_m, :][:, idx_m].transpose((2, 0, 3, 1))).reshape((2, 2))
        A_nn = np.ascontiguousarray(A[idx_n, :][:, idx_n].transpose((2, 0, 3, 1))).reshape(
            (2 * M - 2, 2 * M - 2)
        )
        A_mn = np.ascontiguousarray(A[idx_m, :][:, idx_n].transpose((2, 0, 3, 1))).reshape(
            (2, 2 * M - 2)
        )
        A_nm = np.transpose(A_mn)
        b_m = np.ascontiguousarray(b[idx_m].transpose()).reshape((2,))
        b_n = np.ascontiguousarray(b[idx_n].transpose()).reshape((2 * M - 2,))
        # single-mode A,b,c
        A_ = A_mm - A_mn @ np.linalg.inv(A_nn - X) @ A_nm
        b_ = b_m - A_mn @ np.linalg.inv(A_nn - X) @ b_n
        c_ = (
            c
            * np.exp(-0.5 * b_n @ np.linalg.inv(A_nn - X) @ b_n)
            / np.sqrt(np.linalg.det(A_nn - X))
        )
        # buffers are transposed with respect to the diagram
        buf2 = np.zeros((2, 2), dtype=np.complex128)
        buf3 = np.zeros((2, 3), dtype=np.complex128)
        buf3[0, 1] = c_  # vacuum probability at (0,0)
        norm = np.abs(c_)
        k = 0
        while norm < max_prob and k < max_shape:
            buf2[(k + 1) % 2] = (b_ * buf3[k % 2, 1] + A_ @ buf2[k % 2] * SQRT[k]) / SQRT[k + 1]
            buf3[(k + 1) % 2, 0] = (
                b_[0] * buf2[(k + 1) % 2, 0]
                + A_[0, 0] * buf3[k % 2, 1] * SQRT[k + 1]
                + A_[0, 1] * buf3[k % 2, 0] * SQRT[k]
            ) / SQRT[k + 2]
            buf3[(k + 1) % 2, 1] = (
                b_[1] * buf2[(k + 1) % 2, 0]
                + A_[1, 0] * buf3[k % 2, 1] * SQRT[k + 1]
                + A_[1, 1] * buf3[k % 2, 0] * SQRT[k]
            ) / SQRT[k + 1]
            buf3[(k + 1) % 2, 2] = (
                b_[1] * buf2[(k + 1) % 2, 1]
                + A_[1, 0] * buf3[k % 2, 2] * SQRT[k]
                + A_[1, 1] * buf3[k % 2, 1] * SQRT[k + 1]
            ) / SQRT[k + 2]
            norm += np.abs(buf3[(k + 1) % 2, 1])
            k += 1
        shape[m] = k or 1
    return shape

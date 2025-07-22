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

"""Core vanilla strategies for Fock representation calculation."""

import numpy as np
from numba import njit

from mrmustard.utils.typing import ComplexTensor

SQRT = np.sqrt(np.arange(100000))


@njit(cache=True)
def vanilla_numba(
    shape: tuple[int, ...],
    A,
    b,
    c,
    out: ComplexTensor | None = None,
) -> ComplexTensor:  # pragma: no cover
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
        out (np.ndarray): if provided, the result will be stored in this tensor.

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    # numba doesn't like tuples
    shape_arr = np.array(shape)
    D = b.shape[-1]

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(D - 1, 0, -1):
        strides[i - 1] = strides[i] * shape_arr[i]

    # init flat output tensor
    G = out.ravel() if out is not None else np.zeros(np.prod(shape_arr), dtype=np.complex128)

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

        i = 0
        # calculate (flat) pivot
        for s in strides:
            pivot = flat_index - s
            if pivot >= 0:  # if pivot not outside array
                break
            i += 1

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


@njit(cache=True)
def stable_numba(
    shape: tuple[int, ...],
    A,
    b,
    c,
    out: ComplexTensor | None = None,
) -> ComplexTensor:  # pragma: no cover
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
        out (np.ndarray): if provided, the result will be stored in this tensor.

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    shape_arr = np.array(shape)
    D = b.shape[-1]

    # calculate the strides (e.g. (100,10,1) for shape (10,10,10))
    strides = np.ones_like(shape_arr)
    for i in range(D - 1, 0, -1):
        strides[i - 1] = strides[i] * shape[i]

    # initialize flat output tensor
    G = out.ravel() if out is not None else np.zeros(np.prod(shape_arr), dtype=np.complex128)

    # initialize flat index and n-dim iterator
    flat_index = 0
    nd_index = np.ndindex(shape)

    # write vacuum amplitude
    G[flat_index] = c
    next(nd_index)

    for nd_idx in nd_index:
        flat_index += 1
        num_pivots = 0
        vals = 0
        for i in range(D):
            if nd_idx[i] == 0:
                continue  # pivot would be out of bounds
            num_pivots += 1
            pivot = flat_index - strides[i]

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
        G[flat_index] = vals / num_pivots

    return G.reshape(shape)

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

from mrmustard.utils.typing import ComplexTensor

from .core import stable_numba, vanilla_numba

# ruff: noqa: RUF005


@njit(cache=True, parallel=True)
def vanilla_batch_numba(
    shape: tuple[int, ...],
    A,
    b,
    c,
    stable: bool = False,
    out: ComplexTensor | None = None,
) -> ComplexTensor:  # pragma: no cover
    r"""Batched version of the vanilla algorithm for calculating the fock representation of a
    Gaussian tensor. This implementation assumes that the batch dimension is on the first
    axis of A, b, and c and it's the same for all of them.
    It can use either the standard or the stable algorithm based on the `stable` flag.
    Default is to use the standard algorithm.
    See the documentation of ``vanilla_numba`` and ``stable_numba`` for more details.

    Args:
        shape (tuple[int, ...]): shape of the output tensor excluding the batch dimension
        A (np.ndarray): batched A matrix of the Bargmann representation
        b (np.ndarray): batched b vector of the Bargmann representation
        c (complex): batched vacuum amplitudes
        stable (bool): if ``True``, use the stable algorithm, otherwise use the standard one
        out (np.ndarray): if provided, the result will be stored in this tensor.

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``(batch,) + shape``
    """
    batch_size = b.shape[0]
    G = out if out is not None else np.zeros((batch_size,) + shape, dtype=np.complex128)
    for k in prange(batch_size):
        if stable:
            G[k] = stable_numba(shape, A[k], b[k], c[k])
        else:
            G[k] = vanilla_numba(shape, A[k], b[k], c[k])
    return G

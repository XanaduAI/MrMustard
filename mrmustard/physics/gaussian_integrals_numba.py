# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains gaussian integral functions and related helper functions.
"""

import numpy as np
from numba import njit

from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector


@njit(cache=True)
def real_gaussian_integral_numba(
    A: ComplexMatrix,
    b: ComplexVector,
    c: ComplexTensor,
    idx: tuple[int, ...],
):
    r"""Computes the Gaussian integral of the exponential of a real quadratic form.
    The integral is defined as (note that in general we integrate over a subset of `m` dimensions):

    .. :math::
        \int_{R^m} F(x) dx,

    where

    :math:`F(x) = \textrm{exp}(0.5 x^T A x + b^T x)`

    Here, ``x`` is an ``n``-dim real vector, ``A`` is an ``n x n`` real matrix,
    ``b`` is an ``n``-dim real vector, ``c`` is a real scalar. The integral indices
    are specified by ``idx``.

    Arguments:
        Abc: the ``(A,b,c)`` triple
        idx: the tuple of indices of the x variables

    Returns:
        The ``(A,b,c)`` triple of the result of the integral.
    """
    idx = np.array(idx, dtype=np.int64)
    if len(idx) == 0:
        return A, b, c
    not_idx = np.array([i for i in range(A.shape[-1]) if i not in idx])

    M = A[..., :, idx][..., idx, :]
    bM = b[..., idx]

    if not_idx.shape != (0,):
        D = A[..., idx][..., not_idx, :]
        R = A[..., not_idx][..., not_idx, :]
        bR = b[..., not_idx]
        T = np.transpose
        L = T(np.linalg.solve(T(M), T(D)))
        A_post = R - (L @ T(D))
        b_post = bR - (L @ bM)
    else:
        A_post = np.empty((0, 0), dtype=np.complex128)
        b_post = np.empty((0,), dtype=np.complex128)

    c_post = np.array(
        c
        * np.sqrt((2 * np.pi) ** len(idx))
        * np.sqrt((-1) ** len(idx) / np.linalg.det(M))
        * np.exp(-0.5 * np.sum(bM * np.linalg.solve(M, bM)))
    )

    return A_post, b_post, c_post

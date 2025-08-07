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

from mrmustard.physics.utils import outer_product_batch_str, verify_batch_triple
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


def join_Abc_numba(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    batch_string: str | None = None,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2

    verify_batch_triple(A1, b1, c1)
    verify_batch_triple(A2, b2, c2)

    batch1, batch2 = A1.shape[:-2], A2.shape[:-2]

    if batch_string is None:
        batch_string = outer_product_batch_str(len(batch1), len(batch2))

    A1 = np.array(A1, dtype=np.complex128)
    A2 = np.array(A2, dtype=np.complex128)
    b1 = np.array(b1, dtype=np.complex128)
    b2 = np.array(b2, dtype=np.complex128)
    c1 = np.array(c1, dtype=np.complex128)
    c2 = np.array(c2, dtype=np.complex128)
    return _join_Abc_numba((A1, b1, c1), (A2, b2, c2))


@njit(cache=True)
def _join_Abc_numba(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2

    output_batch_shape = ()
    batch1, batch2 = A1.shape[:-2], A2.shape[:-2]

    core1 = A1.shape[-1]
    core2 = A2.shape[-1]

    poly_shape1 = c1.shape[0:]
    poly_shape2 = c2.shape[0:]

    c1_flat_shape = (*batch1, int(np.prod(np.array(poly_shape1))))
    c2_flat_shape = (*batch2, int(np.prod(np.array(poly_shape2))))
    c1_flat = np.reshape(c1, c1_flat_shape)
    c2_flat = np.reshape(c2, c2_flat_shape)

    m1 = len(poly_shape1)
    m2 = len(poly_shape2)
    n1 = core1 - m1
    n2 = core2 - m2

    # Create a joint A1 and A2 array
    joint_A = np.zeros((*output_batch_shape, core1 + core2, core1 + core2), dtype=np.complex128)
    joint_A[..., :core1, :core1] = A1
    joint_A[..., core1:, core1:] = A2

    # Reorder the rows to group core and derived variables
    rows_A = np.empty((*output_batch_shape, core1 + core2, core1 + core2), dtype=np.complex128)
    rows_A[..., :n1, :] = joint_A[..., :n1, :]
    rows_A[..., n1 : (n1 + n2), :] = joint_A[..., core1 : core1 + n2, :]
    rows_A[..., (n1 + n2) : (n1 + n2 + m1), :] = joint_A[..., n1:core1, :]
    rows_A[..., (n1 + n2 + m1) : (n1 + n2 + m1 + m2), :] = joint_A[..., core1 + n2 :, :]

    # Reorder the columns to group core and derived variables
    A = np.empty((*output_batch_shape, core1 + core2, core1 + core2), dtype=np.complex128)
    A[..., :, :n1] = rows_A[..., :, :n1]
    A[..., :, n1 : (n1 + n2)] = rows_A[..., :, core1 : core1 + n2]
    A[..., :, (n1 + n2) : (n1 + n2 + m1)] = rows_A[..., :, n1:core1]
    A[..., :, (n1 + n2 + m1) : (n1 + n2 + m1 + m2)] = rows_A[..., :, core1 + n2 :]

    b = np.empty((*output_batch_shape, core1 + core2), dtype=np.complex128)
    b[..., :n1] = b1[..., :n1]
    b[..., n1 : (n1 + n2)] = b2[..., :n2]
    b[..., (n1 + n2) : (n1 + n2 + m1)] = b1[..., n1:]
    b[..., (n1 + n2 + m1) : (n1 + n2 + m1 + m2)] = b2[..., n2:]

    c1_expanded = np.expand_dims(c1_flat, axis=-1)
    c2_expanded = np.expand_dims(c2_flat, axis=-2)
    c = c1_expanded * c2_expanded
    c = np.reshape(c, (*output_batch_shape, *poly_shape1, *poly_shape2))
    return A, b, c

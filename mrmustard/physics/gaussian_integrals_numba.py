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
    r"""
    Joins two ``(A,b,c)`` triples into a single ``(A,b,c)``.

    It supports arbitrary batch dimensions using an einsum-like string notation.
    For example:
    - "i,j->ij" is like a "kron" mode (Kronecker product of batch dimensions)
    - "i,i->i" is like a "zip" mode (parallel processing of same-sized batches)
    - "i,j->ij", "ij,ik->ijk", "i,->i" are all valid batch dimension specifications

    The string only refers to the batch dimensions. The core (non-batch) dimensions are handled
    as follows: the A matrices and b vectors are joined, and the shape of c is the
    concatenation of the shapes of c1 and c2 (batch excluded).

    Input parameters are expected to have arbitrary batch dimensions, e.g. ``A1.shape = (batch1, n1, n1)``,
    ``b1.shape = (batch1, n1)``, ``c1.shape = (batch1, *d1)``.
    The number of non-batch dimensions in ``ci`` (i.e. ``len(di)``) corresponds to the number
    of rows and columns of ``Ai`` and ``bi`` that are kept last (derived variables).
    For instance, if ``d1 = (4, 3)`` and ``d2=(7,)``, then the last 2 rows and columns of
    ``A1`` and ``b1`` and the last 1 row and column of ``A2`` and ``b2`` are kept last in the
    joined ``A`` and ``b``.

    Arguments:
        Abc1: The first ``(A,b,c)`` triple
        Abc2: The second ``(A,b,c)`` triple
        batch_string: An (optional) einsum-like string in the format "in1,in2->out" that specifies
            how batch dimensions should be handled. If ``None``, defaults to a kronecker product i.e.
            "i,j->ij".

    Returns:
        The joined ``(A,b,c)`` triple
    """
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2

    # TODO: numbafy broadcasting
    verify_batch_triple(A1, b1, c1)
    verify_batch_triple(A2, b2, c2)

    batch1, batch2 = A1.shape[:-2], A2.shape[:-2]
    batch_dim1, batch_dim2 = len(batch1), len(batch2)

    if batch_string is None:
        batch_string = outer_product_batch_str(len(batch1), len(batch2))

    input_str, output_str = batch_string.split("->")
    input_parts = input_str.split(",")
    in1, in2 = input_parts

    nA1, mA1 = A1.shape[-2:]
    nA2, mA2 = A2.shape[-2:]
    nb1 = b1.shape[-1]
    nb2 = b2.shape[-1]

    poly_shape1 = c1.shape[batch_dim1:]
    poly_shape2 = c2.shape[batch_dim2:]

    # Step 1 & 2: Determine broadcast shape based on batch_string and broadcast tensors
    broadcast_dims = dict(zip(in1, batch1))
    for dim, batch in zip(in2, batch2):
        if dim in broadcast_dims and broadcast_dims[dim] != batch:
            raise ValueError(f"Dimension mismatch for {dim}: {broadcast_dims[dim]} != {batch}")
        broadcast_dims[dim] = batch

    output_shape = []
    for dim in output_str:
        if dim not in broadcast_dims:
            raise ValueError(f"Output dimension {dim} not found in inputs")
        output_shape.append(broadcast_dims[dim])

    # Create broadcast shapes
    broadcast_shape1 = []
    broadcast_shape2 = []

    for dim in output_str:
        if dim in in1:
            idx = in1.index(dim)
            broadcast_shape1.append(batch1[idx])
        else:
            broadcast_shape1.append(1)

        if dim in in2:
            idx = in2.index(dim)
            broadcast_shape2.append(batch2[idx])
        else:
            broadcast_shape2.append(1)

    # Broadcast A, b, c to the output batch shape
    A1_new_shape = (*tuple(broadcast_shape1), nA1, mA1)
    A2_new_shape = (*tuple(broadcast_shape2), nA2, mA2)
    b1_new_shape = (*tuple(broadcast_shape1), nb1)
    b2_new_shape = (*tuple(broadcast_shape2), nb2)
    c1_new_shape = (*tuple(broadcast_shape1), *poly_shape1)
    c2_new_shape = (*tuple(broadcast_shape2), *poly_shape2)

    # Reshape to add broadcasting dimensions
    A1_reshaped = np.reshape(A1, A1_new_shape)
    A2_reshaped = np.reshape(A2, A2_new_shape)
    b1_reshaped = np.reshape(b1, b1_new_shape)
    b2_reshaped = np.reshape(b2, b2_new_shape)
    c1_reshaped = np.reshape(c1, c1_new_shape)
    c2_reshaped = np.reshape(c2, c2_new_shape)

    # Create full output shape for broadcasting
    output_batch_shape = tuple(output_shape)
    A1_broadcast_shape = (*output_batch_shape, nA1, mA1)
    A2_broadcast_shape = (*output_batch_shape, nA2, mA2)
    b1_broadcast_shape = (*output_batch_shape, nb1)
    b2_broadcast_shape = (*output_batch_shape, nb2)
    c1_broadcast_shape = (*output_batch_shape, *poly_shape1)
    c2_broadcast_shape = (*output_batch_shape, *poly_shape2)

    # Step 2: Broadcast tensors to the output shape
    A1_broadcasted = np.broadcast_to(A1_reshaped, A1_broadcast_shape)
    A2_broadcasted = np.broadcast_to(A2_reshaped, A2_broadcast_shape)
    b1_broadcasted = np.broadcast_to(b1_reshaped, b1_broadcast_shape)
    b2_broadcasted = np.broadcast_to(b2_reshaped, b2_broadcast_shape)
    c1_broadcasted = np.broadcast_to(c1_reshaped, c1_broadcast_shape)
    c2_broadcasted = np.broadcast_to(c2_reshaped, c2_broadcast_shape)

    return _join_Abc_numba(
        (A1_broadcasted, b1_broadcasted, c1_broadcasted),
        (A2_broadcasted, b2_broadcasted, c2_broadcasted),
        output_batch_shape,
    )


@njit(cache=True)
def _join_Abc_numba(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    output_batch_shape: tuple[int, ...],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2

    core1 = A1.shape[-1]
    core2 = A2.shape[-1]

    batch_dim = len(output_batch_shape)
    poly_shape1 = c1.shape[batch_dim:]
    poly_shape2 = c2.shape[batch_dim:]

    c1_flat_shape = (*output_batch_shape, int(np.prod(np.array(poly_shape1))))
    c2_flat_shape = (*output_batch_shape, int(np.prod(np.array(poly_shape2))))
    c1_flat = np.reshape(np.ascontiguousarray(c1), c1_flat_shape)
    c2_flat = np.reshape(np.ascontiguousarray(c2), c2_flat_shape)

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

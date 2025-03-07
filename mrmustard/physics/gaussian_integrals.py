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

from typing import Sequence
import numpy as np
from mrmustard import math
from mrmustard.utils.typing import ComplexMatrix, ComplexVector, ComplexTensor


def real_gaussian_integral(
    Abc: tuple,
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
    A, b, c = Abc

    if not idx:
        return A, b, c
    not_idx = tuple(i for i in range(A.shape[-1]) if i not in idx)
    idx = math.astensor(idx)
    not_idx = math.astensor(not_idx)

    M = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2)
    bM = math.gather(b, idx, axis=-1)

    if math.asnumpy(not_idx).shape != (0,):
        D = math.gather(math.gather(A, idx, axis=-1), not_idx, axis=-2)
        R = math.gather(math.gather(A, not_idx, axis=-1), not_idx, axis=-2)
        bR = math.gather(b, not_idx, axis=-1)
        T = math.transpose
        L = T(math.solve(T(M), T(D)))
        A_post = R - math.matmul(L, T(D))
        b_post = bR - math.matvec(L, bM)
    else:
        A_post = math.astensor([])
        b_post = math.astensor([])

    c_post = (
        c
        * math.sqrt((2 * np.pi) ** len(idx), math.complex128)
        * math.sqrt((-1) ** len(idx) / math.det(M), math.complex128)
        * math.exp(-0.5 * math.sum(bM * math.solve(M, bM)))
    )

    return A_post, b_post, c_post


def join_Abc_real(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    idx1: Sequence[int],
    idx2: Sequence[int],
):
    r"""Direct sum of two ``(A,b,c)`` triples into a single ``(A,b,c)`` triple, where indices corresponding to the same variable are "fused together",
    by considering their Bargmann function as having the same variables. For example ``idx1=(0,1,2)`` and ``idx2=(1,2,3)`` means that indices 1 and 2
    will be fused because they are present on both tuples. This is useful for computing real Gaussian integrals where the variable on either object is the same,
    rather than a pair of conjugate variables for complex Gaussian integrals.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple
        idx1: the indices of the first ``(A,b,c)`` triple to fuse
        idx2: the indices of the second ``(A,b,c)`` triple to fuse

    Returns:
        The joined ``(A,b,c)`` triple with the order [idx1(or idx2), not_idx2].
    """
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    c1 = math.astensor(c1)
    c2 = math.astensor(c2)
    if len(idx1) != len(idx2):
        raise ValueError(
            f"idx1 and idx2j must have the same length, got {len(idx1)} and {len(idx2)}"
        )

    if (len(idx1) > A1.shape[-1]) or (len(idx2) > A2.shape[-1]):
        raise ValueError(f"idx1 and idx2 must be valid, got {len(idx1)} and {len(idx2)}")

    not_idx1 = tuple(i for i in range(A1.shape[-1]) if i not in idx1)
    not_idx2 = tuple(i for i in range(A2.shape[-1]) if i not in idx2)

    idx1, idx2 = math.astensor(idx1), math.astensor(idx2)
    not_idx1, not_idx2 = math.astensor(not_idx1), math.astensor(not_idx2)

    A1_idx_idx = math.gather(math.gather(A1, idx1, axis=-1), idx1, axis=-2)
    b1_idx = math.gather(b1, idx1, axis=-1)
    if not_idx1.size > 0:
        A1_idx_notidx = math.gather(math.gather(A1, not_idx1, axis=-1), idx1, axis=-2)
        A1_notidx_idx = math.gather(math.gather(A1, idx1, axis=-1), not_idx1, axis=-2)
        A1_notidx_notidx = math.gather(math.gather(A1, not_idx1, axis=-1), not_idx1, axis=-2)
        b1_notidx = math.gather(b1, not_idx1, axis=-1)
    A2_idx_idx = math.gather(math.gather(A2, idx2, axis=-1), idx2, axis=-2)
    b2_idx = math.gather(b2, idx2, axis=-1)
    if not_idx2.size > 0:
        A2_idx_notidx = math.gather(math.gather(A2, not_idx2, axis=-1), idx2, axis=-2)
        A2_notidx_idx = math.gather(math.gather(A2, idx2, axis=-1), not_idx2, axis=-2)
        A2_notidx_notidx = math.gather(math.gather(A2, not_idx2, axis=-1), not_idx2, axis=-2)
        b2_notidx = math.gather(b2, not_idx2, axis=-1)

    if math.asnumpy(not_idx1).shape == (0,):
        A12 = math.block([[A1 + A2_idx_idx, A2_notidx_idx], [A2_idx_notidx, A2_notidx_notidx]])
        b12 = math.concat([b1 + b2_idx, b2_notidx], axis=-1)
    elif math.asnumpy(not_idx2).shape == (0,):
        A12 = math.block([[A2 + A1_idx_idx, A1_notidx_idx], [A1_idx_notidx, A1_notidx_notidx]])
        b12 = math.concat([b2 + b1_idx, b1_notidx], axis=-1)
    else:
        O_n = math.zeros((len(not_idx1), len(not_idx2)), math.complex128)
        A12 = math.block(
            [
                [A1_idx_idx + A2_idx_idx, A1_idx_notidx, A2_idx_notidx],
                [A1_notidx_idx, A1_notidx_notidx, O_n],
                [A2_notidx_idx, O_n.T, A2_notidx_notidx],
            ]
        )
        b12 = math.concat([b1_idx + b2_idx, b1_notidx, b2_notidx], axis=-1)
    c12 = math.reshape(math.outer(c1, c2), c1.shape + c2.shape)
    return A12, b12, c12


def reorder_abc(Abc: tuple, order: Sequence[int]):
    r"""
    Reorders the indices of the A matrix and b vector of an (A,b,c) triple.

    Arguments:
        Abc: the ``(A,b,c)`` triple
        order: the new order of the indices

    Returns:
        The reordered ``(A,b,c)`` triple
    """
    A, b, c = Abc
    c = math.astensor(c)
    order = list(order)
    if len(order) == 0:
        return A, b, c
    batched = len(A.shape) == 3 and len(b.shape) == 2 and len(c.shape) > 0
    dim_poly = len(c.shape) - int(batched)
    n = A.shape[-1] - dim_poly

    if len(order) != n:
        raise ValueError(f"order must have length {n}, got {len(order)}")

    if any(i >= n or n < 0 for i in order):
        raise ValueError(f"elements in `order` must be between 0 and {n-1}, got {order}")
    order += list(range(len(order), len(order) + dim_poly))
    order = math.astensor(order)
    A = math.gather(math.gather(A, order, axis=-1), order, axis=-2)
    b = math.gather(b, order, axis=-1)
    return A, b, c


def join_Abc(Abc1: tuple, Abc2: tuple, batch_string: str) -> tuple:
    r"""Joins two ``(A,b,c)`` triples into a single ``(A,b,c)``.

    It supports arbitrary batch dimensions using an einsum-like string notation.
    For example:
    - "i,j->ij" is like a "kron" mode (Kronecker product of batch dimensions)
    - "i,i->i" is like a "zip" mode (parallel processing of same-sized batches)
    - "i,j->ij", "ij,ik->ijk", "i,->i" are all valid batch dimension specifications

    The string only refers to the batch dimensions. The core (non-batch) dimensions are handled
    as before: the A matrices and b vectors are joined, and the shape of c is the
    concatenation of the shapes of c1 and c2 (batch excluded).

    It supports varying batch dimensions, e.g. ``A1.shape = (batch1, n1, n1)``,
    ``b1.shape = (batch1, n1)``, ``c1.shape = (batch1, *d1)`` or no batch dimension:
    ``A1.shape = (n1, n1)``, ``b1.shape = (n1)``, ``c1.shape = (*d1)``.
    The number of non-batch dimensions in ``ci`` (i.e. ``len(di)``) corresponds to the number
    of rows and columns of ``Ai`` and ``bi`` that are kept last (derived variables).
    So for instance, if ``d1 = (4, 3)`` and ``d2=(7,)``, then the last 2 rows and columns of
    ``A1`` and ``b1`` and the last 1 row and column of ``A2`` and ``b2`` are kept last in the
    joined ``A`` and ``b``. If inputs are not batched, the output has a batch dimension of size 1
    added to it.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple
        batch_string: an einsum-like string in the format "in1,in2->out" that specifies how batch dimensions should be handled

    Returns:
        The joined ``(A,b,c)`` triple
    """
    # 0. unpack and prepare inputs
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    # A1 = math.atleast_3d(A1, dtype=math.complex128)
    # A2 = math.atleast_3d(A2, dtype=math.complex128)
    # b1 = math.atleast_2d(b1, dtype=math.complex128)
    # b2 = math.atleast_2d(b2, dtype=math.complex128)
    c1 = math.astensor(c1, dtype=math.complex128)
    c2 = math.astensor(c2, dtype=math.complex128)

    # 1. Parse the batch string
    if "->" not in batch_string:
        raise ValueError(
            f"Invalid batch string format: {batch_string}. Expected format: 'in1,in2->out'"
        )

    input_str, output_str = batch_string.split("->")
    input_parts = input_str.split(",")
    if len(input_parts) != 2:
        raise ValueError(f"Expected 2 input parts in batch string, got {len(input_parts)}")

    in1, in2 = input_parts

    # 2. Get dimensions and validate
    batch1_A = A1.shape[:-2]
    batch2_A = A2.shape[:-2]
    nA1, mA1 = A1.shape[-2:]
    nA2, mA2 = A2.shape[-2:]
    batch1_b = b1.shape[:-1]
    batch2_b = b2.shape[:-1]
    nb1 = b1.shape[-1]
    nb2 = b2.shape[-1]
    batch1_c = c1.shape[: len(batch1_A)]
    batch2_c = c2.shape[: len(batch2_A)]
    poly_shape1 = c1.shape[len(batch1_A) :]
    poly_shape2 = c2.shape[len(batch2_A) :]

    # Check if batch dimensions match the provided string
    if len(in1) != len(batch1_A):
        raise ValueError(
            f"Batch dimensions in first input ({len(batch1_A)}) don't match the provided string '{in1}'"
        )
    if len(in2) != len(batch2_A):
        raise ValueError(
            f"Batch dimensions in second input ({len(batch2_A)}) don't match the provided string '{in2}'"
        )

    # Check the batch dimensions across the two triples are the same
    if batch1_A != batch1_b or batch1_A != batch1_c or batch1_b != batch1_c:
        raise ValueError(
            f"Batch dmensions of the first triple ({batch1_A}, {batch1_b}, {batch1_c}) are inconsistent"
        )
    if batch2_A != batch2_b or batch2_A != batch2_c or batch2_b != batch2_c:
        raise ValueError(
            f"Batch dmensions of the second triple ({batch2_A}, {batch2_b}, {batch2_c}) are inconsistent"
        )

    # 3. Get shapes and sizes
    m1 = len(poly_shape1)
    m2 = len(poly_shape2)
    n1 = nA1 - m1
    n2 = nA2 - m2

    # Step 0: Flatten the non-batch dimensions of c1 and c2
    c1_flat_shape = batch1_c + (int(np.prod(poly_shape1)),)
    c2_flat_shape = batch2_c + (int(np.prod(poly_shape2)),)
    c1_flat = math.reshape(c1, c1_flat_shape)
    c2_flat = math.reshape(c2, c2_flat_shape)

    # Step 1 & 2: Determine broadcast shape based on batch_string and broadcast tensors
    broadcast_dims = {}
    for i, dim in enumerate(in1):
        broadcast_dims[dim] = batch1_A[i]
    for i, dim in enumerate(in2):
        if dim in broadcast_dims and broadcast_dims[dim] != batch2_A[i]:
            raise ValueError(
                f"Dimension mismatch for {dim}: {broadcast_dims[dim]} != {batch2_A[i]}"
            )
        broadcast_dims[dim] = batch2_A[i]

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
            broadcast_shape1.append(batch1_A[idx])
        else:
            broadcast_shape1.append(1)

        if dim in in2:
            idx = in2.index(dim)
            broadcast_shape2.append(batch2_A[idx])
        else:
            broadcast_shape2.append(1)

    # Broadcast A, b, c to the output batch shape
    A1_new_shape = tuple(broadcast_shape1) + (nA1, mA1)
    A2_new_shape = tuple(broadcast_shape2) + (nA2, mA2)
    b1_new_shape = tuple(broadcast_shape1) + (nb1,)
    b2_new_shape = tuple(broadcast_shape2) + (nb2,)
    c1_new_shape = tuple(broadcast_shape1) + (c1_flat.shape[-1],)
    c2_new_shape = tuple(broadcast_shape2) + (c2_flat.shape[-1],)

    # Reshape to add broadcasting dimensions
    A1_reshaped = math.reshape(A1, A1_new_shape)
    A2_reshaped = math.reshape(A2, A2_new_shape)
    b1_reshaped = math.reshape(b1, b1_new_shape)
    b2_reshaped = math.reshape(b2, b2_new_shape)
    c1_reshaped = math.reshape(c1_flat, c1_new_shape)
    c2_reshaped = math.reshape(c2_flat, c2_new_shape)

    # Create full output shape for broadcasting
    output_batch_shape = tuple(output_shape)
    A1_broadcast_shape = output_batch_shape + (nA1, mA1)
    A2_broadcast_shape = output_batch_shape + (nA2, mA2)
    b1_broadcast_shape = output_batch_shape + (nb1,)
    b2_broadcast_shape = output_batch_shape + (nb2,)
    c1_broadcast_shape = output_batch_shape + (c1_flat.shape[-1],)
    c2_broadcast_shape = output_batch_shape + (c2_flat.shape[-1],)

    # Step 2: Broadcast tensors to the output shape
    A1_broadcasted = math.broadcast_to(A1_reshaped, A1_broadcast_shape)
    A2_broadcasted = math.broadcast_to(A2_reshaped, A2_broadcast_shape)
    b1_broadcasted = math.broadcast_to(b1_reshaped, b1_broadcast_shape)
    b2_broadcasted = math.broadcast_to(b2_reshaped, b2_broadcast_shape)
    c1_broadcasted = math.broadcast_to(c1_reshaped, c1_broadcast_shape)
    c2_broadcasted = math.broadcast_to(c2_reshaped, c2_broadcast_shape)

    # Step 3: Join A1 and A2
    A1Z = math.concat(
        [A1_broadcasted, math.zeros(output_batch_shape + (nA1, nA2), dtype=A1.dtype)], axis=-1
    )
    ZA2 = math.concat(
        [math.zeros(output_batch_shape + (nA2, nA1), dtype=A2.dtype), A2_broadcasted], axis=-1
    )

    A = math.concat([A1Z, ZA2], axis=-2)

    # Reorder the rows and columns to group core and derived variables
    A = math.concat(
        [
            A[..., :n1, :],
            A[..., nA1 : nA1 + n2, :],
            A[..., n1:nA1, :],
            A[..., nA1 + n2 :, :],
        ],
        axis=-2,
    )

    A = math.concat(
        [
            A[..., :, :n1],
            A[..., :, nA1 : nA1 + n2],
            A[..., :, n1:nA1],
            A[..., :, nA1 + n2 :],
        ],
        axis=-1,
    )

    # Step 4: Concatenate b1 and b2
    b = math.concat([b1_broadcasted, b2_broadcasted], axis=-1)
    b = math.concat(
        [
            b[..., :n1],
            b[..., nA1 : nA1 + n2],
            b[..., n1:nA1],
            b[..., nA1 + n2 :],
        ],
        axis=-1,
    )

    # Step 5 & 6: Compute outer product of the last dimensions of c1 and c2
    c1_expanded = c1_broadcasted[..., :, None]
    c2_expanded = c2_broadcasted[..., None, :]
    c = c1_expanded * c2_expanded
    # Reshape c to the desired output shape
    c_shape = output_batch_shape + poly_shape1 + poly_shape2
    c = math.reshape(c, c_shape)

    return A, b, c


def true_branch_complex_gaussian_integral_1(m, M, bM, det_M, c, D, R, bR):
    r"""
    True branch of the complex gaussian_integral_1 function.
    Executed if the matrix M is invertible.

    Args:
        m: the number of pairs of conjugate variables to integrate over
        M: the matrix of the quadratic form
        bM: the vector of the linear term
        det_M: the determinant of M
        c: the coefficient of the exponential
        D: the matrix of the linear term
        R: the matrix of the quadratic form
        bR: the vector of the linear term

    Returns:
        The post-integration parameters
    """
    inv_M = math.inv(M)
    M_bM = math.solve(M, bM)

    c_factor = math.sqrt(math.cast((-1) ** m / det_M, "complex128")) * math.exp(
        -0.5 * math.sum(bM * M_bM, axis=-1)
    )
    c_reshaped = math.reshape(c_factor, c.shape[:1] + (1,) * (len(c.shape) - 1))
    c_post = c * c_reshaped

    A_post = R - math.einsum("bij,bjk,blk->bil", D, inv_M, D)
    b_post = bR - math.einsum("bij,bj->bi", D, M_bM)
    return (
        math.cast(A_post, "complex128"),
        math.cast(b_post, "complex128"),
        math.cast(c_post, "complex128"),
    )


def false_branch_complex_gaussian_integral_1(m, M, bM, det_M, c, D, R, bR):
    r"""
    False branch of the complex gaussian_integral_1 function.
    Exectued if the matrix M is singular.

    Args:
        m: the number of pairs of conjugate variables to integrate over
        M: the matrix of the quadratic form
        bM: the vector of the linear term
        det_M: the determinant of M
        c: the coefficient of the exponential
        D: the matrix of the linear term
        R: the matrix of the quadratic form
        bR: the vector of the linear term

    Returns:
        The post-integration parameters
    """
    return math.infinity_like(R), math.infinity_like(bR), math.infinity_like(c)


def complex_gaussian_integral_1(
    Abc: tuple,
    idx_z: Sequence[int],
    idx_zconj: Sequence[int],
    measure: float = -1,
):
    r"""Computes the complex Gaussian integral

    :math:`\int_{C^m} F(z,\beta) d\mu(z)`,

    over ``m`` pairs of conjugate variables :math:`z_i,z_j^*` whose positions :math:`i`` and :math:`j` are indicated by ``idx1`` and ``idx2``.
    The function has the form

    :math:`F(z,\beta) = \sum_k c_k\partial_\beta^k \exp(0.5 (z,\beta) A (z,\beta)^T + (z,\beta)b),\quad z\in\mathbb{C}^{n}, \beta\in\mathbb{C}^{N}`,

    where ``k`` is a multi-index of the same dimension as ``\beta`` and the sum is over all multi-indices.
    The ``Abc`` parameters can have an additional batch dimension and the batch size must be the same for all three.
    Inputs are batched when ``A.shape=(batch,n,n)``, ``b.shape=(batch,n)`` and ``c.shape=(batch,d1,d2,...,dN)``.
    The output is then the A,b,c parametrization of a function in the same form with parameters of shape
    ``A.shape=(batch,n-2m,n-2m)``, ``b.shape=(batch,n-2m)``, ``c.shape=(batch,d1,d2,...,dN)``.

    The integration measure is given by (``s`` here is the ``measure`` argument):
    :math:`dmu(z) = \textrm{exp}(s * |z|^2) \frac{d^{2m}z}{\pi^m} = \frac{1}{\pi^m}\textrm{exp}(s * |z|^2) d\textrm{Re}(z) d\textrm{Im}(z)`

    Arguments:
        A,b,c: the ``(A,b,c)`` triple that defines :math:`F(z,\beta)`
        idx_z: the tuple of indices indicating which :math:`z` variables to integrate over
        idx_zconj: the tuple of indices indicating which :math:`z^*` variables to integrate over
        measure: the exponent of the measure (default is -1: Bargmann measure)

    Returns:
        The ``(A,b,c)`` triple which parametrizes the result of the integral with batch dimension preserved (if any).

    Raises:
        ValueError: If ``idx1`` and ``idx2`` have different lengths, or they indicate indices beyond ``n``, or if ``A``, ``b``, ``c`` have non-matching batch size.
    """
    if len(idx_z) != len(idx_zconj):
        raise ValueError(
            f"idx1 and idx2 must have the same length, got {len(idx_z)} and {len(idx_zconj)}"
        )
    A, b, c = Abc
    c = math.astensor(c)
    # assuming c is batched accordingly
    batched = len(A.shape) == 3 and len(b.shape) == 2
    if not batched:
        A = math.atleast_3d(A)
        b = math.atleast_2d(b)
        c = math.expand_dims(c, 0)
    if not A.shape[0] == b.shape[0] == c.shape[0]:
        raise ValueError(
            f"Batch size mismatch: got {A.shape[0]} for A, {b.shape[0]} for b and {c.shape[0]} for c."
        )

    (_, n_plus_N, _) = A.shape
    if b.shape[-1] != A.shape[-1]:
        raise ValueError(f"A and b must have compatible shapes, got {A.shape} and {b.shape}")
    N = len(c.shape[1:])  # number of beta variables
    n = n_plus_N - N  # number of z variables
    m = len(idx_z)  # number of pairs to integrate over
    idx = tuple(idx_z) + tuple(idx_zconj)
    if any(i >= n for i in idx):
        raise ValueError(
            f"Indices must be less than {n}, got {tuple(i for i in idx_z if i >= n)} and {tuple(i for i in idx_zconj if i >= n)}"
        )

    if len(idx) == 0:
        if not batched:
            return A[0], b[0], c[0]
        return A, b, c

    not_idx = math.astensor([i for i in range(n_plus_N) if i not in idx], dtype=math.int64)
    # order matters here; idx should be made a tensor after doing all the list comprehensions and boolean operations.
    idx = math.astensor(idx, dtype=math.int64)
    eye = math.eye(m, dtype=A.dtype)

    eye = math.eye(m, dtype=A.dtype)
    Z = math.zeros((m, m), dtype=A.dtype)
    X = math.block([[Z, eye], [eye, Z]])
    M = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2) + X * measure
    bM = math.gather(b, idx, axis=-1)

    D = math.gather(math.gather(A, idx, axis=-1), not_idx, axis=-2)
    R = math.gather(math.gather(A, not_idx, axis=-1), not_idx, axis=-2)
    bR = math.gather(b, not_idx, axis=-1)
    det_M = math.det(M)
    det_nonzero = math.abs(det_M) > 1e-12

    # return infinity if M is singular; otherwise, return the post-integration parameters
    A_post, b_post, c_post = math.conditional(
        det_nonzero,
        true_branch_complex_gaussian_integral_1,
        false_branch_complex_gaussian_integral_1,
        m,
        M,
        bM,
        det_M,
        c,
        D,
        R,
        bR,
    )

    if not batched:
        return A_post[0], b_post[0], c_post[0]
    return A_post, b_post, c_post


def complex_gaussian_integral_2(
    Abc1: tuple,
    Abc2: tuple,
    idx1: Sequence[int],
    idx2: Sequence[int],
    batch_string: str | None = None,
    measure: float = -1,
) -> tuple:
    r"""Computes the complex Gaussian integral

    :math:`\int_{C^m} F_1(z,\beta_1)F_2(z,\beta_2) d\mu(z)`,

    over ``m`` pairs of conjugate variables :math:`z_i,z_j^*` whose positions :math:`i`` and :math:`j` are indicated by ``idx1`` and ``idx2``.
    Each of the functions has the form

    :math:`F(z,\beta) = \sum_k c_k\partial_\beta^k \exp(0.5 (z,\beta) A (z,\beta)^T + (z,\beta)b),\quad z\in\mathbb{C}^{n_i}, \beta\in\mathbb{C}^{N_i}`,

    where ``k`` is a multi-index of the same dimension as ``\beta`` and the sum is over all multi-indices.
    The ``Abc`` parameters can have an additional batch dimension and the batch size must be the same for all six.
    Inputs are batched when ``A.shape=(batch,n,n)``, ``b.shape=(batch,n)`` and ``c.shape=(batch,d1,d2,...,dN)``.
    The output is then the A,b,c parametrization of a function in the same form with parameters of shape
    ``A.shape=(batch,n-2m,n-2m)``, ``b.shape=(batch,n-2m)``, ``c.shape=(batch,d1,d2,...,dN)``.

    The integration measure is given by (``s`` here is the ``measure`` argument):
    :math:`dmu(z) = \textrm{exp}(s * |z|^2) \frac{d^{2m}z}{\pi^m} = \frac{1}{\pi^m}\textrm{exp}(s * |z|^2) d\textrm{Re}(z) d\textrm{Im}(z)`

    Arguments:
        Abc1: the ``(A,b,c)`` triple that defines :math:`F_1(z,\beta)`
        Abc2: the ``(A,b,c)`` triple that defines :math:`F_2(z,\beta)`
        idx1: the tuple of indices of the :math:`z` variables of the first function to integrate over
        idx2: the tuple of indices of the :math:`z^*` variables of the second function to integrate over
        measure: the exponent of the measure (default is -1: Bargmann measure)
        batch_string: an einsum-like string in the format "in1,in2->out" that specifies how batch dimensions should be handled.
                    Default is "i,j->ij" (equivalent to the old "kron" mode). Use "i,i->i" for the old "zip" mode.

    Returns:
        The ``(A,b,c)`` triple which parametrizes the result of the integral with batch dimension preserved (if any).

    Raises:
        ValueError: If ``idx1`` and ``idx2`` have different lengths, or they indicate indices beyond ``n``, or if ``A``, ``b``, ``c`` have non-matching batch size.
    """

    if batch_string is None:
        str1 = "".join([chr(i) for i in range(97, 97 + len(Abc1[0].shape) - 2)])
        str2 = "".join([chr(i) for i in range(97, 97 + len(Abc2[0].shape) - 2)])
        out = "".join([chr(i) for i in range(97, 97 + len(Abc1[0].shape) + len(Abc2[0].shape) - 4)])
        batch_string = f"{str1},{str2}->{out}"

    # Join the Abc parameters
    A, b, c = join_Abc(Abc1, Abc2, batch_string=batch_string)

    # vectorize the batch dimensions
    batch_shape = A.shape[:-2]
    A_core_shape = A.shape[-2:]
    b_core_shape = b.shape[-1:]
    c_core_shape = c.shape[len(batch_shape) :]
    A = math.reshape(A, (-1,) * bool(batch_shape) + A_core_shape)
    b = math.reshape(b, (-1,) * bool(batch_shape) + b_core_shape)
    c = math.reshape(c, (-1,) * bool(batch_shape) + c_core_shape)

    # offset idx2 to account for the core variables of the first triple
    A1, _, c1 = Abc1
    batch_dims_1 = len(math.atleast_3d(A1).shape) - 2
    derived_1 = len(math.atleast_1d(c1).shape[batch_dims_1:])
    core_1 = A1.shape[-1] - derived_1
    idx2 = tuple(i + core_1 for i in idx2)

    # compute the integral and reshape the output batch dimensions
    A_out, b_out, c_out = complex_gaussian_integral_1((A, b, c), idx1, idx2, measure)
    A_out = math.reshape(A_out, batch_shape + (A_out.shape[-2], A_out.shape[-1]))
    b_out = math.reshape(b_out, batch_shape + (b_out.shape[-1],))
    c_out = math.reshape(c_out, batch_shape + c_out.shape[len(batch_shape) :])
    return A_out, b_out, c_out

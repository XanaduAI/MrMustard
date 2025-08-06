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

from collections.abc import Sequence

import numpy as np

from mrmustard import math
from mrmustard.physics.utils import outer_product_batch_str, verify_batch_triple
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector


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
            f"idx1 and idx2j must have the same length, got {len(idx1)} and {len(idx2)}",
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
            ],
        )
        b12 = math.concat([b1_idx + b2_idx, b1_notidx, b2_notidx], axis=-1)
    c12 = math.reshape(math.outer(c1, c2), c1.shape + c2.shape)
    return A12, b12, c12


def join_Abc(
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

    poly_shape1 = c1.shape[batch_dim1:] if len(c1.shape) >= 1 else ()
    poly_shape2 = c2.shape[batch_dim2:] if len(c2.shape) >= 1 else ()

    m1 = len(poly_shape1)
    m2 = len(poly_shape2)
    n1 = nA1 - m1
    n2 = nA2 - m2

    # Step 0: Flatten the non-batch dimensions of c1 and c2
    c1_flat_shape = (*batch1, int(np.prod(poly_shape1)))
    c2_flat_shape = (*batch2, int(np.prod(poly_shape2)))
    c1_flat = math.reshape(c1, c1_flat_shape)
    c2_flat = math.reshape(c2, c2_flat_shape)

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
    c1_new_shape = (*tuple(broadcast_shape1), c1_flat.shape[-1])
    c2_new_shape = (*tuple(broadcast_shape2), c2_flat.shape[-1])

    # Reshape to add broadcasting dimensions
    A1_reshaped = math.reshape(A1, A1_new_shape)
    A2_reshaped = math.reshape(A2, A2_new_shape)
    b1_reshaped = math.reshape(b1, b1_new_shape)
    b2_reshaped = math.reshape(b2, b2_new_shape)
    c1_reshaped = math.reshape(c1_flat, c1_new_shape)
    c2_reshaped = math.reshape(c2_flat, c2_new_shape)

    # Create full output shape for broadcasting
    output_batch_shape = tuple(output_shape)
    A1_broadcast_shape = (*output_batch_shape, nA1, mA1)
    A2_broadcast_shape = (*output_batch_shape, nA2, mA2)
    b1_broadcast_shape = (*output_batch_shape, nb1)
    b2_broadcast_shape = (*output_batch_shape, nb2)
    c1_broadcast_shape = (*output_batch_shape, c1_flat.shape[-1])
    c2_broadcast_shape = (*output_batch_shape, c2_flat.shape[-1])

    # Step 2: Broadcast tensors to the output shape
    A1_broadcasted = math.broadcast_to(A1_reshaped, A1_broadcast_shape, dtype=math.complex128)
    A2_broadcasted = math.broadcast_to(A2_reshaped, A2_broadcast_shape, dtype=math.complex128)
    b1_broadcasted = math.broadcast_to(b1_reshaped, b1_broadcast_shape, dtype=math.complex128)
    b2_broadcasted = math.broadcast_to(b2_reshaped, b2_broadcast_shape, dtype=math.complex128)
    c1_broadcasted = math.broadcast_to(c1_reshaped, c1_broadcast_shape, dtype=math.complex128)
    c2_broadcasted = math.broadcast_to(c2_reshaped, c2_broadcast_shape, dtype=math.complex128)

    # Step 3: Join A1 and A2
    A1Z = math.concat(
        [A1_broadcasted, math.zeros((*output_batch_shape, nA1, nA2), dtype=math.complex128)],
        axis=-1,
    )
    ZA2 = math.concat(
        [math.zeros((*output_batch_shape, nA2, nA1), dtype=math.complex128), A2_broadcasted],
        axis=-1,
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
    batch_shape = M.shape[:-2]
    batch_dim = len(batch_shape)

    inv_M = math.inv(M)
    M_bM = math.solve(M, bM)

    c_factor = math.sqrt(math.cast((-1) ** m / det_M, "complex128")) * math.exp(
        -0.5 * math.sum(bM * M_bM, axis=-1),
    )
    c_reshaped = math.reshape(c_factor, batch_shape + (1,) * (len(c.shape[batch_dim:])))
    c_post = c * c_reshaped

    A_post = R - math.einsum("...ij,...jk,...lk->...il", D, inv_M, D)
    b_post = bR - math.einsum("...ij,...j->...i", D, M_bM)
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
            f"idx1 and idx2 must have the same length, got {len(idx_z)} and {len(idx_zconj)}",
        )

    A, b, c = Abc

    verify_batch_triple(A, b, c)
    batch_dim = len(A.shape[:-2])
    n_plus_N = A.shape[-1]
    if b.shape[-1] != n_plus_N:
        raise ValueError(f"A and b must have compatible shapes, got {A.shape} and {b.shape}")
    N = len(c.shape[batch_dim:])  # number of derived variables
    n = n_plus_N - N  # number of z variables
    m = len(idx_z)  # number of pairs to integrate over
    idx = tuple(idx_z) + tuple(idx_zconj)
    if any(i >= n for i in idx):
        raise ValueError(
            f"Indices must be less than {n}, got {tuple(i for i in idx_z if i >= n)} and {tuple(i for i in idx_zconj if i >= n)}",
        )

    if len(idx) == 0:
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
    return A_post, b_post, c_post


def complex_gaussian_integral_2(
    Abc1: tuple,
    Abc2: tuple,
    idx1: Sequence[int],
    idx2: Sequence[int],
    batch_string: str | None = None,
    measure: float = -1,
) -> tuple:
    r"""
    Computes the complex Gaussian integral

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
        batch_string: an (optional) einsum-like string in the format "in1,in2->out" that specifies how batch dimensions should be handled.

    Returns:
        The ``(A,b,c)`` triple which parametrizes the result of the integral with batch dimension preserved (if any).
    """
    A, b, c = join_Abc(Abc1, Abc2, batch_string=batch_string)

    # offset idx2 to account for the core variables of the first triple
    A1, _, c1 = Abc1
    batch_dims_1 = len(A1.shape[:-2])
    derived_1 = len(c1.shape[batch_dims_1:])
    core_1 = A1.shape[-1] - derived_1
    idx2 = tuple(i + core_1 for i in idx2)

    return complex_gaussian_integral_1((A, b, c), idx1, idx2, measure)

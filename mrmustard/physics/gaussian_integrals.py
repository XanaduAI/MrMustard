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

    A1_idx_idx = math.gather(math.gather(A1, idx1, axis=-1), idx1, axis=-2)
    b1_idx = math.gather(b1, idx1, axis=-1)
    if not_idx1:
        A1_idx_notidx = math.gather(math.gather(A1, not_idx1, axis=-1), idx1, axis=-2)
        A1_notidx_idx = math.gather(math.gather(A1, idx1, axis=-1), not_idx1, axis=-2)
        A1_notidx_notidx = math.gather(math.gather(A1, not_idx1, axis=-1), not_idx1, axis=-2)
        b1_notidx = math.gather(b1, not_idx1, axis=-1)
    A2_idx_idx = math.gather(math.gather(A2, idx2, axis=-1), idx2, axis=-2)
    b2_idx = math.gather(b2, idx2, axis=-1)
    if not_idx2:
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
    A = math.gather(math.gather(A, order, axis=-1), order, axis=-2)
    b = math.gather(b, order, axis=-1)
    return A, b, c


def join_Abc(Abc1: tuple, Abc2: tuple, mode: str = "kron") -> tuple:
    r"""Joins two ``(A,b,c)`` triples into a single ``(A,b,c)``.

    It supports a batch dimension, e.g. ``A1.shape = (batch, n1, n1)``,
    ``b1.shape = (batch, n1)``, ``c1.shape = (batch, *d1)`` or no batch dimension:
    ``A1.shape = (n1, n1)``, ``b1.shape = (n1)``, ``c1.shape = (*d1)``.
    The number of non-batch dimensions in ``ci`` (i.e. ``len(di)``) corresponds to the number
    of rows and columns of ``Ai`` and ``bi`` that are kept last. So for instance, if ``d1 = (4, 3)``
    and ``d2=(7,)``, then the last 2 rows and columns of ``A1`` and ``b1`` and the last 1 row and
    column of ``A2`` and ``b2`` are kept last in the joined ``A`` and ``b``.
    The shape of ``c`` is the concatenation of the shapes of ``c1`` and ``c2`` (batch excluded).
    If inputs are not batched, the output has a batch dimension of size 1 added to it.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple
        mode: how to treat the batch of the two ``(A,b,c)`` triples. Either ``kron`` for the Kronecker product or ``zip`` for parallel.

    Returns:
        The joined ``(A,b,c)`` triple
    """

    # 0. unpack and prepare inputs
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    A1 = math.atleast_3d(A1, dtype=math.complex128)
    A2 = math.atleast_3d(A2, dtype=math.complex128)
    b1 = math.atleast_2d(b1, dtype=math.complex128)
    b2 = math.atleast_2d(b2, dtype=math.complex128)
    c1 = math.atleast_1d(c1, dtype=math.complex128)
    c2 = math.atleast_1d(c2, dtype=math.complex128)

    # 1. input validation
    (batch1, nA1, mA1) = A1.shape
    (batch2, nA2, mA2) = A2.shape
    (batch1_b, nb1) = b1.shape
    (batch2_b, nb2) = b2.shape
    (batch1_c, *poly_shape1) = c1.shape
    (batch2_c, *poly_shape2) = c2.shape

    if not batch1 == batch1_b == batch1_c:
        raise ValueError(f"Batch sizes of Abc1 must match: got ({batch1}, {batch1_b}, {batch1_c})")
    if not batch2 == batch2_b == batch2_c:
        raise ValueError(f"Batch sizes of Abc2 must match: got ({batch2}, {batch2_b}, {batch2_c})")
    if mode == "zip" and batch1 != batch2:
        raise ValueError(
            f"All batch sizes must match when mode='zip': got ({batch1}, {batch1_b}, {batch1_c})"
            f" and ({batch2}, {batch2_b}, {batch2_c})"
        )
    if not nA1 == mA1 == nb1:
        raise ValueError("A1 must be square and b1 must be compatible with A1.")

    if not nA2 == mA2 == nb2:
        raise ValueError("A2 must be square and b2 must be compatible with A2.")

    # 2. get shapes and sizes

    m1 = len(poly_shape1)
    m2 = len(poly_shape2)
    n1 = nA1 - m1
    n2 = nA2 - m2

    # 3. join triples

    c1 = math.reshape(c1, (batch1, -1))
    c2 = math.reshape(c2, (batch2, -1))

    if mode == "kron":
        A1 = np.repeat(A1, batch2, axis=0)
        A2 = np.tile(A2, (batch1, 1, 1))
        A1Z = np.concatenate([A1, np.zeros((batch1 * batch2, nA1, nA2))], axis=-1)
        ZA2 = np.concatenate([np.zeros((batch1 * batch2, nA2, nA1)), A2], axis=-1)
        b1 = np.repeat(b1, batch2, axis=0)
        b2 = np.tile(b2, (batch1, 1))
        c = math.reshape(
            math.einsum("ba,dc->bdac", c1, c2),
            [batch1 * batch2] + poly_shape1 + poly_shape2,
        )
    elif mode == "zip":
        A1Z = np.concatenate([A1, np.zeros((batch1, nA1, nA2))], axis=-1)
        ZA2 = np.concatenate([np.zeros((batch1, nA2, nA1)), A2], axis=-1)
        c = math.reshape(math.einsum("ba,bc->bac", c1, c2), [batch1] + poly_shape1 + poly_shape2)

    A = np.concatenate([A1Z, ZA2], axis=-2)
    A = np.concatenate(
        [
            A[:, :n1, :],
            A[:, nA1 : nA1 + n2, :],
            A[:, n1:nA1, :],
            A[:, nA1 + n2 :, :],
        ],
        axis=-2,
    )
    A = np.concatenate(
        [
            A[:, :, :n1],
            A[:, :, nA1 : nA1 + n2],
            A[:, :, n1:nA1],
            A[:, :, nA1 + n2 :],
        ],
        axis=-1,
    )
    b = np.concatenate([b1, b2], axis=-1)
    b = np.concatenate(
        [
            b[:, :n1],
            b[:, nA1 : nA1 + n2],
            b[:, n1:nA1],
            b[:, nA1 + n2 :],
        ],
        axis=-1,
    )

    return A, b, c


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
    batched = len(A.shape) == 3 and len(b.shape) == 2 and len(c.shape) > 0
    if not batched:
        A = math.atleast_3d(A)
        b = math.atleast_2d(b)
        c = math.atleast_1d(c)
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

    not_idx = tuple(i for i in range(n_plus_N) if i not in idx)
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
    if np.all(math.abs(det_M) > 1e-12):
        inv_M = math.inv(M)
        c_post = c * math.reshape(
            math.sqrt(math.cast((-1) ** m / det_M, "complex128"))
            * math.exp(-0.5 * math.sum(bM * math.solve(M, bM), axes=[-1])),
            c.shape[:1] + (1,) * (len(c.shape) - 1),
        )
        A_post = R - math.einsum("bij,bjk,blk->bil", D, inv_M, D)
        b_post = bR - math.einsum("bij,bj->bi", D, math.solve(M, bM))
    else:
        A_post = R - math.einsum("bij,bjk,blk->bil", D, M * np.inf, D)
        b_post = bR - math.einsum("bij,bjk,bk->bi", D, M * np.inf, bM)
        c_post = math.real(c) * np.inf
    if not batched:
        return A_post[0], b_post[0], c_post[0]
    return A_post, b_post, c_post


def complex_gaussian_integral_2(
    Abc1: tuple,
    Abc2: tuple,
    idx1: Sequence[int],
    idx2: Sequence[int],
    measure: float = -1,
    mode: str = "kron",
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
        mode: how to treat the batch of the two ``(A,b,c)`` triples. Either ``kron`` for the Kronecker product or ``zip`` for parallel processing.

    Returns:
        The ``(A,b,c)`` triple which parametrizes the result of the integral with batch dimension preserved (if any).

    Raises:
        ValueError: If ``idx1`` and ``idx2`` have different lengths, or they indicate indices beyond ``n``, or if ``A``, ``b``, ``c`` have non-matching batch size.
    """
    A1, _, c1 = Abc1
    A_, b_, c_ = join_Abc(Abc1, Abc2, mode=mode)
    n1_plus_N1 = A1.shape[-1]
    N1 = len(math.atleast_1d(c1).shape[1:])
    idx2 = tuple(i + n1_plus_N1 - N1 for i in idx2)  # have to skip the first n1 variables now
    return complex_gaussian_integral_1((A_, b_, c_), idx1, idx2, measure)

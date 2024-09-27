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


def complex_gaussian_integral(
    Abc: tuple, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...], measure: float = -1
):
    r"""Computes the Gaussian integral of the exponential of a complex quadratic form.
    The integral is defined as (note that in general we integrate over a subset of 2m dimensions):

    :math:`\int_{C^m} F(z) d\mu(z)`

    where

    :math:`F(z) = \textrm{exp}(-0.5 z^T A z + b^T z)`

    Here, ``z`` is an ``n``-dim complex vector, ``A`` is an ``n x n`` complex matrix,
    ``b`` is an ``n``-dim complex vector, ``c`` is a complex scalar, and :math:`d\mu(z)`
    is a non-holomorphic complex measure over a subset of m pairs of z,z* variables. These
    are specified by the indices ``idx_z`` and ``idx_zconj``. The ``measure`` parameter is
    the exponent of the measure:

    :math: `dmu(z) = \textrm{exp}(m * |z|^2) \frac{d^{2n}z}{\pi^n} = \frac{1}{\pi^n}\textrm{exp}(m * |z|^2) d\textrm{Re}(z) d\textrm{Im}(z)`

    Note that the indices must be a complex variable pairs with each other (idx_z, idx_zconj) to make this contraction meaningful.
    Please make sure the corresponding complex variable with respect to your Abc triples.
    For examples, if the indices of Abc denotes the variables ``(\alpha, \beta, \alpha^*, \beta^*, \gamma, \eta)``, the contraction only works
    with the indices between ``(\alpha, \alpha^*)`` pairs and ``(\beta, \beta^*)`` pairs.

    Arguments:
        A,b,c: the ``(A,b,c)`` triple
        idx_z: the tuple of indices of the z variables
        idx_zconj: the tuple of indices of the z* variables
        measure: the exponent of the measure (default is -1: Bargmann measure)

    Returns:
        The ``(A,b,c)`` triple of the result of the integral.

    Raises:
        ValueError: If ``idx_z`` and ``idx_zconj`` have different lengths.
    """
    A, b, c = Abc

    if len(idx_z) != len(idx_zconj):
        raise ValueError(
            f"idx_z and idx_zconj must have the same length, got {len(idx_z)} and {len(idx_zconj)}"
        )
    n = len(idx_z)
    idx = tuple(idx_z) + tuple(idx_zconj)
    if not idx:
        return A, b, c
    not_idx = tuple(i for i in range(A.shape[-1]) if i not in idx)

    I = math.eye(n, dtype=A.dtype)
    Z = math.zeros((n, n), dtype=A.dtype)
    X = math.block([[Z, I], [I, Z]])
    M = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2) + X * measure
    bM = math.gather(b, idx, axis=-1)

    determinant = math.det(M)
    if determinant != 0:
        c_post = (
            c
            * math.sqrt((-1) ** n / determinant)
            * math.exp(-0.5 * math.sum(bM * math.solve(M, bM)))
        )
    else:
        c_post = math.real(c) * np.inf

    if math.asnumpy(not_idx).shape != (0,):
        D = math.gather(math.gather(A, idx, axis=-1), not_idx, axis=-2)
        R = math.gather(math.gather(A, not_idx, axis=-1), not_idx, axis=-2)
        bR = math.gather(b, not_idx, axis=-1)
        A_post = R - math.matmul(D, math.inv(M), math.transpose(D))
        b_post = bR - math.matvec(D, math.solve(M, bM))
    else:
        A_post = math.zeros((0, 0), dtype=A.dtype)
        b_post = math.zeros((0,), dtype=b.dtype)

    return A_post, b_post, c_post


def join_Abc(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
):
    r"""Joins two ``(A,b,c)`` triples into a single ``(A,b,c)`` triple by block addition of the ``A``
    matrices and concatenating the ``b`` vectors.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple

    Returns:
        The joined ``(A,b,c)`` triple
    """
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    c1 = math.astensor(c1)
    c2 = math.astensor(c2)
    A12 = math.block_diag(math.cast(A1, "complex128"), math.cast(A2, "complex128"))
    b12 = math.concat([math.cast(b1, "complex128"), math.cast(b2, "complex128")], axis=-1)
    c12 = math.reshape(math.outer(c1, c2), c1.shape + c2.shape)
    return A12, b12, c12


def join_Abc_real(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    idx1: Sequence[int],
    idx2: Sequence[int],
):
    r"""Direct sum of two ``(A,b,c)`` triples into a single ``(A,b,c)`` triple, where indices corresponding to the same variable are "fused together", by considering their Bargmann function has having the same variables. For example ``idx1=(0,1,2)`` and ``idx2=(1,2,3)`` means that indices 1 and 2 will be fused because they are present on both tuples. This is useful for computing real Gaussian integrals where the variable on either object is the same, rather than a pair of conjugate variables for complex Gaussian integrals.

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
    dim_poly = len(c.shape) - 1

    if order:
        if dim_poly > 0:
            if type(order) == list:
                order.extend(np.arange(len(order), len(order) + dim_poly).tolist())
            elif type(order) == tuple:
                order = order + tuple(np.arange(len(order), len(order) + dim_poly))
        A = math.gather(math.gather(A, order, axis=-1), order, axis=-2)
        b = math.gather(b, order, axis=-1)
    return A, b, c


def contract_two_Abc(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    idx1: Sequence[int],
    idx2: Sequence[int],
):
    r"""
    Returns the contraction of two ``(A,b,c)`` triples with given indices.

    Note that the indices must be a complex variable pairs with each other to make this contraction meaningful. Please make sure
    the corresponding complex variable with respect to your Abc triples.
    For examples, if the indices of Abc1 denotes the variables ``(\alpha, \beta)``, the indices of Abc2 denotes the variables
    ``(\alpha^*,\gamma)``, the contraction only works with ``idx1 = [0], idx2 = [0]``.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple
        idx1: the indices of the first ``(A,b,c)`` triple to contract
        idx2: the indices of the second ``(A,b,c)`` triple to contract

    Returns:
        The contracted ``(A,b,c)`` triple
    """
    Abc = join_Abc(Abc1, Abc2)
    return complex_gaussian_integral(
        Abc, idx1, tuple(n + Abc1[0].shape[-1] for n in idx2), measure=-1.0
    )


def join_Abc_poly(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
):
    r"""Joins two ``(A,b,c)`` triples into a single ``(A,b,c)`` triple by block addition of the ``A``
    matrices and concatenating the ``b`` vectors.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple

    Returns:
        The joined ``(A,b,c)`` triple
    """
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2

    A1 = math.cast(A1, "complex128")
    A2 = math.cast(A2, "complex128")

    c1 = math.astensor(c1)
    c2 = math.astensor(c2)

    dim_n1 = len(c1.shape)
    dim_n2 = len(c2.shape)

    dim_m1 = A1.shape[-1] - dim_n1
    dim_m2 = A2.shape[-1] - dim_n2

    A12 = math.block(
        [
            [
                A1[:dim_m1, :dim_m1],
                math.zeros((dim_m1, dim_m2), dtype=A1.dtype),
                A1[:dim_m1, dim_m1:],
                math.zeros((dim_m1, dim_n2), dtype=A1.dtype),
            ],
            [
                math.zeros((dim_m2, dim_m1), dtype=A1.dtype),
                A2[:dim_m2:, :dim_m2],
                math.zeros((dim_m2, dim_n1), dtype=A1.dtype),
                A2[:dim_m2, dim_m2:],
            ],
            [
                A1[dim_m1:, :dim_m1],
                math.zeros((dim_n1, dim_m2), dtype=A1.dtype),
                A1[dim_m1:, dim_m1:],
                math.zeros((dim_n1, dim_n2), dtype=A1.dtype),
            ],
            [
                math.zeros((dim_n2, dim_m1), dtype=A1.dtype),
                A2[dim_m2:, :dim_m2],
                math.zeros((dim_n2, dim_n1), dtype=A1.dtype),
                A2[dim_m2:, dim_m2:],
            ],
        ]
    )
    b12 = math.concat((b1[:dim_m1], b2[:dim_m2], b1[dim_m1:], b2[dim_m2:]), axis=-1)
    c12 = math.reshape(math.outer(c1, c2), c1.shape + c2.shape)
    return A12, b12, c12


def contract_two_Abc_poly(
    Abc1: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    Abc2: tuple[ComplexMatrix, ComplexVector, ComplexTensor],
    idx1: Sequence[int],
    idx2: Sequence[int],
):
    r"""
    Returns the contraction of two ``(A,b,c)`` triples with given indices.

    Note that the indices must be a complex variable pairs with each other to make this contraction meaningful. Please make sure
    the corresponding complex variable with respect to your Abc triples.
    For examples, if the indices of Abc1 denotes the variables ``(\alpha, \beta)``, the indices of Abc2 denotes the variables
    ``(\alpha^*,\gamma)``, the contraction only works with ``idx1 = [0], idx2 = [0]``.

    Arguments:
        Abc1: the first ``(A,b,c)`` triple
        Abc2: the second ``(A,b,c)`` triple
        idx1: the indices of the first ``(A,b,c)`` triple to contract
        idx2: the indices of the second ``(A,b,c)`` triple to contract

    Returns:
        The contracted ``(A,b,c)`` triple
    """
    Abc = join_Abc_poly(Abc1, Abc2)

    dim_n1 = len(Abc1[2].shape)
    return complex_gaussian_integral(
        Abc, idx1, tuple(n + Abc1[0].shape[-1] - dim_n1 for n in idx2), measure=-1.0
    )


def complex_gaussian_integral_2(
    Abc1: tuple,
    Abc2: tuple,
    idx1: Sequence[int],
    idx2: Sequence[int],
    measure: float = -1,
    batched: bool = False,
):
    r"""Computes the (optionally batched) Gaussian integral of the product of two exponential functions of complex quadratic forms with
    respect to an exponential measure. The first dimension of the ``Abc`` parameters is batched over. The ``c`` parameter
    is allowed to have additional dimensions which are to be combined using an outer product.
    When ``batched=True``, this function expects ``A1.shape=(b,n1,n1)``, ``A2.shape=(b,n2,n2)``, ``b1.shape=(b,n1)``, ``b2.shape=(b,n2)``,
    ``c1.shape=(b,d1*)``, ``c2.shape=(b,d2*)``. The output is the A,b,c parametrization of the exponential of a quadratic form
    with parameters of shape ``A.shape=(b,n1+n2-2m,n1+n2-2m)``, ``b.shape=(b,n1+n2-2m)``, ``c.shape=(b,d1*,d2*)``.
    When ``batched=False``, this function expects ``A1.shape=(n1,n1)``, ``A2.shape=(n2,n2)``, ``b1.shape=(n1,)``, ``b2.shape=(n2,)``,
    ``c1.shape=(d1*)``, ``c2.shape=(d2*)``. The output is the A,b,c parametrization of the exponential of a quadratic form
    with parameters of shape ``A.shape=(n1+n2-2m,n1+n2-2m)``, ``b.shape=(n1+n2-2m)``, ``c.shape=(d1*,d2*)``.

    The integral is defined as

    :math:`\int_{C^m} F_1(z)F_2(z^*) d\mu(z)`

    where the functions ``F_1`` and ``F_2`` are defined over ``n_1`` and ``n_2`` variables respectively, and the integral is taken
    over a subset of  ``m`` pairs of variables whose position is indicated by ``idx1`` and ``idx2``.
    Therefore the resulting function is a new exponential of a quadratic form defined over ``n_1 + n_2 - 2m`` variables.

    The functions are assumed to be in the form
    :math:`F_i(z) = c_i\textrm{exp}(0.5 z^T A_i z + b_i^T z),\quad z\in\mathbb{C}^{n_i}`

    where ``A_i`` is complex symmetric

    The integration measure is given by (``s`` here is the ``measure`` argument):
    :math:`dmu(z) = \textrm{exp}(s * |z|^2) \frac{d^{2n}z}{\pi^n} = \frac{1}{\pi^n}\textrm{exp}(s * |z|^2) d\textrm{Re}(z) d\textrm{Im}(z)`

    Arguments:
        A1,b1,c1: the ``(A,b,c)`` triple that defines ``F_1(z)``
        A2,b2,c2: the ``(A,b,c)`` triple that defines ``F_2(z^*)``
        idx1: the tuple of indices indicating which variables of F_1 to integrate over
        idx2: the tuple of indices indicating which variables of F_2 to integrate over
        measure: the exponent of the measure (default is -1: Bargmann measure)

    Returns:
        The ``(A,b,c)`` triple of the result of the integral.

    Raises:
        ValueError: If ``idx1`` and ``idx2`` have different lengths or if A_i, b_i and c_i have non-matching batch size.
    """
    # first we ensure that Ai, bi, ci have the correct shapes
    A1, b1, c1 = Abc1
    A2, b2, c2 = Abc2
    c1 = math.astensor(c1)
    c2 = math.astensor(c2)
    if not batched:
        A1 = math.atleast_3d(A1)
        A2 = math.atleast_3d(A2)
        b1 = math.atleast_2d(b1)
        b2 = math.atleast_2d(b2)
        c1 = math.atleast_1d(c1)
        c2 = math.atleast_1d(c2)

    if len(idx1) != len(idx2):
        raise ValueError(
            f"idx1 and idx2 must have the same length, got {len(idx1)} and {len(idx2)}"
        )
    if not A1.shape[0] == b1.shape[0] == c1.shape[0] == A2.shape[0] == b2.shape[0] == c2.shape[0]:
        raise ValueError(
            f"Batch size mismatch: got {A1.shape[0]} for A1, {b1.shape[0]} for b1, {c1.shape[0]} for c1, {A2.shape[0]} for A2, {b2.shape[0]} for b2, {c2.shape[0]} for c2."
        )

    # get various sizes for later
    (batch, n1, _) = A1.shape
    (_, n2, _) = A2.shape
    d1 = c1.shape[1:]
    d2 = c2.shape[1:]
    m = len(idx1)  # number of variable pairs to integrate over
    idx = tuple(idx1) + tuple(i + n1 for i in idx2)

    A = math.zeros((batch, n1 + n2, n1 + n2), dtype=A1.dtype)
    for i in range(batch):
        A[i] = math.block_diag(A1[i], A2[i])
    b = math.concat([b1, b2], axis=-1)
    c = math.einsum("bj,bk->bjk", math.reshape(c1, (batch, -1)), math.reshape(c2, (batch, -1)))
    c = math.reshape(c, (batch,) + d1 + d2)

    if len(idx) == 0:
        return A, b, c

    not_idx = tuple(i for i in range(n1 + n2) if i not in idx)

    I = math.eye(m, dtype=A.dtype)
    Z = math.zeros((m, m), dtype=A.dtype)
    X = math.block([[Z, I], [I, Z]])
    M = math.gather(math.gather(A, idx, axis=-1), idx, axis=-2) + X * measure
    bM = math.gather(b, idx, axis=-1)
    cpart1 = math.sqrt(math.cast((-1) ** m / math.det(M), "complex128"))
    cpart2 = math.exp(-0.5 * math.sum(bM * math.solve(M, bM), axes=[-1]))
    c_post = math.einsum("b...,b->b...", c, cpart1 * cpart2)
    D = math.gather(math.gather(A, idx, axis=-1), not_idx, axis=-2)
    R = math.gather(math.gather(A, not_idx, axis=-1), not_idx, axis=-2)
    bR = math.gather(b, not_idx, axis=-1)
    A_post = R - math.einsum("bij,bjk,blk->bil", D, math.inv(M), D)
    b_post = bR - math.einsum("bij,bj->bi", D, math.solve(M, bM))

    if not batched:
        return A_post[0], b_post[0], c_post[0]

    return A_post, b_post, c_post

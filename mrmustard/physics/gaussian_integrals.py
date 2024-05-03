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
from typing import Sequence, Tuple
import numpy as np

from mrmustard import math
from mrmustard.utils.typing import ComplexMatrix, ComplexVector


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
        * math.sqrt((2 * np.pi) ** len(idx) / math.det(M))
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

    not_idx = tuple(i for i in range(A.shape[-1]) if i not in idx)
    if math.asnumpy(not_idx).shape != (0,):
        D = math.gather(math.gather(A, idx, axis=-1), not_idx, axis=-2)
        R = math.gather(math.gather(A, not_idx, axis=-1), not_idx, axis=-2)
        bR = math.gather(b, not_idx, axis=-1)
        A_post = R - math.matmul(D, math.inv(M), math.transpose(D))
        b_post = bR - math.matvec(D, math.solve(M, bM))
    else:
        A_post = math.astensor([])
        b_post = math.astensor([])

    c_post = (
        c * math.sqrt((-1) ** n / math.det(M)) * math.exp(-0.5 * math.sum(bM * math.solve(M, bM)))
    )

    return A_post, b_post, c_post


def join_Abc(
    Abc1: Tuple[ComplexMatrix, ComplexVector, complex],
    Abc2: Tuple[ComplexMatrix, ComplexVector, complex],
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
    A12 = math.block_diag(math.cast(A1, "complex128"), math.cast(A2, "complex128"))
    b12 = math.concat([b1, b2], axis=-1)
    c12 = math.outer(c1, c2)
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
    if order:
        A = math.gather(math.gather(A, order, axis=-1), order, axis=-2)
        b = math.gather(b, order, axis=-1)
        if len(c.shape) == len(order):
            c = math.transpose(c, order)
    return A, b, c


def contract_two_Abc(
    Abc1: Tuple[ComplexMatrix, ComplexVector, complex],
    Abc2: Tuple[ComplexMatrix, ComplexVector, complex],
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

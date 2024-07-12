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

"""
This module contains the classes for the available ansatze.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, Union, Optional

import numpy as np

from mrmustard import math, settings
from mrmustard.utils.argsort import argsort_gen
from mrmustard.utils.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    Matrix,
    Scalar,
    Tensor,
    Vector,
)

__all__ = [
    "Ansatz",
    "ArrayAnsatz",
    "PolyExpBase",
    "PolyExpAnsatz",
    "DiffOpPolyExpAnsatz",
]


class Ansatz(ABC):
    r"""
    A function over a continuous and/or discrete domain.

    An ansatz supports basic mathematical operations such as addition, subtraction,
    multiplication, division, negation, equality, etc.

    Note that ``n``-dimensional arrays are like functions defined over an integer lattice of points,
    so this class also works for, e.g., the Fock representation.

    This class is abstract. Concrete ``Ansatz`` classes have to implement the
    ``__call__``, ``__mul__``, ``__add__``, ``__sub__``, ``__neg__``, and ``__eq__`` methods.
    """

    @abstractmethod
    def __neg__(self) -> Ansatz:
        r"""
        Negates this ansatz.
        """

    @abstractmethod
    def __eq__(self, other: Ansatz) -> bool:
        r"""
        Whether this ansatz is equal to another ansatz.
        """

    @abstractmethod
    def __add__(self, other: Ansatz) -> Ansatz:
        r"""
        Sums this ansatz to another ansatz.
        """

    def __sub__(self, other: Ansatz) -> Ansatz:
        r"""
        Subtracts other from this ansatz.
        """
        try:
            return self.__add__(-other)
        except AttributeError as e:
            raise TypeError(f"Cannot subtract {self.__class__} and {other.__class__}.") from e

    @abstractmethod
    def __call__(self, point: Any) -> Scalar:
        r"""
        Evaluates this ansatz at a given point in the domain.
        """

    @abstractmethod
    def __truediv__(self, other: Union[Scalar, Ansatz]) -> Ansatz:
        r"""
        Divides this ansatz by another ansatz or by a scalar.
        """

    @abstractmethod
    def __mul__(self, other: Union[Scalar, Ansatz]) -> Ansatz:
        r"""
        Multiplies this ansatz by another ansatz.
        """

    @abstractmethod
    def __and__(self, other: Ansatz) -> Ansatz:
        r"""
        Tensor product of this ansatz with another ansatz.
        """

    def __rmul__(self, other: Scalar) -> Ansatz:
        r"""
        Multiplies this ansatz by a scalar.
        """
        return self * other


class PolyExpBase(Ansatz):
    r"""
    A family of Ansatze parametrized by a triple of a matrix, a vector and an array.
    For example, the Bargmann representation :math:`c\:\textrm{exp}(z A z / 2 + b z)` is of this
    form (where ``A``, ``b``, ``c`` is the triple), or the characteristic function of the
    Wigner representation (where ``Sigma``, ``mu``, ``1`` is the triple).

    Note that this class is not initializable (despite having an initializer) because it does
    not implement all the abstract methods of ``Ansatz``, and it is in fact more general.
    Concrete ansatze that inherit from this class need to implement ``__call__``,
    ``__mul__`` and ``__matmul__``, which are representation-specific.

    Note that the arguments are expected to be batched, i.e. to have a batch dimension
    or to be an iterable. This is because this class also provides the linear superposition
    functionality by implementing the ``__add__`` method, which concatenates the batch dimensions.

    As this can blow up the number of terms in the representation, it is recommended to
    run the `simplify()` method after adding terms together, which combines together
    terms that have the same exponential part.

    Args:
        mat: the matrix-like data
        vec: the vector-like data
        array: the array-like data
    """

    def __init__(self, mat: Batch[Matrix], vec: Batch[Vector], array: Batch[Tensor]):
        self.mat = math.atleast_3d(math.astensor(mat))
        self.vec = math.atleast_2d(math.astensor(vec))
        self.array = math.atleast_1d(math.astensor(array))
        self.batch_size = self.mat.shape[0]
        self.num_vars = self.mat.shape[-1]
        self._simplified = False

    def __neg__(self) -> PolyExpBase:
        return self.__class__(self.mat, self.vec, -self.array)

    def __eq__(self, other: PolyExpBase) -> bool:
        return self._equal_no_array(other) and np.allclose(self.array, other.array, atol=1e-10)

    def _equal_no_array(self, other: PolyExpBase) -> bool:
        self.simplify()
        other.simplify()
        return np.allclose(self.vec, other.vec, atol=1e-10) and np.allclose(
            self.mat, other.mat, atol=1e-10
        )

    def __add__(self, other: PolyExpBase) -> PolyExpBase:
        combined_matrices = math.concat([self.mat, other.mat], axis=0)
        combined_vectors = math.concat([self.vec, other.vec], axis=0)
        combined_arrays = math.concat([self.array, other.array], axis=0)
        # note output is not simplified
        return self.__class__(combined_matrices, combined_vectors, combined_arrays)

    @property
    def degree(self) -> int:
        r"""
        The degree of this ansatz.
        """
        if self.array.ndim == 1:
            return 0
        return self.array.shape[-1] - 1

    @property
    def polynomial_degrees(self) -> tuple[int, tuple]:
        r"""
        This method finds the dimensionality of the polynomial, i.e. how many wires
        have polynomials attached to them and what the degree of the polynomial is
        on each of the wires.
        """

        dim_poly = len(self.array.shape) - 1
        shape_poly = self.array.shape[1:]
        return dim_poly, shape_poly

    def simplify(self) -> None:
        r"""
        Simplifies the representation by combining together terms that have the same
        exponential part, i.e. two terms along the batch are considered equal if their
        matrix and vector are equal. In this case only one is kept and the arrays are added.

        Does not run if the representation has already been simplified, so it is safe to call.
        """
        if self._simplified:
            return
        indices_to_check = set(range(self.batch_size))
        removed = []
        while indices_to_check:
            i = indices_to_check.pop()
            for j in indices_to_check.copy():
                if np.allclose(self.mat[i], self.mat[j]) and np.allclose(self.vec[i], self.vec[j]):
                    self.array = math.update_add_tensor(self.array, [[i]], [self.array[j]])
                    indices_to_check.remove(j)
                    removed.append(j)
        to_keep = [i for i in range(self.batch_size) if i not in removed]
        self.mat = math.gather(self.mat, to_keep, axis=0)
        self.vec = math.gather(self.vec, to_keep, axis=0)
        self.array = math.gather(self.array, to_keep, axis=0)
        self._simplified = True

    def simplify_v2(self) -> None:
        r"""
        A different implementation of ``simplify`` that orders the batch dimension first.
        """
        if self._simplified:
            return
        self._order_batch()
        to_keep = [d0 := 0]
        mat, vec = self.mat[d0], self.vec[d0]
        for d in range(1, self.batch_size):
            if np.allclose(mat, self.mat[d]) and np.allclose(vec, self.vec[d]):
                self.array = math.update_add_tensor(self.array, [[d0]], [self.array[d]])
            else:
                to_keep.append(d)
                d0 = d
                mat, vec = self.mat[d0], self.vec[d0]
        self.mat = math.gather(self.mat, to_keep, axis=0)
        self.vec = math.gather(self.vec, to_keep, axis=0)
        self.array = math.gather(self.array, to_keep, axis=0)
        self._simplified = True

    def _order_batch(self):
        r"""
        This method orders the batch dimension by the lexicographical order of the
        flattened arrays (mat, vec, array). This is a very cheap way to enforce
        an ordering of the batch dimension, which is useful for simplification and for
        determining (in)equality between two Bargmann representations.
        """
        generators = [
            itertools.chain(
                math.asnumpy(self.vec[i]).flat,
                math.asnumpy(self.mat[i]).flat,
                math.asnumpy(self.array[i]).flat,
            )
            for i in range(self.batch_size)
        ]
        sorted_indices = argsort_gen(generators)
        self.mat = math.gather(self.mat, sorted_indices, axis=0)
        self.vec = math.gather(self.vec, sorted_indices, axis=0)
        self.array = math.gather(self.array, sorted_indices, axis=0)

    def decompose_ansatz(self) -> DiffOpPolyExpAnsatz:
        r"""
        This method decomposes a DiffOpPolyExpAnsatz. Given an ansatz of dimensions:
        A=(batch,m+n,m+n), b=(batch,m+n), c = (batch,k_1,k_2,...,k_n),
        it can be rewritten as an ansatz of dimensions
        A=(batch,2m,2m), b=(batch,2m), c = (batch,l_1,l_2,...,l_m), with l_i = sum_j k_j
        This decomposition is typically favourable if n>m, and will only run if that is the case.
        """
        dim_beta, shape_beta = self.polynomial_degrees
        dim_alpha = self.mat.shape[-1] - dim_beta
        batch_size = self.batch_size
        if dim_beta > dim_alpha:
            A_bar = np.array(
                [
                    np.block(
                        [
                            [
                                np.zeros((dim_alpha, dim_alpha)),
                                self.mat[i, dim_alpha:, :dim_alpha].T,
                            ],
                            [
                                self.mat[i, dim_alpha:, :dim_alpha],
                                self.mat[i, dim_alpha:, dim_alpha:],
                            ],
                        ]
                    )
                    for i in range(batch_size)
                ]
            )

            b_bar = np.array(
                [
                    np.concatenate((np.zeros(dim_alpha), self.vec[i, dim_alpha:]))
                    for i in range(batch_size)
                ]
            )

            poly_bar = math.hermite_renormalized_batch(
                np.moveaxis(A_bar, 0, -1),
                np.moveaxis(b_bar, 0, -1),
                1,
                (np.sum(shape_beta),) * dim_alpha + shape_beta + (batch_size,),
            )
            poly_bar = np.moveaxis(poly_bar, -1, dim_alpha)
            c_decomp = np.sum(
                poly_bar * self.array,
                axis=tuple(np.arange(len(poly_bar.shape) - dim_beta, len(poly_bar.shape))),
            )
            c_decomp = np.moveaxis(c_decomp, -1, 0)
            A_decomp = np.array(
                [
                    np.block(
                        [
                            [self.mat[i, :dim_alpha, :dim_alpha], np.eye(dim_alpha)],
                            [np.eye(dim_alpha), np.zeros((dim_alpha, dim_alpha))],
                        ]
                    )
                    for i in range(batch_size)
                ]
            )
            b_decomp = np.array(
                [
                    np.concatenate((self.vec[i, :dim_alpha], np.zeros(dim_alpha)))
                    for i in range(batch_size)
                ]
            )
            return DiffOpPolyExpAnsatz(A_decomp, b_decomp, c_decomp)
        else:
            return DiffOpPolyExpAnsatz(self.mat, self.vec, self.array)


class PolyExpAnsatz(PolyExpBase):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz function:

        :math:`F(z) = \sum_i \textrm{poly}_i(z) \textrm{exp}(z^T A_i z / 2 + z^T b_i)`

    where each :math:`poly_i` is a polynomial in ``z`` that can be expressed as

        :math:`\textrm{poly}_i(z) = \sum_k c^(i)_k z^k`,

    with ``k`` being a multi-index. The matrices :math:`A_i` and vectors :math:`b_i` are
    parameters of the exponential terms in the ansatz, and :math:`z` is a vector of variables.

    .. code-block::

        >>> from mrmustard.physics.ansatze import PolyExpAnsatz

        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 1.0])
        >>> c = np.array(1.0)

        >>> F = PolyExpAnsatz(A, b, c)
        >>> z = np.array([1.0, 2.0])

        >>> # calculate the value of the function at ``z``
        >>> val = F(z)

    Args:
        A: The list of square matrices :math:`A_i`
        b: The list of vectors :math:`b_i`
        c: The array of coefficients for the polynomial terms in the ansatz.

    """

    def __init__(
        self,
        A: Optional[Batch[Matrix]] = None,
        b: Optional[Batch[Vector]] = None,
        c: Batch[Tensor | Scalar] = 1.0,
        name: str = "",
    ):
        self.name = name

        if A is None and b is None:
            raise ValueError("Please provide either A or b.")
        super().__init__(mat=A, vec=b, array=c)

    @property
    def A(self) -> Batch[ComplexMatrix]:
        r"""
        The list of square matrices :math:`A_i`.
        """
        return self.mat

    @property
    def b(self) -> Batch[ComplexVector]:
        r"""
        The list of vectors :math:`b_i`.
        """
        return self.vec

    @property
    def c(self) -> Batch[ComplexTensor]:
        r"""
        The array of coefficients for the polynomial terms in the ansatz.
        """
        return self.array

    def __call__(self, z: Batch[Vector]) -> Scalar:
        r"""
        Value of this ansatz at ``z``.

        Args:
            z: point in C^n where the function is evaluated

        Returns:
            The value of the function.
        """
        z = np.atleast_2d(z)  # shape (..., n)
        zz = np.einsum("...a,...b->...ab", z, z)[..., None, :, :]  # shape (..., 1, n, n))
        A_part = 0.5 * math.sum(
            zz * self.A, axes=[-1, -2]
        )  # sum((...,1,n,n) * (b,n,n), [-1,-2]) ~ (...,b)
        b_part = np.sum(z[..., None, :] * self.b, axis=-1)  # sum((...,1,n) * (b,n), -1) ~ (...,b)
        exp_sum = np.exp(A_part + b_part)  # (..., b)
        result = exp_sum * self.c  # (..., b)
        val = np.sum(result, axis=-1)  # (...)
        return val

    def __mul__(self, other: Union[Scalar, PolyExpAnsatz]) -> PolyExpAnsatz:
        r"""
        Multiplies this ansatz by a scalar or another ansatz.

        Args:
            other: A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            PolyExpAnsatz: The product of this ansatz and other.
        """
        if isinstance(other, PolyExpAnsatz):
            new_a = [A1 + A2 for A1, A2 in itertools.product(self.A, other.A)]
            new_b = [b1 + b2 for b1, b2 in itertools.product(self.b, other.b)]
            new_c = [c1 * c2 for c1, c2 in itertools.product(self.c, other.c)]
            return self.__class__(A=new_a, b=new_b, c=new_c)
        else:
            try:
                return self.__class__(self.A, self.b, self.c * other)
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __truediv__(self, other: Union[Scalar, PolyExpAnsatz]) -> PolyExpAnsatz:
        r"""
        Divides this ansatz by a scalar or another ansatz.

        Args:
            other: A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            PolyExpAnsatz: The division of this ansatz by other.
        """
        if isinstance(other, PolyExpAnsatz):
            new_a = [A1 - A2 for A1, A2 in itertools.product(self.A, other.A)]
            new_b = [b1 - b2 for b1, b2 in itertools.product(self.b, other.b)]
            new_c = [c1 / c2 for c1, c2 in itertools.product(self.c, other.c)]
            return self.__class__(A=new_a, b=new_b, c=new_c)
        else:
            try:
                return self.__class__(self.A, self.b, self.c / other)
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""
        Tensor product of this ansatz with another ansatz.
        Equivalent to :math:`F(a) * G(b)` (with different arguments, that is).
        As it distributes over addition on both self and other,
        the batch size of the result is the product of the batch
        size of this anzatz and the other one.

        Args:
            other: Another ansatz.

        Returns:
            The tensor product of this ansatz and other.
        """
        As = [math.block_diag(a1, a2) for a1 in self.A for a2 in other.A]
        bs = [math.concat([b1, b2], axis=-1) for b1 in self.b for b2 in other.b]
        cs = [math.outer(c1, c2) for c1 in self.c for c2 in other.c]
        return self.__class__(As, bs, cs)


class DiffOpPolyExpAnsatz(PolyExpBase):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz function:

        :math:`F(z) = \sum_i [\sum_k c^(i)_k \partial_y^k \textrm{exp}((z,y)^T A_i (z,y) / 2 + (z,y)^T b_i)|_{y=0}]`

    with ``k`` being a multi-index. The matrices :math:`A_i` and vectors :math:`b_i` are
    parameters of the exponential terms in the ansatz, and :math:`z` is a vector of variables.

    .. code-block::

        >>> from mrmustard.physics.ansatze import DiffOpPolyExpAnsatz

        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 1.0])
        >>> c = np.array([1.0,2.0,3.0])

        >>> F = PolyExpAnsatz(A, b, c)
        >>> z = np.array([1.0, 2.0])

        >>> # calculate the value of the function at ``z``
        >>> val = F(z)

    Args:
        A: The list of square matrices :math:`A_i`
        b: The list of vectors :math:`b_i`
        c: The array of coefficients for the polynomial terms in the ansatz.

    """

    def __init__(
        self,
        A: Optional[Batch[Matrix]] = None,
        b: Optional[Batch[Vector]] = None,
        c: Batch[Tensor] = np.array([1.0]),
        name: str = "",
    ):
        self.name = name

        if A is None and b is None:
            raise ValueError("Please provide either A or b.")
        A = math.astensor(A)
        b = math.astensor(b)
        c = math.astensor(c)
        super().__init__(mat=A, vec=b, array=c)

    @property
    def A(self) -> Batch[ComplexMatrix]:
        r"""
        The list of square matrices :math:`A_i`.
        """
        return self.mat

    @property
    def b(self) -> Batch[ComplexVector]:
        r"""
        The list of vectors :math:`b_i`.
        """
        return self.vec

    @property
    def c(self) -> Batch[ComplexTensor]:
        r"""
        The array of coefficients for the polynomial terms in the ansatz.
        """
        return self.array

    def __call__(self, z: Batch[Vector]) -> Scalar:
        r"""
        Value of this ansatz at ``z``.

        Args:
            z: point in C^n where the function is evaluated

        Returns:
            The value of the function.
        """
        dim_beta, shape_beta = self.polynomial_degrees
        dim_alpha = self.A.shape[-1] - dim_beta
        zz = np.einsum("...a,...b->...ab", z, z)[..., None, :, :]
        z = z[..., None, :]

        A_part = np.sum(self.A[..., :dim_alpha, :dim_alpha] * zz, axis=(-1, -2))
        b_part = np.sum(self.b[..., :dim_alpha] * z[..., None, :], axis=-1)

        exp_sum = np.exp(1 / 2 * A_part + b_part)
        if dim_beta == 0:
            val = np.sum(exp_sum * self.c, axis=-1)
        else:
            b_poly = math.astensor(
                [
                    np.sum(self.A[..., dim_alpha:, :dim_alpha] * z[i, None, :], axis=-1)
                    + self.b[..., dim_alpha:]
                    for i in range(z.shape[0])
                ]
            )
            b_poly = np.moveaxis(b_poly, 0, -1)
            A_poly = self.A[..., dim_alpha:, dim_alpha:]
            poly = math.astensor(
                [
                    math.hermite_renormalized_batch(
                        A_poly[i], b_poly[i], 1, shape_beta + (b_poly.shape[-1],)
                    )
                    for i in range(A_poly.shape[0])
                ]
            )
            poly = np.moveaxis(poly, -1, 0)
            val = np.sum(
                exp_sum * np.sum(poly * self.c, axis=tuple(np.arange(2, 2 + dim_beta))), axis=-1
            )
        return val

    def __mul__(self, other: Union[Scalar, DiffOpPolyExpAnsatz]) -> DiffOpPolyExpAnsatz:
        r"""Multiplies this ansatz by a scalar or another ansatz or a plain scalar.

        Args:
            other: A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            PolyExpAnsatz: The product of this ansatz and other.
        """

        def mulA(A1, A2, dim_alpha, dim_beta1, dim_beta2):
            A3 = np.block(
                [
                    [
                        A1[:dim_alpha, :dim_alpha] + A2[:dim_alpha, :dim_alpha],
                        A1[:dim_alpha, dim_alpha:],
                        A2[:dim_alpha, dim_alpha:],
                    ],
                    [
                        A1[dim_alpha:, :dim_alpha],
                        A1[dim_alpha:, dim_alpha:],
                        np.zeros((dim_beta1, dim_beta2)),
                    ],
                    [
                        A2[dim_alpha:, :dim_alpha],
                        np.zeros((dim_beta2, dim_beta1)),
                        A2[dim_alpha:, dim_alpha:],
                    ],
                ]
            )
            return A3

        def mulb(b1, b2, dim_alpha):
            b3 = np.concatenate((b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]))
            return b3

        def mulc(c1, c2):
            c3 = np.outer(c1, c2).reshape(c1.shape + c2.shape)
            return c3

        if isinstance(other, DiffOpPolyExpAnsatz):

            dim_beta1, _ = self.polynomial_degrees
            dim_beta2, _ = other.polynomial_degrees

            dim_alpha1 = self.A.shape[-1] - dim_beta1
            dim_alpha2 = other.A.shape[-1] - dim_beta2
            assert dim_alpha1 == dim_alpha2
            dim_alpha = dim_alpha1

            new_a = [
                mulA(A1, A2, dim_alpha, dim_beta1, dim_beta2)
                for A1, A2 in itertools.product(self.A, other.A)
            ]
            new_b = [mulb(b1, b2, dim_alpha) for b1, b2 in itertools.product(self.b, other.b)]
            new_c = [mulc(c1, c2) for c1, c2 in itertools.product(self.c, other.c)]

            return self.__class__(A=new_a, b=new_b, c=new_c)
        else:
            try:
                return self.__class__(self.A, self.b, self.c * other)
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e

    def __truediv__(self, other: Union[Scalar, DiffOpPolyExpAnsatz]) -> DiffOpPolyExpAnsatz:
        r"""Multiplies this ansatz by a scalar or another ansatz or a plain scalar.

        Args:
            other: A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            PolyExpAnsatz: The product of this ansatz and other.
        """

        def divA(A1, A2, dim_alpha, dim_beta1, dim_beta2):
            A3 = np.block(
                [
                    [
                        A1[:dim_alpha, :dim_alpha] + A2[:dim_alpha, :dim_alpha],
                        A1[:dim_alpha, dim_alpha:],
                        A2[:dim_alpha, dim_alpha:],
                    ],
                    [
                        A1[dim_alpha:, :dim_alpha],
                        A1[dim_alpha:, dim_alpha:],
                        np.zeros((dim_beta1, dim_beta2)),
                    ],
                    [
                        A2[dim_alpha:, :dim_alpha],
                        np.zeros((dim_beta2, dim_beta1)),
                        A2[dim_alpha:, dim_alpha:],
                    ],
                ]
            )
            return A3

        def divb(b1, b2, dim_alpha):
            b3 = np.concatenate((b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]))
            return b3

        def divc(c1, c2):
            c3 = np.outer(c1, c2).reshape(c1.shape + c2.shape)
            return c3

        if isinstance(other, DiffOpPolyExpAnsatz):

            dim_beta1, _ = self.polynomial_degrees
            dim_beta2, _ = other.polynomial_degrees
            if dim_beta1 == 0 and dim_beta2 == 0:
                dim_alpha1 = self.A.shape[-1] - dim_beta1
                dim_alpha2 = other.A.shape[-1] - dim_beta2
                assert dim_alpha1 == dim_alpha2
                dim_alpha = dim_alpha1

                new_a = [
                    divA(A1, -A2, dim_alpha, dim_beta1, dim_beta2)
                    for A1, A2 in itertools.product(self.A, other.A)
                ]
                new_b = [divb(b1, -b2, dim_alpha) for b1, b2 in itertools.product(self.b, other.b)]
                new_c = [divc(c1, 1 / c2) for c1, c2 in itertools.product(self.c, other.c)]

                return self.__class__(A=new_a, b=new_b, c=new_c)
            else:
                raise NotImplementedError("Only implemented if both c are scalars")
        else:
            try:
                return self.__class__(self.A, self.b, self.c / other)
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: DiffOpPolyExpAnsatz) -> DiffOpPolyExpAnsatz:
        r"""Tensor product of this ansatz with another ansatz.
        Equivalent to :math:`F(a) * G(b)` (with different arguments, that is).
        As it distributes over addition on both self and other,
        the batch size of the result is the product of the batch
        size of this anzatz and the other one.

        Args:
            other: Another ansatz.

        Returns:
            The tensor product of this ansatz and other.
        """

        def andA(A1, A2, dim_alpha1, dim_alpha2, dim_beta1, dim_beta2):
            A3 = np.block(
                [
                    [
                        A1[:dim_alpha1, :dim_alpha1],
                        np.zeros((dim_alpha1, dim_alpha2)),
                        A1[:dim_alpha1, dim_alpha1:],
                        np.zeros((dim_alpha1, dim_beta2)),
                    ],
                    [
                        np.zeros((dim_alpha2, dim_alpha1)),
                        A2[:dim_alpha2:, :dim_alpha2],
                        np.zeros((dim_alpha2, dim_beta1)),
                        A2[:dim_alpha2, dim_alpha2:],
                    ],
                    [
                        A1[dim_alpha1:, :dim_alpha1],
                        np.zeros((dim_beta1, dim_alpha2)),
                        A1[dim_alpha1:, dim_alpha1:],
                        np.zeros((dim_beta1, dim_beta2)),
                    ],
                    [
                        np.zeros((dim_beta2, dim_alpha1)),
                        A2[dim_alpha2:, :dim_alpha2],
                        np.zeros((dim_beta2, dim_beta1)),
                        A2[dim_alpha2:, dim_alpha2:],
                    ],
                ]
            )
            return A3

        def andb(b1, b2, dim_alpha1, dim_alpha2):
            b3 = np.concatenate(
                (b1[:dim_alpha1], b2[:dim_alpha2], b1[dim_alpha1:], b2[dim_alpha2:])
            )
            return b3

        def andc(c1, c2):
            if c1.shape == (1,) and c2.shape == (1,):
                c3 = c1 * c2
            else:
                c3 = np.outer(c1, c2).reshape(c1.shape + c2.shape)
            return c3

        dim_beta1, _ = self.polynomial_degrees
        dim_beta2, _ = other.polynomial_degrees

        dim_alpha1 = self.A.shape[-1] - dim_beta1
        dim_alpha2 = other.A.shape[-1] - dim_beta2

        As = [
            andA(A1, A2, dim_alpha1, dim_alpha2, dim_beta1, dim_beta2)
            for A1, A2 in itertools.product(self.A, other.A)
        ]
        bs = [andb(b1, b2, dim_alpha1, dim_alpha2) for b1, b2 in itertools.product(self.b, other.b)]
        cs = [andc(c1, c2) for c1, c2 in itertools.product(self.c, other.c)]
        return self.__class__(As, bs, cs)


class ArrayAnsatz(Ansatz):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz as a multidimensional array.

    .. code-block::

          >>> from mrmustard.physics.ansatze import ArrayAnsatz

          >>> array = np.random.random((2, 4, 5))
          >>> ansatz = ArrayAnsatz(array)

    Args:
        array: A (potentially) batched array.
        batched: Whether the array input has a batch dimension.

    Note: The args can be passed non-batched, as they will be automatically broadcasted to the
    correct batch shape if ``batched`` is set to ``False``.
    """

    def __init__(self, array: Batch[Tensor], batched: bool = True):
        array = math.astensor(array)
        if not batched:
            array = array[None, ...]
        self.array = array
        self.num_vars = len(self.array.shape) - 1

    def __neg__(self) -> ArrayAnsatz:
        r"""
        Negates the values in the array.
        """
        return self.__class__(array=-self.array)

    def __eq__(self, other: Ansatz) -> bool:
        r"""
        Whether this ansatz's array is equal to another ansatz's array.

        Note that the comparison is done by numpy allclose with numpy's default rtol and atol.

        Raises:
            ValueError: If the arrays don't have the same shape.
        """
        try:
            return np.allclose(self.array, other.array)
        except Exception as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e

    def __add__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        r"""
        Adds the array of this ansatz and the array of another ansatz.

        Args:
            other: Another ansatz.

        Raises:
            ValueError: If the arrays don't have the same shape.

        Returns:
            ArrayAnsatz: The addition of this ansatz and other.
        """
        try:
            new_array = [a + b for a in self.array for b in other.array]
            return self.__class__(array=math.astensor(new_array))
        except Exception as e:
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.") from e

    def __call__(self, point: Any) -> Scalar:
        r"""
        Evaluates this ansatz at a given point in the domain.
        """
        raise AttributeError("Cannot plot ArrayAnsatz.")

    def __truediv__(self, other: Union[Scalar, ArrayAnsatz]) -> ArrayAnsatz:
        r"""
        Divides this ansatz by another ansatz.

        Args:
            other: A scalar or another ansatz.

        Raises:
            ValueError: If the arrays don't have the same shape.

        Returns:
            ArrayAnsatz: The division of this ansatz and other.
        """
        if isinstance(other, ArrayAnsatz):
            try:
                new_array = [a / b for a in self.array for b in other.array]
                return self.__class__(array=math.astensor(new_array))
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e
        else:
            return self.__class__(array=self.array / other)

    def __mul__(self, other: Union[Scalar, ArrayAnsatz]) -> ArrayAnsatz:
        r"""
        Multiplies this ansatz by another ansatz.

        Args:
            other: A scalar or another ansatz.

        Raises:
            ValueError: If both of array don't have the same shape.

        Returns:
            ArrayAnsatz: The product of this ansatz and other.
        """
        if isinstance(other, ArrayAnsatz):
            try:
                new_array = [a * b for a in self.array for b in other.array]
                return self.__class__(array=math.astensor(new_array))
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
        else:
            return self.__class__(array=self.array * other)

    def __and__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        r"""
        Tensor product of this ansatz with another ansatz.

        Args:
            other: Another ansatz.

        Returns:
            The tensor product of this ansatz and other.
            Batch size is the product of two batches.
        """
        new_array = [math.outer(a, b) for a in self.array for b in other.array]
        return self.__class__(array=math.astensor(new_array))

    @property
    def conj(self):
        r"""
        The conjugate of this ansatz.
        """
        return self.__class__(math.conj(self.array))


def bargmann_Abc_to_phasespace_cov_means(
    A: Matrix, b: Vector, c: Scalar
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    Function to derive the covariance matrix and mean vector of a Gaussian state from its Wigner characteristic function in ABC form.

    The covariance matrix and mean vector can be used to write the characteristic function of a Gaussian state
    :math:
        \Chi_G(r) = \exp\left( -\frac{1}{2}r^T \Omega^T cov \Omega r + i r^T\Omega^T mean \right),
    and the Wigner function of a Gaussian state:
    :math:
        W_G(r) = \frac{1}{\sqrt{\Det(cov)}} \exp\left( -\frac{1}{2}(r - mean)^T cov^{-1} (r-mean) \right).

    The internal expression of our Gaussian state :math:`\rho` is in Bargmann representation, one can write the characteristic function of a Gaussian state in Bargmann representation as
    :math:
        \Chi_G(\alpha) = \Tr(\rho D) = c \exp\left( -\frac{1}{2}\alpha^T A \alpha + \alpha^T b \right).

    This function is to go from the Abc triple in characteristic phase space into the covariance and mean vector for Gaussian state.

    Args:
        A, b, c: The ``(A, b, c)`` triple of the state in characteristic phase space.

    Returns:
        The covariance matrix, mean vector and coefficient of the state in phase space.
    """
    num_modes = A.shape[-1] // 2
    Omega = math.cast(math.transpose(math.J(num_modes)), dtype=math.complex128)
    W = math.transpose(math.conj(math.rotmat(num_modes)))
    coeff = c
    cov = [
        -Omega @ W @ Amat @ math.transpose(W) @ math.transpose(Omega) * settings.HBAR for Amat in A
    ]
    mean = [
        1j * math.matvec(Omega @ W, bvec) * math.sqrt(settings.HBAR, dtype=math.complex128)
        for bvec in b
    ]
    return math.astensor(cov), math.astensor(mean), coeff

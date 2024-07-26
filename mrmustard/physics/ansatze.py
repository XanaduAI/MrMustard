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
from typing import Any, Callable, Union, Optional

import numpy as np

from mrmustard import math, settings
from mrmustard.math.parameters import Variable
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

    def __init__(self) -> None:
        self._fn = None
        self._kwargs = {}

    @abstractmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Ansatz:
        r"""
        Returns an ansatz from a function and kwargs.
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


# pylint: disable=too-many-instance-attributes
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

    def __init__(
        self,
        mat: Batch[Matrix],
        vec: Batch[Vector],
        array: Batch[Tensor],
    ):
        super().__init__()
        self._mat = mat
        self._vec = vec
        self._array = array

        # if (mat, vec, array) have been converted to backend
        self._backends = [False, False, False]

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
        r"""
        Adds two ansatze together. This means concatenating them in the batch dimension.
        In the case where c is a polynomial of different shapes it will add padding zeros to make
        the shapes fit. Example: If the shape of c1 is (1,3,4,5) and the shape of c2 is (1,5,4,3) then the
        shape of the combined object will be (2,5,4,5).
        """
        combined_matrices = math.concat([self.mat, other.mat], axis=0)
        combined_vectors = math.concat([self.vec, other.vec], axis=0)

        a0s = self.array.shape[1:]
        a1s = other.array.shape[1:]
        if a0s == a1s:
            combined_arrays = math.concat([self.array, other.array], axis=0)
        else:
            s_max = np.maximum(np.array(a0s), np.array(a1s))

            padding_array0 = np.array(
                (
                    np.zeros(len(s_max) + 1),
                    np.concatenate((np.array([0]), np.array((s_max - a0s)))),
                ),
                dtype=int,
            ).T
            padding_tuple0 = tuple(tuple(padding_array0[i]) for i in range(len(s_max) + 1))

            padding_array1 = np.array(
                (
                    np.zeros(len(s_max) + 1),
                    np.concatenate((np.array([0]), np.array((s_max - a1s)))),
                ),
                dtype=int,
            ).T
            padding_tuple1 = tuple(tuple(padding_array1[i]) for i in range(len(s_max) + 1))
            a0_new = np.pad(self.array, padding_tuple0, "constant")
            a1_new = np.pad(other.array, padding_tuple1, "constant")
            combined_arrays = math.concat([a0_new, a1_new], axis=0)
        # note output is not simplified
        return self.__class__(combined_matrices, combined_vectors, combined_arrays)

    @property
    def array(self) -> Batch[ComplexMatrix]:
        r"""
        The array of this ansatz.
        """
        self._generate_ansatz()
        if not self._backends[2]:
            self._array = math.atleast_1d(self._array)
            self._backends[2] = True
        return self._array

    @array.setter
    def array(self, array):
        self._array = array
        self._backends[2] = False

    @property
    def batch_size(self):
        r"""
        The batch size of this ansatz.
        """
        return self.mat.shape[0]

    @property
    def degree(self) -> int:
        r"""
        The polynomial degree of this ansatz.
        """
        if self.array.ndim == 1:
            return 0
        return self.array.shape[-1] - 1

    @property
    def polynomial_shape(self) -> tuple[int, tuple]:
        r"""
        This method finds the dimensionality of the polynomial, i.e. how many wires
        have polynomials attached to them and what the degree of the polynomial is
        on each of the wires.
        """
        dim_poly = len(self.array.shape) - 1
        shape_poly = self.array.shape[1:]
        return dim_poly, shape_poly

    @property
    def mat(self) -> Batch[ComplexMatrix]:
        r"""
        The matrix of this ansatz.
        """
        self._generate_ansatz()
        if not self._backends[0]:
            self._mat = math.atleast_3d(self._mat)
            self._backends[0] = True
        return self._mat

    @mat.setter
    def mat(self, array):
        self._mat = array
        self._backends[0] = False

    @property
    def num_vars(self):
        r"""
        The number of variables in this ansatz.
        """
        return self.mat.shape[-1] - self.polynomial_shape[0]

    @property
    def vec(self) -> Batch[ComplexMatrix]:
        r"""
        The vector of this ansatz.
        """
        self._generate_ansatz()
        if not self._backends[1]:
            self._vec = math.atleast_2d(self._vec)
            self._backends[1] = True
        return self._vec

    @vec.setter
    def vec(self, array):
        self._vec = array
        self._backends[1] = False

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

    def _generate_ansatz(self):
        r"""
        This method computes and sets the matrix, vector and array given a function
        and some kwargs.
        """
        names = list(self._kwargs.keys())
        vars = list(self._kwargs.values())

        params = {}
        param_types = []
        for name, param in zip(names, vars):
            try:
                params[name] = param.value
                param_types.append(type(param))
            except AttributeError:
                params[name] = param

        if self._array is None or Variable in param_types:
            mat, vec, array = self._fn(**params)
            self.mat = mat
            self.vec = vec
            self.array = array

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
        A=(batch,n+m,n+m), b=(batch,n+m), c = (batch,k_1,k_2,...,k_m),
        it can be rewritten as an ansatz of dimensions
        A=(batch,2n,2n), b=(batch,2n), c = (batch,l_1,l_2,...,l_n), with l_i = sum_j k_j
        This decomposition is typically favourable if m>n, and will only run if that is the case.
        The naming convention is ``n = dim_alpha``  and ``m = dim_beta`` and ``(k_1,k_2,...,k_m) = shape_beta``
        """
        dim_beta, shape_beta = self.polynomial_shape
        dim_alpha = self.mat.shape[-1] - dim_beta
        batch_size = self.batch_size
        if dim_beta > dim_alpha:
            A_bar = math.block(
                [
                    [
                        math.zeros((batch_size, dim_alpha, dim_alpha), dtype=self.mat.dtype),
                        self.mat[..., :dim_alpha, dim_alpha:],
                    ],
                    [
                        self.mat[..., dim_alpha:, :dim_alpha],
                        self.mat[..., dim_alpha:, dim_alpha:],
                    ],
                ]
            )

            b_bar = math.block(
                [
                    [
                        math.zeros((batch_size, dim_alpha), dtype=self.vec.dtype),
                        self.vec[..., dim_alpha:],
                    ]
                ]
            )
            poly_bar = math.hermite_renormalized_batch(
                A_bar,
                b_bar,
                complex(1),
                (batch_size,) + (math.sum(shape_beta),) * dim_alpha + shape_beta,
            )
            poly_bar = math.moveaxis(poly_bar, 0, dim_alpha)
            c_decomp = math.sum(
                poly_bar * self.array,
                axes=math.arange(
                    len(poly_bar.shape) - dim_beta, len(poly_bar.shape), dtype=math.int32
                ).tolist(),
            )
            c_decomp = math.moveaxis(c_decomp, -1, 0)

            A_decomp = math.block(
                [
                    [
                        self.mat[..., :dim_alpha, :dim_alpha],
                        math.outer(
                            math.ones(batch_size, dtype=self.mat.dtype),
                            math.eye(dim_alpha, dtype=self.mat.dtype),
                        ),
                    ],
                    [
                        math.outer(
                            math.ones(batch_size, dtype=self.mat.dtype),
                            math.eye((dim_alpha), dtype=self.mat.dtype),
                        ),
                        math.zeros((batch_size, dim_alpha, dim_alpha), dtype=self.mat.dtype),
                    ],
                ]
            )
            b_decomp = math.block(
                [
                    [
                        self.vec[..., :dim_alpha],
                        math.zeros((batch_size, dim_alpha), dtype=self.vec.dtype),
                    ]
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

        if A is None and b is None and c is not None:
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

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> PolyExpAnsatz:
        r"""
        Returns a PolyExpAnsatz object from a generator function.
        """
        ret = cls(None, None, None)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

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

        :math:`F(z) = \sum_i [\sum_k c^{(i)}_k \partial_y^k \textrm{exp}((z,y)^T A_i (z,y) / 2 + (z,y)^T b_i)|_{y=0}]`

    with ``k`` being a multi-index. The matrices :math:`A_i` and vectors :math:`b_i` are
    parameters of the exponential terms in the ansatz, and :math:`z` is a vector of variables, and  and :math:`y` is a vector linked to the polynomial coefficients.
    The dimension of ``z + y`` must be equal to the dimension of ``A`` and ``b``.

        .. code-block::

        >>> from mrmustard.physics.ansatze import DiffOpPolyExpAnsatz

        >>> A = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 1.0])
        >>> c = np.array([[1.0,2.0,3.0]])

        >>> F = DiffOpPolyExpAnsatz(A, b, c)
        >>> z = np.array([[1.0],[2.0],[3.0]])

        >>> # calculate the value of the function at the three different ``z``, since z is batched.
        >>> val = F(z)

    A and b can be batched or not, but c needs to include an explicit batch dimension that matches A and b.
    Args:
        A: The list of square matrices :math:`A_i`
        b: The list of vectors :math:`b_i`
        c: The list of arrays :math:`c_i` is coefficients for the polynomial terms in the ansatz.
        An explicit batch dimension that matched A and b has to be given for c.

    """

    def __init__(
        self,
        A: Optional[Batch[Matrix]] = None,
        b: Optional[Batch[Vector]] = None,
        c: Batch[Tensor | Scalar] = np.array([[1.0]]),
        name: str = "",
    ):
        self.name = name

        if A is None and b is None and c is not None:
            raise ValueError("Please provide either A or b.")
        super().__init__(mat=A, vec=b, array=c)

        if self.A.shape[0] != self.c.shape[0] or self.A.shape[0] != self.b.shape[0]:
            raise ValueError("Batch size of A,b,c must be the same.")

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

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> DiffOpPolyExpAnsatz:
        r"""
        Returns a DiffOpPolyExpAnsatz object from a generator function.
        """
        ret = cls(None, None, None)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

    def __call__(self, z: Batch[Vector]) -> Scalar:
        r"""
        Value of this ansatz at ``z``. If ``z`` is batched a value of the function at each of the batches are returned.
        If ``Abc`` is batched it is thought of as a linear combination, and thus the results are added linearly together.
        Note that the batch dimension of ``z`` and ``Abc`` can be different.

        Conventions in code comments:
            n: is the same as dim_alpha
            m: is the same as dim_beta

        Args:
            z: point in C^n where the function is evaluated

        Returns:
            The value of the function.
        """
        dim_beta, shape_beta = self.polynomial_shape
        dim_alpha = self.A.shape[-1] - dim_beta
        batch_size = self.batch_size

        z = math.atleast_2d(z)  # shape (b_arg, n)
        batch_size_arg = z.shape[0]
        if z.shape[-1] != dim_alpha or z.shape[-1] != self.num_vars:
            raise ValueError(
                "The sum of the dimension of the argument and polynomial must be equal to the dimension of A and b."
            )
        zz = math.einsum("...a,...b->...ab", z, z)[..., None, :, :]  # shape (b_arg, 1, n, n))

        A_part = math.sum(
            self.A[..., :dim_alpha, :dim_alpha] * zz, axes=[-1, -2]
        )  # sum((b_arg,1,n,n) * (b_abc,n,n), [-1,-2]) ~ (b_arg,b_abc)
        b_part = math.sum(
            self.b[..., :dim_alpha] * z[..., None, :], axes=[-1]
        )  # sum((b_arg,1,n) * (b_abc,n), [-1]) ~ (b_arg,b_abc)

        exp_sum = math.exp(1 / 2 * A_part + b_part)  # (b_arg, b_abc)
        if dim_beta == 0:
            val = math.sum(exp_sum * self.c, axes=[-1])  # (b_arg)
        else:
            b_poly = math.astensor(
                math.einsum(
                    "ijk,hk",
                    math.cast(self.A[..., dim_alpha:, :dim_alpha], "complex128"),
                    math.cast(z, "complex128"),
                )
                + self.b[..., dim_alpha:]
            )  # (b_arg, b_abc, m)
            b_poly = math.moveaxis(b_poly, 0, 1)  # (b_abc, b_arg, m)
            A_poly = self.A[..., dim_alpha:, dim_alpha:]  # (b_abc, m)
            poly = math.astensor(
                [
                    math.hermite_renormalized_batch(
                        A_poly[i], b_poly[i], complex(1), (batch_size_arg,) + shape_beta
                    )
                    for i in range(batch_size)
                ]
            )  # (b_abc,b_arg,poly)
            poly = math.moveaxis(poly, 0, 1)  # (b_arg,b_abc,poly)
            val = math.sum(
                exp_sum
                * math.sum(
                    poly * self.c, axes=math.arange(2, 2 + dim_beta, dtype=math.int32).tolist()
                ),
                axes=[-1],
            )  # (b_arg)
        return val

    def __mul__(self, other: Union[Scalar, DiffOpPolyExpAnsatz]) -> DiffOpPolyExpAnsatz:
        r"""Multiplies this ansatz by a scalar or another ansatz or a plain scalar.

        Args:
            other: A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            DiffOpPolyExpAnsatz: The product of this ansatz and other.
        """

        def mul_A(A1, A2, dim_alpha, dim_beta1, dim_beta2):
            A3 = math.block(
                [
                    [
                        A1[:dim_alpha, :dim_alpha] + A2[:dim_alpha, :dim_alpha],
                        A1[:dim_alpha, dim_alpha:],
                        A2[:dim_alpha, dim_alpha:],
                    ],
                    [
                        A1[dim_alpha:, :dim_alpha],
                        A1[dim_alpha:, dim_alpha:],
                        math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        A2[dim_alpha:, :dim_alpha],
                        math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                        A2[dim_alpha:, dim_alpha:],
                    ],
                ]
            )
            return A3

        def mul_b(b1, b2, dim_alpha):
            b3 = math.reshape(
                math.block([[b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]]]), -1
            )
            return b3

        def mul_c(c1, c2):
            c3 = math.reshape(math.outer(c1, c2), (c1.shape + c2.shape))
            return c3

        if isinstance(other, DiffOpPolyExpAnsatz):

            dim_beta1, _ = self.polynomial_shape
            dim_beta2, _ = other.polynomial_shape

            dim_alpha1 = self.A.shape[-1] - dim_beta1
            dim_alpha2 = other.A.shape[-1] - dim_beta2
            if dim_alpha1 != dim_alpha2:
                raise TypeError("The dimensionality of the two ansatze must be the same.")
            dim_alpha = dim_alpha1

            new_a = [
                mul_A(
                    math.cast(A1, "complex128"),
                    math.cast(A2, "complex128"),
                    dim_alpha,
                    dim_beta1,
                    dim_beta2,
                )
                for A1, A2 in itertools.product(self.A, other.A)
            ]
            new_b = [mul_b(b1, b2, dim_alpha) for b1, b2 in itertools.product(self.b, other.b)]
            new_c = [mul_c(c1, c2) for c1, c2 in itertools.product(self.c, other.c)]

            return self.__class__(A=new_a, b=new_b, c=new_c)
        else:
            try:
                return self.__class__(self.A, self.b, self.c * other)
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __truediv__(self, other: Union[Scalar, DiffOpPolyExpAnsatz]) -> DiffOpPolyExpAnsatz:
        r"""Multiplies this ansatz by a scalar or another ansatz or a plain scalar.

        Args:
            other: A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            DiffOpPolyExpAnsatz: The product of this ansatz and other.
        """

        def div_A(A1, A2, dim_alpha, dim_beta1, dim_beta2):
            A3 = math.block(
                [
                    [
                        A1[:dim_alpha, :dim_alpha] + A2[:dim_alpha, :dim_alpha],
                        A1[:dim_alpha, dim_alpha:],
                        A2[:dim_alpha, dim_alpha:],
                    ],
                    [
                        A1[dim_alpha:, :dim_alpha],
                        A1[dim_alpha:, dim_alpha:],
                        math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        A2[dim_alpha:, :dim_alpha],
                        math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                        A2[dim_alpha:, dim_alpha:],
                    ],
                ]
            )
            return A3

        def div_b(b1, b2, dim_alpha):
            b3 = math.reshape(
                math.block([[b1[:dim_alpha] + b2[:dim_alpha], b1[dim_alpha:], b2[dim_alpha:]]]), -1
            )
            return b3

        def div_c(c1, c2):
            c3 = math.reshape(math.outer(c1, c2), (c1.shape + c2.shape))
            return c3

        if isinstance(other, DiffOpPolyExpAnsatz):

            dim_beta1, _ = self.polynomial_shape
            dim_beta2, _ = other.polynomial_shape
            if dim_beta1 == 0 and dim_beta2 == 0:
                dim_alpha1 = self.A.shape[-1] - dim_beta1
                dim_alpha2 = other.A.shape[-1] - dim_beta2
                if dim_alpha1 != dim_alpha2:
                    raise TypeError("The dimensionality of the two ansatze must be the same.")
                dim_alpha = dim_alpha1

                new_a = [
                    div_A(
                        math.cast(A1, "complex128"),
                        -math.cast(A2, "complex128"),
                        dim_alpha,
                        dim_beta1,
                        dim_beta2,
                    )
                    for A1, A2 in itertools.product(self.A, other.A)
                ]
                new_b = [div_b(b1, -b2, dim_alpha) for b1, b2 in itertools.product(self.b, other.b)]
                new_c = [div_c(c1, 1 / c2) for c1, c2 in itertools.product(self.c, other.c)]

                return self.__class__(A=new_a, b=new_b, c=new_c)
            else:
                raise NotImplementedError("Only implemented if both c are scalars")
        else:
            try:
                return self.__class__(self.A, self.b, self.c / other)
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e

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
            A3 = math.block(
                [
                    [
                        A1[:dim_alpha1, :dim_alpha1],
                        math.zeros((dim_alpha1, dim_alpha2), dtype=math.complex128),
                        A1[:dim_alpha1, dim_alpha1:],
                        math.zeros((dim_alpha1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        math.zeros((dim_alpha2, dim_alpha1), dtype=math.complex128),
                        A2[:dim_alpha2:, :dim_alpha2],
                        math.zeros((dim_alpha2, dim_beta1), dtype=math.complex128),
                        A2[:dim_alpha2, dim_alpha2:],
                    ],
                    [
                        A1[dim_alpha1:, :dim_alpha1],
                        math.zeros((dim_beta1, dim_alpha2), dtype=math.complex128),
                        A1[dim_alpha1:, dim_alpha1:],
                        math.zeros((dim_beta1, dim_beta2), dtype=math.complex128),
                    ],
                    [
                        math.zeros((dim_beta2, dim_alpha1), dtype=math.complex128),
                        A2[dim_alpha2:, :dim_alpha2],
                        math.zeros((dim_beta2, dim_beta1), dtype=math.complex128),
                        A2[dim_alpha2:, dim_alpha2:],
                    ],
                ]
            )
            return A3

        def andb(b1, b2, dim_alpha1, dim_alpha2):
            b3 = math.reshape(
                math.block([[b1[:dim_alpha1], b2[:dim_alpha2], b1[dim_alpha1:], b2[dim_alpha2:]]]),
                -1,
            )
            return b3

        def andc(c1, c2):
            c3 = math.reshape(math.outer(c1, c2), (c1.shape + c2.shape))
            return c3

        dim_beta1, _ = self.polynomial_shape
        dim_beta2, _ = other.polynomial_shape

        dim_alpha1 = self.A.shape[-1] - dim_beta1
        dim_alpha2 = other.A.shape[-1] - dim_beta2

        As = [
            andA(
                math.cast(A1, "complex128"),
                math.cast(A2, "complex128"),
                dim_alpha1,
                dim_alpha2,
                dim_beta1,
                dim_beta2,
            )
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
        super().__init__()

        self._array = array if batched else [array]
        self._backend_array = False

    @property
    def batch_size(self):
        r"""
        The batch size of this ansatz.
        """
        return self.array.shape[0]

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array of this ansatz.
        """
        self._generate_ansatz()
        if not self._backend_array:
            self._array = math.astensor(self._array)
            self._backend_array = True
        return self._array

    @array.setter
    def array(self, value):
        self._array = value
        self._backend_array = False

    @property
    def num_vars(self) -> int:
        r"""
        The number of variables in this ansatz.
        """
        return len(self.array.shape) - 1

    @classmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> ArrayAnsatz:
        r"""
        Returns an ArrayAnsatz object from a generator function.
        """
        ret = cls(None, True)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

    def _generate_ansatz(self):
        r"""
        This method computes and sets the array given a function
        and some kwargs.
        """
        if self._array is None:
            self.array = [self._fn(**self._kwargs)]

    def __neg__(self) -> ArrayAnsatz:
        r"""
        Negates the values in the array.
        """
        return self.__class__(array=-self.array)

    def __eq__(self, other: Ansatz) -> bool:
        r"""
        Whether this ansatz's array is equal to another ansatz's array.

        Note that the comparison is done by numpy allclose with numpy's default rtol and atol.

        """
        slices = (slice(0, None),) + tuple(
            slice(0, min(si, oi)) for si, oi in zip(self.array.shape[1:], other.array.shape[1:])
        )
        return np.allclose(self.array[slices], other.array[slices], atol=1e-10)

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
            return self.__class__(array=new_array)
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
                return self.__class__(array=new_array)
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
                return self.__class__(array=new_array)
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
        return self.__class__(array=new_array)

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

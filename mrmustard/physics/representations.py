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

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Union

import numpy as np

from mrmustard.math import math
from mrmustard.utils.typing import Batch, Matrix, Scalar, Tensor, Vector


class Ansatz(ABC):
    r"""Abstract parent class for the various structures that we use to define quantum objects.
    It supports the basic mathematical operations (addition, subtraction, multiplication,
    division, negation, equality, etc).

    Effectively it can be thought of as a function over a continuous and/or discrete domain.
    Note that n-dimensional arrays are like functions defined over an integer lattice of points,
    so this class is also the parent of e.g. the Fock representation.
    """

    @abstractmethod
    def __neg__(self) -> Ansatz:
        ...

    @abstractmethod
    def __eq__(self, other: Ansatz) -> bool:
        ...

    @abstractmethod
    def __add__(self, other: Ansatz) -> Ansatz:
        ...

    def __sub__(self, other: Ansatz) -> Ansatz:
        try:
            return self.__add__(-other)
        except AttributeError as e:
            raise TypeError(
                f"Cannot subtract {self.__class__} and {other.__class__}."
            ) from e

    @abstractmethod
    def __call__(self, point: Any) -> Scalar:
        r"""Evaluate the function at the given point in the domain."""
        ...

    @abstractmethod
    def __truediv__(self, other: Union[Scalar, Ansatz]) -> Ansatz:
        ...

    @abstractmethod
    def __mul__(self, other: Union[Scalar, Ansatz]) -> Ansatz:
        ...

    def __rmul__(self, other: Scalar) -> Ansatz:
        return self.__mul__(other=other)

    @abstractmethod
    def __matmul__(self, other: Ansatz) -> Ansatz:
        ...


class MatVecArray(Ansatz):
    r"""The ansatz for certain continuous representations is parametrized by
    a triple of a matrix, a vector and a polynomial (array of coefficients).
    For example, the Bargmann representation c exp(z A z / 2 + b z) is of this form
    (where A,b,c form the triple), or the Wigner representation (where Sigma,mu,1 form the triple).

    Note that this class is not initializable (despite having an initializer)
    because it doesn't implement all the abstract methods of Ansatz.
    Specifically, it doesn't implement the __call__, __mul__ and  __matmul__
    methods, which are representation-specific.

    Its meaning is to group together functionalities that are common to all
    (matrix, vector, array)-type representations and avoid code duplication.

    Args:
        mat (Batch[Matrix]):    the matrix-like data
        vec (Batch[Vector]):    the vector-like data
        array (Batch[Tensor]):  the array-like data
    """

    def __init__(self, mat: Batch[Matrix], vec: Batch[Vector], array: Batch[Tensor]):
        self.mat = math.atleast_3d(math.astensor(mat))
        self.vec = math.atleast_2d(math.astensor(vec))
        self.array = math.atleast_1d(math.astensor(array))
        self.batch_size = self.mat.shape[0]
        self.dim = self.mat.shape[-1]
        self._simplified = False

    def __neg__(self) -> MatVecArray:
        return self.__class__(self.mat, self.vec, -self.array)

    def __eq__(self, other: MatVecArray) -> bool:
        return self._equal_no_array(self, other) and np.allclose(
            self.array, other.array
        )

    def _equal_no_array(self, other: MatVecArray) -> bool:
        self.simplify()
        other.simplify()
        return np.allclose(self.vec, other.vec) and np.allclose(self.mat, other.mat)

    def __add__(self, other: MatVecArray) -> MatVecArray:
        combined_matrices = math.concat([self.mat, other.mat], axis=0)
        combined_vectors = math.concat([self.vec, other.vec], axis=0)
        combined_arrays = math.concat([self.array, other.array], axis=0)
        # note output is not simplified
        return self.__class__(combined_matrices, combined_vectors, combined_arrays)

    def __truediv__(self, x: Scalar) -> MatVecArray:
        if not isinstance(x, (int, float, complex)):
            raise TypeError(f"Cannot divide {self.__class__} by {x.__class__}.")
        new_array = self.array / x
        return self.__class__(self.mat, self.vec, new_array)

    def simplify(self) -> None:
        r"""Simplifies the representation by combining together terms that have the same
        exponential part, i.e. two terms along the batch are considered equal if their
        matrix and vector are equal. In this case only one is kept and the arrays are added.

        Does not run if the representation has already been simplified, so it's safe to call.
        """
        if self._simplified:
            return
        indices_to_check = set(range(self.batch_size))
        removed = []
        while indices_to_check:
            i = indices_to_check.pop()
            for j in indices_to_check.copy():
                if np.allclose(self.mat[i], self.mat[j]) and np.allclose(
                    self.vec[i], self.vec[j]
                ):
                    self.array[i] += self.array[j]
                    indices_to_check.remove(j)
                    removed.append(j)
        to_keep = [i for i in range(self.batch_size) if i not in removed]
        self.mat = math.gather(self.mat, to_keep, axis=0)
        self.vec = math.gather(self.vec, to_keep, axis=0)
        self.array = math.gather(self.array, to_keep, axis=0)
        self._simplified = True

    def simplify_v2(self) -> None:
        r"""A different implementation that orders the batch dimension first."""
        if self._simplified:
            return
        self._order_batch()
        to_keep = [d0 := 0]
        mat, vec = self.mat[d0], self.vec[d0]
        for d in range(1, self.batch_size):
            if np.allclose(mat, self.mat[d]) and np.allclose(vec, self.vec[d]):
                self.array[d0] += self.array[d]
            else:
                to_keep = [d0 := d]
                mat, vec = self.mat[d0], self.vec[d0]
        self.mat = math.gather(self.mat, to_keep, axis=0)
        self.vec = math.gather(self.vec, to_keep, axis=0)
        self.array = math.gather(self.array, to_keep, axis=0)
        self._simplified = True

    def _order_batch(self):
        flattened_tensors = []
        for i in range(self.batch_size):
            flattened_tensors.append(
                math.concat(
                    [self.vec[i].flatten(), self.mat[i].flatten(), self.array[i]],
                    axis=0,
                )  # check in vec, mat, array order
            )
        sorted_indices = np.argsort(flattened_tensors, axis=0, kind="stable")
        self.mat = math.gather(self.mat, sorted_indices, axis=0)
        self.vec = math.gather(self.vec, sorted_indices, axis=0)
        self.array = math.gather(self.array, sorted_indices, axis=0)


class PolyExpAnsatz(MatVecArray):
    """
    Represents the ansatz function:

        F(z) = sum_i poly_i(z) exp(z^T A_i z / 2 + z^T b_i),

    where each poly_i is a polynomial in z that can be expressed as:

        poly_i(z) = sum_k c^(i)_k z^k,

    with k being a multi-index. The batch of arrays c^(i) are not just array values but can be polynomials
    of varying order, defined by the terms arr_k z^k for each i. The matrices A_i and vectors b_i
    are parameters of the exponential terms in the ansatz, and z is a vector of variables.

    Attributes:
        A (Batch[Matrix]): The list of square matrices A_i
        b (Batch[Vector]): The list of vectors b_i
        c (Batch[Tensor]): The array of coefficients for the polynomial terms in the ansatz.

    Example:
        >>> A = [np.array([[1.0, 0.0], [0.0, 1.0]])]
        >>> b = [np.array([1.0, 1.0])]
        >>> c = [np.array(1.0)]
        >>> F = PolyExpAnsatz(A, b, c)
        >>> z = np.array([1.0, 2.0])
        >>> print(F(z))  # prints the value of F at z
    """

    def __init__(self, A: Batch[Matrix], b: Batch[Vector], c: Batch[Tensor]):
        super().__init__(mat=A, vec=b, array=c)

    @property
    def degree(self) -> int:
        return self.array.shape[-1] - 1

    def __call__(self, z: Vector) -> Scalar:
        r"""Value of this ansatz at z.

        Args:
            z (ComplexVector): point at which the function is evaluated

        Returns:
            Scalar: value of the function
        """
        val = 0.0
        for A, b, c in zip(self.A, self.b, self.c):
            val += math.exp(
                0.5 * math.sum(z * math.matvec(A, z)) + math.sum(z * b)
            ) * math.polyval(z, c)  # TODO: implement math.polyval
        return val

    def __mul__(self, other: Union[Scalar, PolyExpAnsatz]) -> PolyExpAnsatz:
        """Multiplies this ansatz by a scalar or another ansatz.

        Args:
            other (Union[Scalar, PolyExpAnsatz]): A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            PolyExpAnsatz: The product of this ansatz and other.
        """
        if isinstance(other, PolyExpAnsatz):
            new_a = [A1 + A2 for A1, A2 in product(self.A, other.A)]
            new_b = [b1 + b2 for b1, b2 in product(self.b, other.b)]
            new_c = [c1 * c2 for c1, c2 in product(self.c, other.c)]
            return self.__class__(A=new_a, b=new_b, c=new_c)
        else:
            try:  # array
                return self.__class__(self.A, self.b, other * self.c)
            except Exception as e:  # Neither same object type nor a array case
                raise TypeError(
                    f"Cannot multiply {self.__class__} and {other.__class__}."
                ) from e

    def __and__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        As = [math.block_diag(a1, a2) for a1 in self.A for a2 in other.A]
        bs = [math.concat([b1, b2], axis=-1) for b1 in self.b for b2 in other.b]
        cs = [math.outer(c1, c2) for c1 in self.c for c2 in other.c]
        return self.__class__(As, bs, cs)

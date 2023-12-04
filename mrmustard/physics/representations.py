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

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Union, Optional

from mrmustard.math import math

from mrmustard.utils.typing import Scalar, Batch, Matrix, Vector
from mrmustard.utils import settings


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
            raise TypeError(f"Cannot subtract {self.__class__} and {other.__class__}.") from e

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


class MatVecScalar(Ansatz):
    r"""The ansatz for certain continuous representations is parametrized by
    a triple of a matrix, a vector and a scalar. For example, the Bargmann
    representation c exp(z A z / 2 + b z) is of this form (where A,b,c form
    the triple), or the Wigner representation (where Sigma,mu,1 form the triple).

    Note that this class is not initializable (despite having an initializer)
    because it doesn't implement all the abstract methods of Ansatz.
    Specifically, it doesn't implement the __call__, __mul__ and  __matmul__
    methods, which are representation-specific.

    Its meaning is to group together functionalities that are common to all
    (matrix, vector, scalar)-type representations and avoid code duplication.

    Args:
        mat (Batch[Matrix]):    the matrix-like data
        vec (Batch[Vector]):    the vector-like data
        scalar (Batch[Scalar]): the scalar-like data
    """

    def __init__(
        self, mat: Batch[Matrix], vec: Batch[Vector], scalar: Optional[Batch[Scalar]]
    ) -> None:
        self.mat = math.atleast_3d(math.astensor(mat))
        self.vec = math.atleast_2d(math.astensor(vec))
        if scalar is None:  # default all 1s
            scalar = math.ones(self.vec.shape[:-1], dtype=math.float64)
        self.scalar = math.atleast_1d(math.astensor(scalar))
        self.batch_size = self.mat.shape[0]
        # self.dim = self.mat.shape[-1]
        self._simplified = False

    def __neg__(self) -> MatVecScalar:
        return self.__class__(self.mat, self.vec, -self.scalar)

    def __eq__(
        self, other: MatVecScalar
    ) -> bool:  # TODO: This method still needs to be rigorously tested and benchmarked
        # simplify and order the batch dimension
        if not self._simplified:
            self.simplify()
        if not other._simplified:
            other.simplify()
        # check for differences
        return (
            np.allclose(self.scalar, other.scalar)
            and np.allclose(self.vec, other.vec)
            and np.allclose(self.mat, other.mat)
        )

    def _equal_no_scalar(self, other: MatVecScalar) -> bool:
        # simplify and order the batch dimension
        if not self._simplified:
            self.simplify()
        if not other._simplified:
            other.simplify()
        return np.allclose(self.vec, other.vec) and np.allclose(self.mat, other.mat)

    def __add__(self, other: MatVecScalar) -> MatVecScalar:
        if self._equal_no_scalar(other):
            new_scalar = self.scalar + other.scalar
            new_self = self.__class__(self.mat, self.vec, new_scalar)
            new_self._simplified = True
            return new_self
        combined_matrices = math.concat([self.mat, other.mat], axis=0)
        combined_vectors = math.concat([self.vec, other.vec], axis=0)
        combined_scalar = math.concat([self.scalar, other.scalar], axis=0)
        # note output is not simplified
        return self.__class__(combined_matrices, combined_vectors, combined_scalar)

    def __truediv__(self, x: Scalar) -> MatVecScalar:
        if not isinstance(x, (int, float, complex)):
            raise TypeError(f"Cannot divide {self.__class__} by {x.__class__}.")
        new_scalar = self.scalar / x
        return self.__class__(self.mat, self.vec, new_scalar)

    def simplify(self) -> None:
        r"""Simplifies the representation by combining together terms that are equal.
        Two terms are considered equal if their matrix and vector parts are equal.
        In which case only one is kept and the scalars are added."""
        indices_to_check = set(range(self.batch_size))
        removed = []
        while indices_to_check:
            i = indices_to_check.pop()
            for j in indices_to_check.copy():
                if np.allclose(self.mat[i], self.mat[j]) and np.allclose(self.vec[i], self.vec[j]):
                    self.scalar[i] += self.scalar[j]
                    indices_to_check.remove(j)
                    removed.append(j)
        to_keep = [i for i in range(self.batch_size) if i not in removed]
        self.mat = math.gather(self.mat, to_keep, axis=0)
        self.vec = math.gather(self.vec, to_keep, axis=0)
        self.scalar = math.gather(self.scalar, to_keep, axis=0)
        self._simplified = True

    def simplify_v2(self) -> None:
        r"""A different implementation that orders the batch dimension first."""
        self._order_batch()
        to_keep = [d0 := 0]
        mat,vec = self.mat[d0], self.vec[d0]
        for d in range(1,self.batch_size):
            if np.allclose(mat, self.mat[d]) and np.allclose(vec, self.vec[d]):
                self.scalar[d0] += self.scalar[d]
            else:
                to_keep = [d0 := d]
                mat,vec = self.mat[d0], self.vec[d0]
        self.mat = math.gather(self.mat, to_keep, axis=0)
        self.vec = math.gather(self.vec, to_keep, axis=0)
        self.scalar = math.gather(self.scalar, to_keep, axis=0)
        self._simplified = True

    def _order_batch(self):
        flattened_tensors = []
        for i in range(self.batch_size):
            flattened_tensors.append(
                math.concat([self.vec[i].flatten(), self.mat[i].flatten(), self.scalar[i]], axis=0)  # vec, mat, scalar ordering
            )
        sorted_indices = np.argsort(flattened_tensors, axis=0, kind="stable")
        self.mat = math.gather(self.mat, sorted_indices, axis=0)
        self.vec = math.gather(self.vec, sorted_indices, axis=0)
        self.scalar = math.gather(self.scalar, sorted_indices, axis=0)

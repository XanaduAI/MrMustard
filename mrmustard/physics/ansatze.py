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

import itertools
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

from mrmustard import math
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


class Ansatz(ABC):
    r"""Abstract parent class for Ansatze that we use to define quantum objects.
    It supports all the mathematical operations (addition, subtraction, multiplication,
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


class PolyExpBase(Ansatz):
    r"""A family of Ansatze parametrized by a triple of a matrix, a vector and an array.
    For example, the Bargmann representation :math:c exp(z A z / 2 + b z): is of this form
    (where ``A``, ``b``, ``c`` is the triple), or the Wigner representation
    (where ``Sigma``, ``mu``, ``1`` is the triple).

    Note that this class is not initializable (despite having an initializer)
    because it doesn't implement all the abstract methods of Ansatz, and it is in
    fact more general.
    Concrete ansatze that inherit from this class will have to implement ``__call__``,
    ``__mul__`` and ``__matmul__``, which are representation-specific.

    Note that the arguments are expected to be batched, i.e. to have a batch dimension
    or to be an iterable. This is because this class also provides the linear superposition
    functionality by implementing the ``__add__`` method, which concatenates the batch dimensions.

    As this can blow up the number of terms in the representation, it is recommended to
    run the `simplify()` method after adding terms together, which will combine together
    terms that have the same exponential part.

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

    def __truediv__(self, x: Scalar) -> PolyExpBase:
        if not isinstance(x, (int, float, complex)):
            raise TypeError(f"Cannot divide {self.__class__} by {x.__class__}.")
        new_array = self.array / x
        return self.__class__(self.mat, self.vec, new_array)

    def simplify(self) -> None:
        r"""Simplifies the representation by combining together terms that have the same
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
        r"""A different implementation that orders the batch dimension first."""
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


class PolyExpAnsatz(PolyExpBase):
    r"""
    Represents the ansatz function:

        :math:`F(z) = sum_i poly_i(z) exp(z^T A_i z / 2 + z^T b_i)`

    where each :math:`poly_i` is a polynomial in ``z`` that can be expressed as:

        :math:`poly_i(z) = sum_k c^(i)_k z^k`,

    with ``k`` being a multi-index. The batch of arrays :math:`c^{(i)}` are not just array values but can be polynomials
    of varying order, defined by the terms :math:`arr_k z^k` for each ``i``.
    The matrices :math:`A_i` and vectors :math:`b_i` are parameters of the
    exponential terms in the ansatz, and :math:`z` is a vector of variables.

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

    def __init__(self, A: Batch[Matrix], b: Batch[Vector], c: Batch[Tensor], name: str = ""):
        self.name = name
        super().__init__(mat=A, vec=b, array=c)

    @property
    def A(self) -> Batch[ComplexMatrix]:
        return self.mat

    @property
    def b(self) -> Batch[ComplexVector]:
        return self.vec

    @property
    def c(self) -> Batch[ComplexTensor]:
        return self.array

    @property
    def degree(self) -> int:
        return self.array.shape[-1] - 1

    def plot(
        self,
        just_phase: bool,
        with_measure: bool,
        log_scale: bool,
        xlim=(-2 * np.pi, 2 * np.pi),
        ylim=(-2 * np.pi, 2 * np.pi),
    ):
        # eval F(z) on a grid of complex numbers
        X, Y = np.mgrid[xlim[0] : xlim[1] : 400j, ylim[0] : ylim[1] : 400j]
        Z = (X + 1j * Y).T
        f_values = self(Z[..., None])
        if log_scale:
            f_values = np.log(np.abs(f_values)) * np.exp(1j * np.angle(f_values))
        if with_measure:
            f_values = f_values * np.exp(-np.abs(Z) ** 2)

        # Get phase and magnitude of F(z)
        phases = np.angle(f_values) / (2 * np.pi) % 1
        magnitudes = np.abs(f_values)
        magnitudes_scaled = magnitudes / np.max(magnitudes)

        # Convert to RGB
        hsv_values = np.zeros(f_values.shape + (3,))
        hsv_values[..., 0] = phases
        hsv_values[..., 1] = 1
        hsv_values[..., 2] = 1 if just_phase else magnitudes_scaled
        rgb_values = colors.hsv_to_rgb(hsv_values)

        # Plot the image
        im, ax = plt.subplots()
        ax.imshow(rgb_values, origin="lower", extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
        ax.set_xlabel("$Re(z)$")
        ax.set_ylabel("$Im(z)$")

        name = "F_{" + self.name + "}(z)"
        name = f"\\arg({name})\\log|{name}|" if log_scale else name
        title = name + "e^{-|z|^2}" if with_measure else name
        title = f"\\arg({name})" if just_phase else title
        ax.set_title(f"${title}$")
        plt.show(block=False)  # why block=False?
        return im, ax

    def __call__(self, z: Batch[Vector]) -> Scalar:
        r"""Value of this ansatz at z. This consumes the last dimension of z.
        i.e. the output is

        Args:
            z (ComplexVector): point at which the function is evaluated

        Returns:
            Scalar: value of the function
        """
        z = np.atleast_2d(z)  # shape (Z, n)
        zz = np.einsum("...a,...b->...ab", z, z)[..., None, :, :]  # shape (Z, 1, n, n))
        A_part = 0.5 * math.sum(
            zz * self.A, axes=[-1, -2]
        )  # sum((Z,1,n,n) * (b,n,n), [-1,-2]) ~ (Z,b)
        b_part = np.sum(z[..., None, :] * self.b, axis=-1)  # sum((Z,1,n) * (b,n), -1) ~ (Z,b)
        exp_sum = np.exp(A_part + b_part)  # (Z, b)
        result = exp_sum * self.c  # (Z, b)
        val = np.sum(result, axis=-1)  # (Z)
        return val

    def __mul__(self, other: Union[Scalar, PolyExpAnsatz]) -> PolyExpAnsatz:
        r"""Multiplies this ansatz by a scalar or another ansatz or a plain scalar.

        Args:
            other (Union[Scalar, PolyExpAnsatz]): A scalar or another ansatz.

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
                return self.__class__(self.A, self.b, other * self.c)
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: PolyExpAnsatz) -> PolyExpAnsatz:
        r"""Tensor product of this ansatz with another ansatz.
        Equivalent to F(a) * G(b) (with different arguments, that is).
        As it distributes over addition on both self and other,
        the batch size of the result is the product of the batch
        size of self and other.

        Args:
            other (PolyExpAnsatz): Another ansatz.

        Returns:
            PolyExpAnsatz: The tensor product of this ansatz and other.
        """
        As = [math.block_diag(a1, a2) for a1 in self.A for a2 in other.A]
        bs = [math.concat([b1, b2], axis=-1) for b1 in self.b for b2 in other.b]
        cs = [math.outer(c1, c2) for c1 in self.c for c2 in other.c]
        return self.__class__(As, bs, cs)

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

from typing import Optional, Union
from itertools import product
import numpy as np

from mrmustard import settings
from mrmustard.physics.data import Data
from mrmustard.physics.bargmann import contract_two_Abc, reorder_abc
from mrmustard.math import Math
from mrmustard.utils.typing import Batch, ComplexMatrix, ComplexVector, Matrix, Scalar, Vector

math = Math()


class MatVecData(Data):  # Note: this class is abstract!
    r"""Contains matrix and vector -like data for certain Representation objects.

    Args:
        mat (Batch[Matrix]): the matrix-like data to be contained in the class
        vec (Batch[Vector]):    the vector-like data to be contained in the class
        coeffs (Batch[Scalar]): the coefficients
    """

    def __init__(self, mat: Batch[Matrix], vec: Batch[Vector], coeffs: Batch[Scalar]) -> None:
        if coeffs is None:  # default all 1s
            coeffs = math.ones(len(vec), dtype=math.float64)

        self.mat = math.atleast_3d(math.astensor(mat))
        self.vec = math.atleast_2d(math.astensor(vec))
        self.coeffs = math.atleast_1d(math.astensor(coeffs))
        assert (
            len(self.mat) == len(self.vec) == len(self.coeffs)
        ), "All inputs must have the same batch size."
        assert (
            self.mat.shape[-1] == self.mat.shape[-2] == self.vec.shape[-1]
        ), "A and b must have the same dimension and A must be symmetric"
        self.batch_dim = self.mat.shape[0]
        self.dim = self.mat.shape[-1]

    def __neg__(self) -> MatVecData:
        return self.__class__(self.mat, self.vec, -self.coeffs)

    def __eq__(self, other: MatVecData, exclude_scalars: bool = False) -> bool:
        A, B = sorted([self, other], key=lambda x: x.batch_dim)  # A smaller or equal batch than B
        # check scalars
        Ac = np.around(A.coeffs, settings.EQUALITY_PRECISION_DECIMALS)
        Bc = memoryview(np.around(B.coeffs, settings.EQUALITY_PRECISION_DECIMALS)).tobytes()
        if exclude_scalars or all(memoryview(c).tobytes() in Bc for c in Ac):
            # check vectors
            Av = np.around(A.vec, settings.EQUALITY_PRECISION_DECIMALS)
            Bv = memoryview(np.around(B.vec, settings.EQUALITY_PRECISION_DECIMALS)).tobytes()
            if all(memoryview(v).tobytes() in Bv for v in Av):
                # check matrices
                Am = np.around(A.mat, settings.EQUALITY_PRECISION_DECIMALS)
                Bm = memoryview(np.around(B.mat, settings.EQUALITY_PRECISION_DECIMALS)).tobytes()
                if all(memoryview(m).tobytes() in Bm for m in Am):
                    return True
        return False

    def __add__(self, other: MatVecData) -> MatVecData:
        if self.__eq__(other, exclude_scalars=True):
            new_coeffs = self.coeffs + other.coeffs
            return self.__class__(self.mat, self.vec, new_coeffs)
        combined_matrices = math.concat([self.mat, other.mat], axis=0)
        combined_vectors = math.concat([self.vec, other.vec], axis=0)
        combined_coeffs = math.concat([self.coeffs, other.coeffs], axis=0)
        return self.__class__(combined_matrices, combined_vectors, combined_coeffs)

    def __truediv__(self, x: Scalar) -> MatVecData:
        if not isinstance(x, (int, float, complex)):
            raise TypeError(f"Cannot divide {self.__class__} by {x.__class__}.")
        new_coeffs = self.coeffs / x
        return self.__class__(self.mat, self.vec, new_coeffs)

    # # TODO: decide which simplify we want to keep
    # def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> MatVecData:
    #     N = self.mat.shape[0]
    #     mask = np.ones(N, dtype=np.int8)

    #     for i in range(N):

    #         for j in range(i + 1, N):

    #             if mask[i] == 0 or i == j:  # evaluated previously
    #                 continue

    #             if np.allclose(
    #                 self.mat[i], self.mat[j], rtol=rtol, atol=atol, equal_nan=True
    #             ) and np.allclose(
    #                 self.vec[i], self.vec[j], rtol=rtol, atol=atol, equal_nan=True
    #             ):
    #                 self.coeffs[i] += self.coeffs[j]
    #                 mask[j] = 0

    #     return self.__class__(
    #         mat = self.mat[mask == 1],
    #         vec = self.vec[mask == 1],
    #         coeffs = self.coeffs[mask == 1]
    #         )

    # # TODO: decide which simplify we want to keep
    # def old_simplify(self) -> None:
    #     indices_to_check = set(range(self.batch_size))
    #     removed = set()

    #     while indices_to_check:
    #         i = indices_to_check.pop()

    #         for j in indices_to_check.copy():
    #             if np.allclose(self.mat[i], self.mat[j]) and np.allclose(
    #                 self.vec[i], self.vec[j]
    #             ):
    #                 self.coeffs[i] += self.coeffs[j]
    #                 indices_to_check.remove(j)
    #                 removed.add(j)

    #     to_keep = [i for i in range(self.batch_size) if i not in removed]
    #     self.mat = self.mat[to_keep]
    #     self.vec = self.vec[to_keep]
    #     self.coeffs = self.coeffs[to_keep]


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


class ABCData(MatVecData):
    r"""Exponential of quadratic polynomial for the Bargmann representation.

    Quadratic Gaussian data is made of: quadratic coefficients, linear coefficients, constant.
    Each of these has a batch dimension, and the batch dimension is the same for all of them.
    They are the parameters of the function `c * exp(x^T A x / 2 + x^T b)`.

    Note that if constants are not provided, they will all be initialized at 1.

    Args:
        A (Batch[Matrix]):          series of quadratic coefficient
        b (Batch[Vector]):          series of linear coefficients
        c (Optional[Batch[Scalar]]):series of constants
    """

    def __init__(
        self, A: Batch[ComplexMatrix], b: Batch[ComplexVector], c: Optional[Batch[Scalar]] = None
    ) -> None:
        super().__init__(mat=A, vec=b, coeffs=c)
        self._contract_idxs = []

    def __call__(self, z: ComplexVector) -> Scalar:
        r"""Value of this Fock-Bargmann function at z.

        Args:
            z (Vector): point at which the function is evaluated

        Returns:
            Scalar: value of the function
        """
        val = 0.0
        for A, b, c in zip(self.A, self.b, self.c):
            val += math.exp(0.5 * math.sum(z * math.matvec(A, z)) + math.sum(z * b)) * c
        return val

    @property
    def A(self) -> Batch[ComplexMatrix]:
        return self.mat

    @property
    def b(self) -> Batch[ComplexVector]:
        return self.vec

    @property
    def c(self) -> Batch[Scalar]:
        return self.coeffs

    def __mul__(self, other: Union[Scalar, ABCData]) -> ABCData:
        if isinstance(other, ABCData):
            new_a = [A1 + A2 for A1, A2 in product(self.A, other.A)]
            new_b = [b1 + b2 for b1, b2 in product(self.b, other.b)]
            new_c = [c1 * c2 for c1, c2 in product(self.c, other.c)]
            return self.__class__(A=new_a, b=new_b, c=new_c)
        else:
            try:  # scalar
                return self.__class__(self.A, self.b, other * self.c)
            except Exception as e:  # Neither same object type nor a scalar case
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: ABCData) -> ABCData:
        As = [math.block_diag(a1, a2) for a1 in self.A for a2 in other.A]
        bs = [math.concat([b1, b2], axis=-1) for b1 in self.b for b2 in other.b]
        cs = [c1 * c2 for c1 in self.c for c2 in other.c]
        return self.__class__(As, bs, cs)

    def conj(self):
        new = self.__class__(math.conj(self.A), math.conj(self.b), math.conj(self.c))
        new._contract_idxs = self._contract_idxs
        return new

    def __matmul__(self, other: ABCData) -> ABCData:
        r"""Implements the contraction of (A,b,c) triples across the marked indices."""
        A, b, c = contract_two_Abc(
            (self.A, self.b, self.c),
            (other.A, other.b, other.c),
            self._contract_idxs,
            other._contract_idxs,
        )

        return self.__class__(A, b, c)

    def __getitem__(self, idx: int | tuple[int, ...]) -> ABCData:
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i > self.dim:
                raise IndexError(
                    f"Index {i} out of bounds for {self.__class__.__qualname__} of dimension {self.dim}."
                )
        new = self.__class__(self.A, self.b, self.c)
        new._contract_idxs = idx
        return new

    def reorder(self, order: tuple[int, ...] | list[int]) -> ABCData:
        A, b, c = reorder_abc((self.A, self.b, self.c), order)
        new = self.__class__(A, b, c)
        new._contract_idxs = self._contract_idxs
        return new

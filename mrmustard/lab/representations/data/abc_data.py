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

from itertools import product
from typing import TYPE_CHECKING, Optional, Union

from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Batch, C, ComplexMatrix, ComplexVector, Scalar

# if TYPE_CHECKING: # This is to avoid the circular import issue with GaussianData<>ABCData
#     from mrmustard.lab.representations.data.gaussian_data import GaussianData


math = Math()


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
        if self._helper_check_is_symmetric(A):
            super().__init__(mat=A, vec=b, coeffs=c)
        else:
            raise ValueError("Matrix A is not symmetric, object can't be initialized.")

    def value(self, x: ComplexVector) -> Scalar:
        r"""Value of this function at x.

        Args:
            x (Vector): point at which the function is evaluated

        Returns:
            Scalar: value of the function
        """
        val = 0.0
        for A, b, c in zip(self.A, self.b, self.c):
            val += math.exp(0.5 * math.sum(x * math.matvec(A, x)) + math.sum(x * b)) * c
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
            new_a = math.astensor([A1 + A2 for A1, A2 in product(self.A, other.A)])
            new_b = math.astensor([b1 + b2 for b1, b2 in product(self.b, other.b)])
            new_c = math.astensor([c1 * c2 for c1, c2 in product(self.c, other.c)])
            return self.__class__(A=new_a, b=new_b, c=new_c)
        else:
            try:  # scalar
                return self.__class__(self.A, self.b, other * self.c)
            except (TypeError, ValueError) as e:  # Neither same object type nor a scalar case
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: ABCData) -> ABCData:
        try:
            As = [math.block_diag(a1, a2) for a1 in self.A for a2 in other.A]
            bs = [math.concat([b1, b2], axis=-1) for b1 in self.b for b2 in other.b]
            cs = [c1 * c2 for c1 in self.c for c2 in other.c]

            return self.__class__(math.astensor(As), math.astensor(bs), math.astensor(cs))

        except AttributeError as e:
            raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__}.") from e

    def __matmul__(self, other: ABCData) -> ABCData:
        r"""Implements the contraction of (A,b,c) triples across the marked indices."""
        for i, j in zip(self._contract_idxs, other._contract_idxs):
            j = j + self.dim
            together = self & other

            noij = list(range(i)) + list(range(i + 1, j) + list(range(j + 1, together.dim)), axis=0)
            Abar = math.gather(math.gather(together.A, noij, axis=1), noij, axis=2)
            bbar = math.gather(together.b, noij, axis=1)
            D = math.concat([together.A[..., i], together.A[..., j]], axis=-1)
            M = math.astensor(
                [
                    [together.A[:, i, i], together.A[:, j, i] - 1],
                    [together.A[:, i, j] - 1, together.A[:, j, j]],
                ]
            )
            Minv = math.inv(M)
            b_ = math.astensor([together.b[:, i], together.b[:, j]])

            newA = Abar - math.einsum("bij,bjk,blk", D, Minv, D)
            newb = bbar - math.einsum("bij,bjk,bk", D, Minv, b_)
            newc = (
                together.c
                * math.exp(-math.einsum("bi,bij,bj", b_, Minv, b_) / 2)
                / math.sqrt(math.det(M))
            )

            return self.__class__(newA, newb, newc)

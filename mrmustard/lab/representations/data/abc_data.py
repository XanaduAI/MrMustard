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

import numpy as np

from mrmustard.lab.representations.data.matvec_data import MatVecData
from mrmustard.math import Math
from mrmustard.typing import Batch, ComplexMatrix, ComplexVector, Scalar

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
        super().__init__(mat=A, vec=b, coeffs=c)

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

    def __call__(self, x: ComplexVector) -> Scalar:
        return self.value(x)

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
        try:
            As = [math.block_diag(a1, a2) for a1 in self.A for a2 in other.A]
            bs = [math.concat([b1, b2], axis=-1) for b1 in self.b for b2 in other.b]
            cs = [c1 * c2 for c1 in self.c for c2 in other.c]
            return self.__class__(As, bs, cs)
        except Exception as e:
            raise TypeError(f"Cannot tensor product {self.__class__} and {other.__class__}.") from e

    def __matmul__(self, other: ABCData) -> ABCData:
        r"""Implements the contraction of (A,b,c) triples across the marked indices."""
        graph = self & other
        newA = graph.A
        newb = graph.b
        newc = graph.c
        for n, (i, j) in enumerate(zip(self._contract_idxs, other._contract_idxs)):
            i = i - np.sum(np.array(self._contract_idxs[:n]) < i)
            j = j + self.dim - n - np.sum(np.array(other._contract_idxs[:n]) < j)
            noij = list(range(i)) + list(range(i + 1, j)) + list(range(j + 1, newA.shape[-1]))
            Abar = math.gather(math.gather(newA, noij, axis=1), noij, axis=2)
            bbar = math.gather(newb, noij, axis=1)
            D = math.gather(
                math.concat([newA[..., i][..., None], newA[..., j][..., None]], axis=-1),
                noij,
                axis=1,
            )
            M = math.concat(
                [
                    math.concat(
                        [
                            newA[:, i, i][:, None, None],
                            newA[:, j, i][:, None, None] - 1,
                        ],
                        axis=-1,
                    ),
                    math.concat(
                        [
                            newA[:, i, j][:, None, None] - 1,
                            newA[:, j, j][:, None, None],
                        ],
                        axis=-1,
                    ),
                ],
                axis=-2,
            )
            Minv = math.inv(M)
            b_ = math.concat([newb[:, i][:, None], newb[:, j][:, None]], axis=-1)

            newA = Abar - math.einsum("bij,bjk,blk", D, Minv, D)
            newb = bbar - math.einsum("bij,bjk,bk", D, Minv, b_)
            newc = (
                newc
                * math.exp(-math.einsum("bi,bij,bj", b_, Minv, b_) / 2)
                / math.sqrt(-math.det(M))
            )
        return self.__class__(newA, newb, newc)

    def __getitem__(self, idx: int | tuple[int, ...]) -> ABCData:
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i > self.dim:
                raise IndexError(
                    f"Index {i} out of bounds for {self.__class__.__qualname__} of dimension {self.dim}."
                )
        self._contract_idxs = idx
        return self

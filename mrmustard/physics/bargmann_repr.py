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

from mrmustard.math import math
from mrmustard.physics.bargmann import contract_two_Abc, reorder_ab
from mrmustard.physics.representations import PolyExpAnsatz
from mrmustard.utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector


class Representation:
    def from_ansatz(self, ansatz: PolyExpAnsatz) -> Representation:
        raise NotImplementedError

    def __add__(self, other):
        return self.from_ansatz(self.ansatz + other.ansatz)

    def __sub__(self, other):
        return self.from_ansatz(self.ansatz - other.ansatz)

    def __mul__(self, other):
        try:
            return self.from_ansatz(self.ansatz * other.ansatz)
        except AttributeError:
            return self.from_ansatz(self.ansatz * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        try:
            return self.from_ansatz(self.ansatz / other.ansatz)
        except AttributeError:
            return self.from_ansatz(self.ansatz / other)

    def __rtruediv__(self, other):
        return self.from_ansatz(other / self.ansatz)


class Bargmann(Representation):
    r"""Fock-Bargmann representation of a broad class of quantum states,
    transformations, measurements, channels, etc.
    The ansatz available in this representation is a linear combination of
    exponentials of bilinear forms with a polynomial part:
    .. math::
        F(z) = sum_i poly_i(z) exp(z^T A_i z / 2 + z^T b_i)

    This function allows for vector space operations on BargmannExp objects including linear combinations,
    outer product, and inner product. The inner product is defined as the contraction of two
    BargmannExp objects across marked indices. This can also be used to contract existing indices
    in one BargmannExp object, e.g. to implement the partial trace.

    Args:
        A (Batch[ComplexMatrix]):          batch of quadratic coefficient A_i
        b (Batch[ComplexVector]):          batch of linear coefficients b_i
        c (Optional[Batch[ComplexTensor]]):      batch of arrays c_i (default: [1.0])
    """

    def __init__(
        self, A: Batch[ComplexMatrix], b: Batch[ComplexVector], c: Batch[ComplexTensor]
    ):
        self.ansatz = PolyExpAnsatz(A, b, c)

    def from_ansatz(self, ansatz: PolyExpAnsatz) -> Bargmann:
        r"""Returns a Bargmann object from an ansatz object."""
        return self.__class__(ansatz.A, ansatz.b, ansatz.c)

    @property
    def A(self) -> Batch[ComplexMatrix]:
        return self.ansatz.A

    @property
    def b(self) -> Batch[ComplexVector]:
        return self.anstaz.b

    @property
    def c(self) -> Batch[ComplexTensor]:
        return self.anstaz.c

    def conj(self):
        new = self.__class__(
            math.conj(self.A), math.conj(self.b), math.conj(self.c), math.conj(self.c)
        )
        new._contract_idxs = self._contract_idxs
        return new

    def __getitem__(self, idx: int | tuple[int, ...]) -> Bargmann:
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= self.ansatz.dim:
                raise IndexError(
                    f"Index {i} out of bounds for ansatz {self.ansatz.__class__.__qualname__} of dimension {self.ansatz.dim}."
                )
        new = self.__class__(self.A, self.b, self.c, self.c)
        new._contract_idxs = idx
        return new

    def __matmul__(self, other: Bargmann) -> Bargmann:
        r"""Implements the inner product of ansatzs across the marked indices."""
        if self.ansatz.degree > 0 or other.ansatz.degree > 0:
            raise NotImplementedError(
                "Inner product of ansatzs is only supported for ansatzs with polynomial of degree 0."
            )
        Abc = []
        for A1, b1, c1 in zip(self.A, self.b, self.c):
            for A2, b2, c2 in zip(other.A, other.b, other.c):
                Abc.append(
                    contract_two_Abc(
                        (A1, b1, c1),
                        (A2, b2, c2),
                        self._contract_idxs,
                        other._contract_idxs,
                        measure=1.0,  # this is for the inner product in Fock-Bargmann representation
                    )
                )
        A, b, c = zip(*Abc)
        return self.__class__(math.astensor(A), math.astensor(b), math.astensor(c))

    def reorder(self, order: tuple[int, ...] | list[int]) -> Bargmann:
        r"""Reorders the indices of the A matrix and b vector of an (A,b,c) triple."""
        A, b = reorder_ab((self.A, self.b), order)
        new = self.__class__(A, b, math.transpose(self.c, order))
        new._contract_idxs = self._contract_idxs
        return new

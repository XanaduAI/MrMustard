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

from mrmustard import math
from mrmustard.physics import bargmann
from mrmustard.physics.ansatze import Ansatz, PolyExpAnsatz
from mrmustard.utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector


class Representation:
    def from_ansatz(self, ansatz: Ansatz) -> Ansatz:
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

    This function allows for vector space operations on Bargmann objects including linear combinations,
    outer product, and inner product. The inner product is defined as the contraction of two
    Bargmann objects across marked indices. This can also be used to contract existing indices
    in one Bargmann object, e.g. to implement the partial trace.

    Args:
        A (Batch[ComplexMatrix]): batch of quadratic coefficient A_i
        b (Batch[ComplexVector]): batch of linear coefficients b_i
        c (Batch[ComplexTensor]): batch of arrays c_i (default: [1.0])
    """

    def __init__(
        self,
        A: Batch[ComplexMatrix],
        b: Batch[ComplexVector],
        c: Batch[ComplexTensor] = [1.0],
    ):
        self.ansatz = PolyExpAnsatz(A, b, c)

    def from_ansatz(self, ansatz: PolyExpAnsatz) -> Bargmann:
        r"""Returns a Bargmann object from an ansatz object."""
        return self.__class__(ansatz.A, ansatz.b, ansatz.c)

    @property
    def A(self) -> Batch[ComplexMatrix]:
        return self.ansatz.mat

    @property
    def b(self) -> Batch[ComplexVector]:
        return self.ansatz.vec

    @property
    def c(self) -> Batch[ComplexTensor]:
        return self.ansatz.array

    def conj(self):
        new = self.__class__(math.conj(self.A), math.conj(self.b), math.conj(self.c))
        new._contract_idxs = self._contract_idxs
        return new

    def __getitem__(self, idx: int | tuple[int, ...]) -> Bargmann:
        idx = (idx,) if isinstance(idx, int) else idx
        for i in idx:
            if i >= self.ansatz.dim:
                raise IndexError(
                    f"Index {i} out of bounds for ansatz {self.ansatz.__class__.__qualname__} of dimension {self.ansatz.dim}."
                )
        new = self.__class__(self.A, self.b, self.c)
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
                    bargmann.contract_two_Abc(
                        (A1, b1, c1),
                        (A2, b2, c2),
                        self._contract_idxs,
                        other._contract_idxs,
                    )
                )
        A, b, c = zip(*Abc)
        return self.__class__(math.astensor(A), math.astensor(b), math.astensor(c))

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> Bargmann:
        r"""Implements the partial trace over the given index pairs.

        Args:
            idx_z (tuple[int, ...]): indices to trace over
            idx_zconj (tuple[int, ...]): indices to trace over

        Returns:
            Bargmann: the ansatz with the given indices traced over
        """
        if self.ansatz.degree > 0:
            raise NotImplementedError(
                "Partial trace is only supported for ansatzs with polynomial of degree 0."
            )
        if len(idx_z) != len(idx_zconj):
            raise ValueError("The number of indices to trace over must be the same for z and z*.")
        A, b, c = [], [], []
        for Ai, bi, ci in zip(self.A, self.b, self.c):
            Aij, bij, cij = bargmann.trace_Abc(Ai, bi, ci, idx_z, idx_zconj)
            A.append(Aij)
            b.append(bij)
            c.append(cij)
        return self.__class__(math.astensor(A), math.astensor(b), math.astensor(c))

    def reorder(self, order: tuple[int, ...] | list[int]) -> Bargmann:
        r"""Reorders the indices of the A matrix and b vector of an (A,b,c) triple."""
        A, b, c = bargmann.reorder_abc((self.A, self.b, self.c), order)
        new = self.__class__(A, b, c)
        new._contract_idxs = self._contract_idxs
        return new

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
This module contains the class for representations.
"""

from __future__ import annotations
from typing import Sequence
from mrmustard import math
from mrmustard.utils.typing import (
    ComplexTensor,
    ComplexMatrix,
    ComplexVector,
    Batch,
)

from .ansatz import Ansatz, PolyExpAnsatz, ArrayAnsatz
from .triples import identity_Abc
from .wires import Wires, ReprEnum

__all__ = ["Representation"]


class Representation:
    r"""
    A class for representations.

    A representation handles the underlying ansatz, wires and keeps track
    of each wire's representation.

    The dictionary to keep track of representations maps the indices of the wires
    to a tuple of the form ``(ReprEnum, parameter)``.

    Args:
        ansatz: An ansatz for this representation.
        wires: The wires of this representation.
    """

    def __init__(self, ansatz: Ansatz | None = None, wires: Wires | None = None) -> None:
        self._ansatz = ansatz
        self._wires = wires or Wires(set(), set(), set(), set())

    @property
    def adjoint(self) -> Representation:
        r"""
        The adjoint of this representation obtained by conjugating the ansatz and swapping
        the ket and bra wires.
        """
        bras = self.wires.bra.indices
        kets = self.wires.ket.indices
        ansatz = self.ansatz.reorder(kets + bras).conj if self.ansatz else None
        return Representation(ansatz, self.wires.adjoint)

    @property
    def ansatz(self) -> Ansatz | None:
        r"""
        The underlying ansatz of this representation.
        """
        return self._ansatz

    @property
    def dual(self) -> Representation:
        r"""
        The dual of this representation obtained by conjugating the ansatz and swapping
        the input and output wires.
        """
        ok = self.wires.ket.output.indices
        ik = self.wires.ket.input.indices
        ib = self.wires.bra.input.indices
        ob = self.wires.bra.output.indices
        ansatz = self.ansatz.reorder(ib + ob + ik + ok).conj if self.ansatz else None
        return Representation(ansatz, self.wires.dual)

    @property
    def wires(self) -> Wires | None:
        r"""
        The wires of this representation.
        """
        return self._wires

    def bargmann_triple(
        self, batched: bool = False
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The Bargmann parametrization of this representation, if available.
        It returns a triple (A, b, c) such that the Bargmann function of this is
        :math:`F(z) = c \exp\left(\frac{1}{2} z^T A z + b^T z\right)`

        If ``batched`` is ``False`` (default), it removes the batch dimension if it is of size 1.

        Args:
            batched: Whether to return the triple batched.
        """
        try:
            A, b, c = self.ansatz.triple
            if not batched and self.ansatz.batch_size == 1:
                return A[0], b[0], c[0]
            else:
                return A, b, c
        except AttributeError as e:
            raise AttributeError("No Bargmann data for this component.") from e

    def fock_array(self, shape: int | Sequence[int], batched=False) -> ComplexTensor:
        r"""
        Returns an array of this representation in the Fock basis with the given shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component if it is
        available, otherwise it defaults to the value of ``AUTOSHAPE_MAX`` in the settings.

        Args:
            shape: The shape of the returned array. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is estimated.
            batched: Whether the returned array is batched or not. If ``False`` (default)
                it will squeeze the batch dimension if it is 1.
        Returns:
            array: The Fock array of this representation.
        """
        num_vars = self.ansatz.num_vars
        if isinstance(shape, int):
            shape = (shape,) * num_vars
        try:
            As, bs, cs = self.bargmann_triple(batched=True)
            if len(shape) != num_vars:
                raise ValueError(
                    f"Expected Fock shape of length {num_vars}, got length {len(shape)}"
                )
            if self.ansatz.polynomial_shape[0] == 0:
                arrays = [
                    math.hermite_renormalized(A, b, c, shape=shape) for A, b, c in zip(As, bs, cs)
                ]
            else:
                arrays = [
                    math.sum(
                        math.hermite_renormalized(A, b, 1, shape=shape + c.shape) * c,
                        axis=math.arange(
                            num_vars, num_vars + len(c.shape), dtype=math.int32
                        ).tolist(),
                    )
                    for A, b, c in zip(As, bs, cs)
                ]
        except AttributeError as e:
            if len(shape) != num_vars:
                raise ValueError(
                    f"Expected Fock shape of length {num_vars}, got length {len(shape)}"
                ) from e
            arrays = self.ansatz.reduce(shape).array
        array = math.sum(arrays, axis=[0])
        arrays = math.expand_dims(array, 0) if batched else array
        return arrays

    def to_bargmann(self) -> Representation:
        r"""
        Converts this representation to a Bargmann representation.
        """
        if isinstance(self.ansatz, PolyExpAnsatz):
            return self
        else:
            if self.ansatz._original_abc_data:
                A, b, c = self.ansatz._original_abc_data
            else:
                A, b, _ = identity_Abc(len(self.wires.quantum))
                c = self.ansatz.data
            bargmann = PolyExpAnsatz(A, b, c)
            for w in self.wires.quantum:
                w.repr = ReprEnum.BARGMANN
            return Representation(bargmann, self.wires)

    def to_fock(self, shape: int | Sequence[int]) -> Representation:
        r"""
        Converts this representation to a Fock representation.

        Args:
            shape: The shape of the returned representation. If ``shape``is given as
                an ``int``, it is broadcasted to all the dimensions. If ``None``, it
                defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        """
        fock = ArrayAnsatz(self.fock_array(shape, batched=True), batched=True)
        try:
            if self.ansatz.polynomial_shape[0] == 0:
                fock._original_abc_data = self.ansatz.triple
        except AttributeError:
            fock._original_abc_data = None
        wires = self.wires.copy()
        for w in wires.quantum_wires:
            w.repr = ReprEnum.FOCK
        return Representation(fock, wires)

    def __eq__(self, other):
        if isinstance(other, Representation):
            return self.ansatz == other.ansatz and self.wires == other.wires
        return False

    def __matmul__(self, other: Representation):
        wires_result, perm = self.wires @ other.wires
        idx_z, idx_zconj = self.wires.contracted_indices(other.wires)

        if type(self.ansatz) is type(other.ansatz):
            self_ansatz = self.ansatz
            other_ansatz = other.ansatz
        else:
            self_ansatz = self.to_bargmann().ansatz
            other_ansatz = other.to_bargmann().ansatz

        rep = self_ansatz.contract(other_ansatz, idx_z, idx_zconj)
        rep = rep.reorder(perm) if perm else rep
        return Representation(rep, wires_result)

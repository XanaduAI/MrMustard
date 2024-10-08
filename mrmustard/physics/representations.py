# Copyright 2024 Xanadu Quantum Technologies Inc.

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
from enum import Enum

from mrmustard import math
from mrmustard.utils.typing import (
    ComplexTensor,
    ComplexMatrix,
    ComplexVector,
    Batch,
)

from .ansatz import Ansatz, PolyExpAnsatz, ArrayAnsatz
from .triples import identity_Abc
from .wires import Wires

__all__ = ["Representation"]


class RepEnum(Enum):
    r"""
    An enum to represent what representation a wire is in.
    """

    NONETYPE = 0
    BARGMANN = 1
    FOCK = 2
    QUADRATURE = 3
    PHASESPACE = 4

    @classmethod
    def from_ansatz(cls, value: Ansatz):
        r"""
        Returns a ``RepEnum`` from an ``Ansatz``.
        """
        if isinstance(value, PolyExpAnsatz):
            return cls(1)
        elif isinstance(value, ArrayAnsatz):
            return cls(2)
        else:
            return cls(0)

    @classmethod
    def _missing_(cls, value):
        return cls.NONETYPE

    def __repr__(self) -> str:
        return self.name


class Representation:
    r"""
    A class for representations.

    A representation handles the underlying ansatz, wires and keeps track
    of each wire's representation.

    Args:
        ansatz: An ansatz for this representation.
        wires: The wires of this representation.
        wire_reps: An optional dictionary for keeping track of each wire's representation.
    """

    def __init__(
        self,
        ansatz: Ansatz | None,
        wires: Wires | None,
        wire_reps: dict | None = None,
    ) -> None:
        self._ansatz = ansatz
        self._wires = wires
        self._wire_reps = wire_reps or dict.fromkeys(wires.modes, RepEnum.from_ansatz(ansatz))

    @property
    def ansatz(self) -> Ansatz | None:
        r"""
        The underlying ansatz of this representation.
        """
        return self._ansatz

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

    def fock(self, shape: int | Sequence[int], batched=False) -> ComplexTensor:
        r"""
        Returns an array representation of this component in the Fock basis with the given shape.
        If the shape is not given, it defaults to the ``auto_shape`` of the component if it is
        available, otherwise it defaults to the value of ``AUTOSHAPE_MAX`` in the settings.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it is estimated.
            batched: Whether the returned representation is batched or not. If ``False`` (default)
                it will squeeze the batch dimension if it is 1.
        Returns:
            array: The Fock representation of this component.
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
                arrays = [math.hermite_renormalized(A, b, c, shape) for A, b, c in zip(As, bs, cs)]
            else:
                arrays = [
                    math.sum(
                        math.hermite_renormalized(A, b, 1, shape + c.shape) * c,
                        axes=math.arange(
                            num_vars, num_vars + len(c.shape), dtype=math.int32
                        ).tolist(),
                    )
                    for A, b, c in zip(As, bs, cs)
                ]
        except AttributeError:
            if len(shape) != num_vars:
                raise ValueError(
                    f"Expected Fock shape of length {num_vars}, got length {len(shape)}"
                )
            arrays = self.ansatz.reduce(shape).array
        array = math.sum(arrays, axes=[0])
        arrays = math.expand_dims(array, 0) if batched else array
        return arrays

    def to_bargmann(self) -> Representation:
        r"""
        Returns a new circuit component with the same attributes as this and a ``Bargmann`` representation.
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
            return Representation(bargmann, self.wires)

    def to_fock(self, shape: int | Sequence[int]) -> Representation:
        r"""
        Returns a new representation with an ``ArrayAnsatz``.

        Args:
            shape: The shape of the returned representation. If ``shape``is given as
                an ``int``, it is broadcasted to all the dimensions. If ``None``, it
                defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        """
        fock = ArrayAnsatz(self.fock(shape, batched=True), batched=True)
        try:
            if self.ansatz.polynomial_shape[0] == 0:
                fock._original_abc_data = self.ansatz.triple
        except AttributeError:
            fock._original_abc_data = None
        return Representation(fock, self.wires)

    def _matmul_indices(self, other: Representation) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r"""
        Finds the indices of the wires being contracted when ``self @ other`` is called.
        """
        # find the indices of the wires being contracted on the bra side
        bra_modes = tuple(self.wires.bra.output.modes & other.wires.bra.input.modes)
        idx_z = self.wires.bra.output[bra_modes].indices
        idx_zconj = other.wires.bra.input[bra_modes].indices
        # find the indices of the wires being contracted on the ket side
        ket_modes = tuple(self.wires.ket.output.modes & other.wires.ket.input.modes)
        idx_z += self.wires.ket.output[ket_modes].indices
        idx_zconj += other.wires.ket.input[ket_modes].indices
        return idx_z, idx_zconj

    def __eq__(self, other):
        if isinstance(other, Representation):
            return (
                self.ansatz == other.ansatz
                and self.wires == other.wires
                and self._wire_reps == other._wire_reps
            )
        return False

    def __matmul__(self, other: Representation):
        wires_result, perm = self.wires @ other.wires
        idx_z, idx_zconj = self._matmul_indices(other)
        if type(self.ansatz) is type(other.ansatz):
            self_rep = self.ansatz
            other_rep = other.ansatz
        else:
            self_rep = self.to_bargmann().ansatz
            other_rep = other.to_bargmann().ansatz

        rep = self_rep[idx_z] @ other_rep[idx_zconj]
        rep = rep.reorder(perm) if perm else rep
        return Representation(rep, wires_result)

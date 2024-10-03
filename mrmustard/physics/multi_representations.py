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
This module contains the class for multi-representations.
"""

from __future__ import annotations
from typing import Any, Sequence
from enum import Enum

from mrmustard import settings, math, widgets as mmwidgets
from mrmustard.utils.typing import (
    Scalar,
    ComplexTensor,
    ComplexMatrix,
    ComplexVector,
    Vector,
    Batch,
)

from .representations import Representation, Bargmann, Fock
from .triples import identity_Abc
from .wires import Wires

__all__ = ["MultiRepresentation"]


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
    def from_representation(cls, value: Representation):
        r""" """
        return cls[value.__class__.__name__.upper()]

    @classmethod
    def _missing_(cls, value):
        return cls.NONETYPE

    def __repr__(self) -> str:
        return self.name


class MultiRepresentation:
    r"""
    A class for multi-representations.

    A multi-representation handles the underlying representation, the wires of
    said representation and keeps track of representation conversions.

    Args:
        representation: A representation for this multi-representation.
        wires: The wires of this multi-representation.
        wire_reps: An optional dictionary for keeping track of each wire's representation.
    """

    def __init__(
        self,
        representation: Representation | None,
        wires: Wires | None,
        wire_reps: dict | None = None,
    ) -> None:
        self._representation = representation
        self._wires = wires
        rep_enum = (
            RepEnum[representation.__class__.__name__.upper()] if representation else RepEnum(1)
        )
        self._wire_reps = wire_reps or dict.fromkeys(wires.modes, rep_enum)

    @property
    def representation(self) -> Representation | None:
        r"""
        The underlying representation of this multi-representation.
        """
        return self._representation

    @property
    def wires(self) -> Wires | None:
        r"""
        The wires of this multi-representation.
        """
        return self._wires

    def bargmann_triple(
        self, batched: bool = False
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The Bargmann parametrization of this multi-representation, if available.
        It returns a triple (A, b, c) such that the Bargmann function of this is
        :math:`F(z) = c \exp\left(\frac{1}{2} z^T A z + b^T z\right)`

        If ``batched`` is ``False`` (default), it removes the batch dimension if it is of size 1.

        Args:
            batched: Whether to return the triple batched.
        """
        try:
            A, b, c = self.representation.triple
            if not batched and self.representation.ansatz.batch_size == 1:
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
        num_vars = self.representation.ansatz.num_vars
        if isinstance(shape, int):
            shape = (shape,) * num_vars
        try:
            As, bs, cs = self.bargmann_triple(batched=True)
            if len(shape) != num_vars:
                raise ValueError(
                    f"Expected Fock shape of length {num_vars}, got length {len(shape)}"
                )
            if self.representation.ansatz.polynomial_shape[0] == 0:
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
            arrays = self.representation.reduce(shape).array
        array = math.sum(arrays, axes=[0])
        arrays = math.expand_dims(array, 0) if batched else array
        return arrays

    def to_bargmann(self) -> MultiRepresentation:
        r"""
        Returns a new circuit component with the same attributes as this and a ``Bargmann`` representation.
        """
        if isinstance(self.representation, Bargmann):
            return self
        else:
            if self.representation.ansatz._original_abc_data:
                A, b, c = self.representation.ansatz._original_abc_data
            else:
                A, b, _ = identity_Abc(len(self.wires.quantum))
                c = self.representation.data
            bargmann = Bargmann(A, b, c)
            return MultiRepresentation(bargmann, self.wires)

    def to_fock(self, shape: int | Sequence[int]) -> MultiRepresentation:
        r"""
        Returns a new multi-representation with  a ``Fock`` representation.

        Args:
            shape: The shape of the returned representation. If ``shape``is given as
                an ``int``, it is broadcasted to all the dimensions. If ``None``, it
                defaults to the value of ``AUTOSHAPE_MAX`` in the settings.
        """
        fock = Fock(self.fock(shape, batched=True), batched=True)
        try:
            if self.representation.ansatz.polynomial_shape[0] == 0:
                fock.ansatz._original_abc_data = self.representation.triple
        except AttributeError:
            fock.ansatz._original_abc_data = None
        return MultiRepresentation(fock, self.wires)

    def _matmul_indices(
        self, other: MultiRepresentation
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
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
        if isinstance(other, MultiRepresentation):
            return (
                self.representation == other.representation
                and self.wires == other.wires
                and self._wire_reps == other._wire_reps
            )
        return False

    def __matmul__(self, other: MultiRepresentation):
        wires_result, perm = self.wires @ other.wires
        idx_z, idx_zconj = self._matmul_indices(other)
        if type(self.representation) is type(other.representation):
            self_rep = self.representation
            other_rep = other.representation
        else:
            self_rep = self.to_bargmann().representation
            other_rep = other.to_bargmann().representation

        rep = self_rep[idx_z] @ other_rep[idx_zconj]
        rep = rep.reorder(perm) if perm else rep
        return MultiRepresentation(rep, wires_result)

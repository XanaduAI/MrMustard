# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the implementation of the :class:`State` class."""

from typing import Any, List

from mrmustard.lab.abstract import Measurement, State, Transformation


class Operation:
    r"""Circuit operation. Can be a transformation, measurement or a state.
    It interfaces the circuit model with the Tensor Network model.

    The circuit model consists in operations having input modes and output modes.
    Here modes don't distinguish if the state is pure or mixed or whether a transformation
    is unitary or non-unitary, as they only describe the circuit, not the underlying tensors.
    Wires are the same as modes, but they split into left and right components for the input
    and left and right components for the output. NOTE that left and right does not correspond
    to left and right in the circuit picture, but to the left and right in the tensor network
    picture.

    The tensor network (TN) model is the mathematical model of the circuit. It consists in a
    collection of tensors, one per operation, and a prescription on how to contract them.
    Diagrammatically, the TN model for a circuit is easier to imagine from top to bottom.
    Inputs are on top, outputs are on the bottom, and each wire/mode splits
    into a left and right component.

    In either model:
    - If the operation is a transformation, it acts on wires by transforming the L and R
    indices at the input to new L and R indices at the output.
    - If the operation is a measurement, it consumes pairs of L and R indices from the input
    and produces a measurement outcome.
    - If the operation is a preparation, it creates new output wires with L and R indices.

    Orderings:
    Choi tensor axis order is (ol1, ol2,..., il1, il2, ..., or1, or2, ..., ir1, ir2, ...)
    Unitary/kraus axis order is (ol1,ol2,..., il1, il2, ...)
    Measurement axis order is (il1, il2, ..., ir1, ir2, ...)
    State axis order is (ol1, ol2,..., or1, or2, ...)

    If a state is rank-1 then the axis order is (ol1, ol2,...)
    If a measurement is rank-1 then the axis order is (il1, il2, ...)
    unitary is like a rank-1 choi (i.e. having only left indices)
    """

    def __init__(self, op: Any):
        if isinstance(op, Transformation):
            if op.is_unitary:
                self.from_unitary(op.modes)
            else:
                self.from_choi(op.modes, op.modes)  # update when channel distinguishes in/out modes

        elif isinstance(op, Measurement):
            if op.is_projective:
                self.from_proj(op.modes)
            else:
                self.from_povm(op.modes)

        elif isinstance(op, State):
            if op.is_pure:
                self.from_ket(op.modes)
            else:
                self.from_dm(op.modes)

    def __init_circuit__(self, olk: List[int], ilk: List[int], ork: List[int], irk: List[int]):
        r"""Initializes the operation at the circuit level.

        Args:
            olk: list of left output wires
            ilk: list of left input wires
            ork: list of right output wires
            irk: list of right input wires
        """
        self.olk = olk
        self.ilk = ilk
        self.ork = ork
        self.irk = irk

    def olk_axis(self, olk: int):
        r"""Returns the axis of the left output mode in the underlying tensor."""
        return self.olk.index(olk) if olk in self.olk else None

    def ilk_axis(self, ilk: int):
        r"""Returns the axis of the left input mode in the underlying tensor."""
        return self.ilk.index(ilk) + len(self.olk) if ilk in self.ilk else None

    def ork_axis(self, ork: int):
        r"""Returns the axis of the right output mode in the underlying tensor."""
        return self.ork.index(ork) + len(self.olk) + len(self.ilk) if ork in self.ork else None

    def irk_axis(self, irk: int):
        r"""Returns the axis of the right input mode in the underlying tensor."""
        return (
            self.irk.index(irk) + len(self.olk) + len(self.ilk) + len(self.ork)
            if irk in self.irk
            else None
        )

    def from_ket(self, modes: List[int]):
        r"""Initializes the operation from a ket.

        Args:
            modes: modes of the ket
        """
        self.modes_out = modes
        self.__init_circuit__(olk=modes, ilk=[], ork=[], irk=[])

    def from_dm(self, modes: List[int]):
        r"""Initializes the operation from a density matrix.

        Args:
            modes: modes of the density matrix
        """
        self.modes_out = modes
        self.__init_circuit__(olk=modes, ilk=[], ork=modes, irk=[])

    def from_choi(self, in_modes: List[int], out_modes: List[int]):
        r"""Initializes the operation from a choi op.

        Args:
            in_modes: modes on which the choi op acts
            out_modes: modes that the choi op outputs
        """
        self.modes_in = in_modes
        self.modes_out = out_modes
        self.__init_circuit__(olk=out_modes, ilk=in_modes, ork=out_modes, irk=in_modes)

    def from_kraus(self, in_modes: List[int], out_modes: List[int]):
        r"""Initializes the operation from a kraus op.

        Args:
            in_modes: modes on which the kraus op acts
            out_modes: modes that the kraus op outputs
        """
        self.modes_in = in_modes
        self.modes_out = out_modes
        self.__init_circuit__(olk=out_modes, ilk=in_modes, ork=[], irk=[])

    def from_unitary(self, modes: List[int]):
        r"""Initializes the operation from a unitary transformation.

        Args:
            modes: modes on which the transformation acts (which are the same as the output modes)
        """
        self.modes_in = modes
        self.modes_out = modes
        self.__init_circuit__(olk=modes, ilk=modes, ork=[], irk=[])

    def from_povm(self, modes: List[int]):
        r"""Initializes the measurement operation from a povm.

        Args:
            modes: modes of the measurement
        """
        self.modes_in = modes
        self.__init_circuit__(olk=[], ilk=modes, ork=[], irk=modes)

    def from_proj(self, modes: List[int], rank1: bool = False):
        r"""Initializes the measurement operation from a ket.

        Args:
            modes: modes of the measurement
        """
        self.modes_in = modes
        self.__init_circuit__(olk=[], ilk=modes, ork=[], irk=[])

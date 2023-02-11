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

from collections import namedtuple
from typing import Any, List

from mrmustard.lab.abstract import Measurement, State, Transformation


class Operation:
    r"""Circuit operation. It can wrap a transformation, measurement or state.
    The purpose it to interface the circuit model with the Tensor Network model.

    The circuit model consists in operations having input and output modes.
    Here modes don't distinguish if the state is pure or mixed or whether a transformation
    is unitary or non-unitary, as they only describe the circuit, not the underlying tensors.

    The tensor network (TN) model is the mathematical model of the circuit. It consists in a
    collection of tensors, one per operation, and a prescription on how to contract them.
    Diagrammatically, the TN model for a circuit is easier to imagine from top to bottom.
    Inputs are on top, outputs are on the bottom, and each mode splits
    into a left and right component to account for density matrices and channels.

    In either model:
    - If the operation is a transformation, it acts on wires by transforming the L and R
    indices at the input to new L and R indices at the output.
    - If the operation is a measurement, it consumes pairs of L and R indices from the input
    and produces a measurement outcome.
    - If the operation is a preparation, it creates new output wires with L and R indices.

    Assumed axis order:
    - Choi tensor axis order is (ol1, ol2,..., il1, il2, ..., or1, or2, ..., ir1, ir2, ...)
    - Unitary/kraus axis order is (ol1,ol2,..., il1, il2, ...)
    - Measurement axis order is (il1, il2, ..., ir1, ir2, ...)
    - State axis order is (ol1, ol2,..., or1, or2, ...)

    If a state is a Hilbert vector then the axis order is (ol1, ol2,...).
    If a measurement is rank-1 then the axis order is (il1, il2, ...) like the Hilbert vector
    that it corresponds to. A unitary is like a rank-1 choi (i.e. with only left indices).

    Arguments:
        op: operation to be wrapped

    Example:
        >>> op = Operation(Sgate(r=1.0, modes=[0]))
        >>> op.ol_, op.il_, op.or_, op.ir_
        [0], [0], [], []
        >>> op.axes_for_mode(0)
        axis_spec(ol_=0, il_=1, or_=None, ir_=None)

    Example:
        >>> op = Operation(Coherent(x=1.0), modes=[0])
        >>> op.ol_, op.il_, op.or_, op.ir_
        [0], [], [], []
        >>> op.axes_for_mode(0)
        axis_spec(ol_=0, il_=None, or_=None, ir_=None)

    Example:
        >>> op = Operation(Attenuator(0.5, modes=[0]))
        >>> op.ol_, op.il_, op.or_, op.ir_
        [0], [0], [0], [0]
        >>> op.axes_for_mode(0)
        axis_spec(ol_=0, il_=1, or_=2, ir_=3)


    """

    def __init__(self, op: Any):
        self.name = op.__class__.__qualname__
        if isinstance(op, Transformation):
            if op.is_unitary:
                self.from_unitary(op.modes)
            else:
                self.from_choi(op.modes, op.modes)  # update when channel distinguishes in/out modes

        elif isinstance(op, Measurement):
            if op.outcome.is_pure:
                self.from_proj(op.modes)
            else:
                self.from_povm(op.modes)

        elif isinstance(op, State):
            if op.is_pure:
                self.from_ket(op.modes)
            else:
                self.from_dm(op.modes)

    def __init_circuit__(self, ol_: List[int], il_: List[int], or_: List[int], ir_: List[int]):
        r"""Initializes the operation at the circuit level.

        Args:
            ol_: list of left output wires
            il_: list of left input wires
            or_: list of right output wires
            ir_: list of right input wires
        """
        self.ol_ = ol_
        self.il_ = il_
        self.or_ = or_
        self.ir_ = ir_

    @property
    def num_axes(self):
        r"""Returns the number of axes of the underlying tensor."""
        return len(self.ol_) + len(self.il_) + len(self.or_) + len(self.ir_)

    def axes_for_mode(self, mode: int):
        r"""Returns the axes of the underlying tensor corresponding to a mode.

        Args:
            mode: mode of the operation
        """
        axis_spec = namedtuple("axis_spec", ["ol_", "il_", "or_", "ir_"])
        return axis_spec(
            ol_=self.ol_axis(mode),
            il_=self.il_axis(mode),
            or_=self.or_axis(mode),
            ir_=self.ir_axis(mode),
        )

    def ol_axis(self, ol_: int):
        r"""Returns the axis of the left output mode in the underlying tensor."""
        return self.ol_.index(ol_) if ol_ in self.ol_ else None

    def il_axis(self, il_: int):
        r"""Returns the axis of the left input mode in the underlying tensor."""
        return self.il_.index(il_) + len(self.ol_) if il_ in self.il_ else None

    def or_axis(self, or_: int):
        r"""Returns the axis of the right output mode in the underlying tensor."""
        return self.or_.index(or_) + len(self.ol_) + len(self.il_) if or_ in self.or_ else None

    def ir_axis(self, ir_: int):
        r"""Returns the axis of the right input mode in the underlying tensor."""
        return (
            self.ir_.index(ir_) + len(self.ol_) + len(self.il_) + len(self.or_)
            if ir_ in self.ir_
            else None
        )

    def from_ket(self, modes: List[int]):
        r"""Initializes the operation from a ket.

        Args:
            modes: modes of the ket
        """
        self.modes_in = []
        self.modes_out = modes
        self.__init_circuit__(ol_=modes, il_=[], or_=[], ir_=[])

    def from_dm(self, modes: List[int]):
        r"""Initializes the operation from a density matrix.

        Args:
            modes: modes of the density matrix
        """
        self.modes_in = []
        self.modes_out = modes
        self.__init_circuit__(ol_=modes, il_=[], or_=modes, ir_=[])

    def from_choi(self, in_modes: List[int], out_modes: List[int]):
        r"""Initializes the operation from a choi op.

        Args:
            in_modes: modes on which the choi op acts
            out_modes: modes that the choi op outputs
        """
        self.modes_in = in_modes
        self.modes_out = out_modes
        self.__init_circuit__(ol_=out_modes, il_=in_modes, or_=out_modes, ir_=in_modes)

    def from_kraus(self, in_modes: List[int], out_modes: List[int]):
        r"""Initializes the operation from a kraus op.

        Args:
            in_modes: modes on which the kraus op acts
            out_modes: modes that the kraus op outputs
        """
        self.modes_in = in_modes
        self.modes_out = out_modes
        self.__init_circuit__(ol_=out_modes, il_=in_modes, or_=[], ir_=[])

    def from_unitary(self, modes: List[int]):
        r"""Initializes the operation from a unitary transformation.

        Args:
            modes: modes on which the transformation acts (which are the same as the output modes)
        """
        self.modes_in = modes
        self.modes_out = modes
        self.__init_circuit__(ol_=modes, il_=modes, or_=[], ir_=[])

    def from_povm(self, modes: List[int]):
        r"""Initializes the measurement operation from a povm.

        Args:
            modes: modes of the measurement
        """
        self.modes_in = modes
        self.modes_out = []
        self.__init_circuit__(ol_=[], il_=modes, or_=[], ir_=modes)

    def from_proj(self, modes: List[int], rank1: bool = False):
        r"""Initializes the measurement operation from a ket.

        Args:
            modes: modes of the measurement
        """
        self.modes_in = modes
        self.modes_out = []
        self.__init_circuit__(ol_=[], il_=modes, or_=[], ir_=[])

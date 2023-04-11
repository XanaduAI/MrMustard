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

from typing import Dict, List, Optional, Union

from mrmustard.lab.abstract import Measurement, State, Transformation
from mrmustard.utils.tagdispenser import TagDispenser

Op = Union[State, Transformation, Measurement]


class Circuit:
    r"""Circuit operation. It can wrap a transformation, a measurement or a state.
    It is an interface between the circuit model and the fock model, useful for
    example for representing a circuit as a tensor network.

    Upon initialization, the Operation is passed a list of input modes and a list of output modes.
    The input modes are the modes that the operation will act on, and the output modes are the
    modes that the operation will create. This covers all the kinds of objects that we
    can use in a circuit, and  can model states (no input modes), transformations
    (both inputs and outputs) and measurements (no outputs).

    The main idea is that the Operation wraps a transformation, a measurement or a state,
    and it allows us to use them within a circuit.
    Operation[State] is a state at the input to a circuit
    Operation[Transformation] is a gate in a circuit
    Operation[Measurement] is a measurement at the end of a circuit

    The circuit model consists in a list of Operations, whose input and output modes
    inform the circuit on how to connect them up. In the fock model, the operations
    are unaware of the mode in which they are or act, and they are represented as tensors with
    a given number of wires. For this reason, the fock model can be represented as a
    tensor network (TN), which is a Fock representation of the entire circuit, to be
    contracted in order to obtain the final result.

    Note that in the circuit model, modes don't distinguish if the state is pure or mixed or
    whether a transformation is unitary or non-unitary, as they only describe the
    connections of the circuit, not the underlying physical objects. In fact, the correspondence
    between modes and tensor wires is not always 1:1, as a mode can be associated to two wires
    (e.g. for density matrices).

    The circuit model is easier to imagine flowing from left to right, while
    the fock model is easier to imagine flowing from top to bottom (so that the Left and
    Right indices of a mode of density matrices (first two wires of the dm tensor) are still on the
    left and on the right). Input states are on top, outputs are on the bottom, and each mode has
    a left and right component to account for density matrices and channels, e.g.

    Circuit model (left to right):

    rho --U-- povm          ket --U-- proj          rho --choi--

    Fock Tensor Network model (top to bottom):

    L   --rho--  R          L   --ket               L   --rho--  R
        |     |                 |                       |     |
        U     U*                U                       -choi--
        |     |                 |                       |     |
        -povm--                 -proj

    In either model:
    - If the operation is a state (preparation), it creates new output wires
    with L / R indices.
    - If the operation is a transformation, it acts on wires by transforming
    the L and R indices at the input to new L and R indices at the output.
    - If the operation is a measurement, it consumes pairs of L and R indices
    from the input and produces a measurement outcome.

    Assumed wire order (o = out, i = in, l = left, r = right):
    - Choi tensor wire order is (ol1, ol2,..., il1, il2, ..., or1, or2, ..., ir1, ir2, ...)
    - Unitary/kraus wire order is (ol1,ol2,..., il1, il2, ...)
    - POVM wire order is (il1, il2, ..., ir1, ir2, ...)
    - Density Matrix wire order is (ol1, ol2,..., or1, or2, ...)

    If a state is a Hilbert vector then the wire order is (ol1, ol2, ...).
    If a measurement is projective then the wire order is (il1, il2, ...).
    Likewise, a unitary is like a rank-1 choi (i.e. with only left indices).

    Arguments:
        layers: list of layers of operations
        double_wires: whether to use double wires for density matrices
        name: name of the operation
        in_modes: list of input modes
        out_modes: list of output modes

    Returns:
        Operation: wrapped operation
    """

    def __init__(
        self,
        layers: List[List[Op]] = [[]],
        double_wires: bool = False,
        name: str = "?",
        in_modes: List[int] = [],
        out_modes: List[int] = [],
    ):
        self.layers: List[List[Op]] = layers
        self.double_wires: bool = double_wires
        self.name: str = name
        self.in_modes = in_modes
        self.out_modes = out_modes

        OUT: int = len(out_modes)
        IN: int = len(in_modes)
        dispenser: TagDispenser = TagDispenser()

        self.tags: Dict[str, List[int]] = {
            "out_L": [dispenser.get_tag() for _ in range(OUT)],
            "in_L": [dispenser.get_tag() for _ in range(IN)],
            "out_R": [dispenser.get_tag() for _ in range(self.LR * OUT)],
            "in_R": [dispenser.get_tag() for _ in range(self.LR * IN)],
        }
        self.wires: Dict[str, List[int]] = {
            "out_L": [i for i in range(OUT)],
            "in_L": [i + OUT for i in range(IN)],
            "out_R": [i + OUT + IN for i in range(self.LR * OUT)],
            "in_R": [i + 2 * OUT + IN for i in range(self.LR * IN)],
        }

        self.connect_layers()

    def connect_layers(self):
        "set wire connections for TN contractions or phase space products"
        # NOTE: if double_wires is True for one op, then it must be for all ops. Revisit this at some point.
        for i, layer in enumerate(self.layers):
            for op in layer:
                if op.double_wires:
                    self.double_wires = True
                    break
            if not self.double_wires:
                break

        for i, layeri in enumerate(self.layers):
            for j, layerj in enumerate(self.layers[i + 1 :]):
                for op1 in layeri:
                    for op2 in layerj:
                        for mode in set(op1.modes_out) & set(op2.modes_in):
                            axes1 = op1.mode_to_axes(mode)
                            axes2 = op2.mode_to_axes(mode)
                            for ax1, ax2 in zip(axes1, axes2):
                                min_tag = min(op2.tags[ax2], op1.tags[ax1])
                                op2.tags[ax2] = min_tag
                                op1.tags[ax1] = min_tag

    def __del__(self):
        # release tags
        for tag in set(self.tags.values()):
            TagDispenser().give_back_tag(tag)
        super().__del__()

    def __gt__(self, other):
        r"""Overloads the "greater than" operator to allow chaining of operations.

        Args:
            other (Operation): operation to chain

        Returns:
            Circuit: chained operation
        """
        if not set(self.out_modes) & set(other.in_modes):
            raise ValueError("No connections between operations. Use & to parallelize.")
        out_modes_1 = set(self.out_modes) - set(other.in_modes)
        out_modes_2 = set(other.out_modes)
        if set(out_modes_1) & set(out_modes_2):
            raise ValueError("Overlapping output modes.")
        in_modes_2 = set(other.in_modes) - set(self.out_modes)
        in_modes_1 = set(self.in_modes)
        if set(in_modes_1) & set(in_modes_2):
            raise ValueError("Overlapping input modes.")
        return Circuit(
            in_modes=sorted(list(in_modes_1 | in_modes_2)),
            out_modes=sorted(list(out_modes_1 | out_modes_2)),
            name=self.name + " >> " + other.name,
            layers=[[self], [other]],
        )

    def __and__(self, other):
        r"""Overloads the and operator to allow parallelization of operations.

        Args:
            other (Operation): operation to parallelize

        Returns:
            Circuit: parallelized operation
        """
        if set(self.out_modes) & set(other.in_modes):
            raise ValueError("Overlapping input/output modes. Use >> to chain.")
        return Circuit(
            in_modes=sorted(list(set(self.in_modes) | set(other.in_modes))),
            out_modes=sorted(list(set(self.out_modes) | set(other.out_modes))),
            name=self.name + " & " + other.name,
            layers=[[self, other]],
        )

    def tag_to_wire(self, tag: int) -> int:
        r"""Returns the wire corresponding to a given tag.
        One to one mapping.

        Args:
            tag (int): tag

        Returns:
            int: wire
        """
        for kind, tag_list in self.tags.items():
            if tag in tag_list:
                return self.wires[kind][tag_list.index(tag)]
        raise ValueError("Tag not found.")

    def wire_to_tag(self, wire: int) -> int:
        r"""Returns the tag corresponding to a given wire.
        One to one mapping.

        Args:
            wire (int): wire

        Returns:
            int: tag
        """
        for kind, wire_list in self.wires.items():
            if wire in wire_list:
                return self.tags[kind][wire_list.index(wire)]
        raise ValueError("Wire not found.")

    def mode_to_wires(self, mode: int) -> List[int]:
        r"""Returns the wires corresponding to a given mode.
        One to many mapping.

        Args:
            mode (int): mode

        Returns:
            List[int]: wires
        """
        return [self.tag_to_wire(tag) for tag in self.mode_to_tags(mode)]

    def mode_to_tags(self, mode: int) -> List[int]:
        r"""Returns the tags corresponding to a given mode.
        One to many mapping.

        Args:
            mode (int): mode

        Returns:
            List[int]: tags
        """
        tags = []
        if mode in self.out_modes:
            tags += self.tags["out_L"] + self.tags["out_R"]
        if mode in self.in_modes:
            tags += self.tags["in_L"] + self.tags["in_R"]
        else:
            raise ValueError("Mode not found.")
        return tags

    ########################################
    # Convenience initializers

    @classmethod
    def from_ket(cls, modes: List[int], op: Optional[State] = None):
        r"""Initializes the operation from a ket.

        Args:
            modes: modes of the ket
        """
        return cls(in_modes=[], out_modes=modes, LR=True, name="Ket", op=op)

    @classmethod
    def from_dm(cls, modes: List[int], op: Optional[State] = None):
        r"""Initializes the operation from a density matrix.

        Args:
            modes: modes of the density matrix
        """
        return cls(in_modes=modes, out_modes=modes, LR=True)

    @classmethod
    def from_choi(
        cls, in_modes: List[int], out_modes: List[int], op: Optional[Transformation] = None
    ):
        r"""Initializes the operation from a choi op.

        Args:
            in_modes: modes on which the choi op acts
            out_modes: modes that the choi op outputs
            op: optional choi op
        """
        return cls(in_modes=in_modes, out_modes=out_modes, LR=True)

    @classmethod
    def from_kraus(
        cls, in_modes: List[int], out_modes: List[int], op: Optional[Transformation] = None
    ):
        r"""Initializes the operation from a kraus op.

        Args:
            in_modes: modes on which the kraus op acts
            out_modes: modes that the kraus op outputs
            op: optional kraus op
        """
        return cls(in_modes=in_modes, out_modes=out_modes)

    @classmethod
    def from_unitary(cls, modes: List[int]):
        r"""Initializes the operation from a unitary transformation.

        Args:
            modes: modes on which the transformation acts (which are the same as the output modes)
        """
        return cls(in_modes=modes, out_modes=modes)

    @classmethod
    def from_povm(cls, modes: List[int]):
        r"""Initializes the measurement operation from a povm.

        Args:
            modes: modes of the measurement
        """
        return cls(in_modes=modes, out_modes=modes, LR=True)

    @classmethod
    def from_proj(cls, modes: List[int]):
        r"""Initializes the measurement operation from a ket.

        Args:
            modes: modes of the measurement
        """
        return cls(in_modes=modes, out_modes=[])


class ParallelOperation(Operation):
    pass


class SequentialOperation(Operation):
    pass


# %%


class Circuit:
    name = "Circuit"

    def __init__(self, gate):
        self.gate = gate


def circuit_factory(gate_instance, modes):
    # make a circuit instance wrapping the gate

    return Circuit(gate_instance)


class Sgate:
    name = "Sgate"

    def __new__(cls, *args, **kwargs):
        modes = kwargs.get("modes", None)
        gate_instance = super().__new__(cls)

        # if modes are defined, return an instance of a circuit wrapping the gate
        if modes is not None:
            gate_instance.__init__(*args, **kwargs)
            circ_instance = circuit_factory(gate_instance, modes)
            return circ_instance

        # if no modes, just return the gate
        return gate_instance

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        print(self.name)


print(type(Sgate(modes=[1, 2, 3])))x
# %%

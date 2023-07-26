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

from typing import Any, List, Optional


class Wire:
    r"""
    Represents a wire of a tensor, to be used in a tensor network.

    Attributes:
        id: An integer unique identifier for this wire.
        duality: A string indicating the duality of this wire, either 'L' or 'R'.
        mode: An integer mode for this wire, only wires with the same mode can be connected.
        data: An arbitrary object associated with this wire.
    """
    _id_counter: int = 0

    def __init__(self, is_input: bool, duality: str, mode: Optional[int] = None, data: Any = None):
        r"""
        Initializes a Wire instance.

        Arguments:
            is_input (bool): A boolean value indicating whether this wire is an input wire.
            duality (str): A string indicating the duality of this wire, can only connect to wires of the same duality.
            mode (int): An integer mode for this wire, can only connect to wires with the same mode.
            data (Any): An optional arbitrary object associated with this wire.
        """
        self.id: int = Wire._id_counter * 2 + 1 if is_input else Wire._id_counter * 2
        Wire._id_counter += 1
        self.duality: str = duality
        self.mode: int = mode
        self.data: object = data

    def __repr__(self):
        return f"Wire(id={self.id}, duality={self.duality}, mode={self.mode}, data={self.data})"


class CircuitPart:
    r"""CircuitPart supplies functionality for constructing circuits out of components.
    A CircuitPart does not know about its "physical role", rather it is only concerned with the
    connectivity of its own wires.

    Effectively it enables any Kraus operator constructed out of tensor network operations.

    A CircuitPart provides the following functionality:
        - it can check whether it can be connected to another CircuitPart at a given mode
        - it can return its input and output modes
        - it keeps a list of its wires as Wire objects
        - keeps a reference to the State/Transformation/Measurement associated with this CircuitPart
    """
    _id_counter: int = 0
    _repr_markdown_ = None  # otherwise takes over the repr due to mro

    def __init__(
        self,
        name: str,
        modes_output_L: List[int],
        modes_input_L: List[int],
        modes_output_R: List[int],
        modes_input_R: List[int],
        data: Any = None,
        **kwargs,
    ):
        r"""
        Initializes a CircuitPart instance.

        Arguments:
            name: A string name for this circuit part.
            modes_output_L: A list of output wire modes.
            modes_input_L: A list of input wire modes.
            modes_output_R: A list of dual output wire modes.
            modes_input_R: A list of dual input wire modes.
            data: An optional arbitrary object associated with this circuit part.
            name: A string name for this circuit part.
            kwargs: Additional keyword arguments to pass to the next class in the mro.
        """
        self.id: int = CircuitPart._id_counter
        CircuitPart._id_counter += 1
        self.name: str = name
        self.output_wires_L: List[Wire] = [Wire(False, "L", mode) for mode in modes_output_L]
        self.input_wires_L: List[Wire] = [Wire(True, "L", mode) for mode in modes_input_L]
        self.output_wires_R: List[Wire] = [Wire(False, "R", mode) for mode in modes_output_R]
        self.input_wires_R: List[Wire] = [Wire(True, "R", mode) for mode in modes_input_R]
        self.data: Any = data
        super().__init__(**kwargs)

    @property
    def modes(self) -> Optional[list[int]]:
        r"""Returns the modes that this Operation is defined on.
        For backward compatibility, modes raises a ValueError if modes_in != modes_out."""
        if self.modes_in == self.modes_out:
            return list(self.modes_in)
        elif len(self.modes_in) == 0:  # state
            return list(self.modes_out)
        elif len(self.modes_out) == 0:  # measurement
            return list(self.modes_in)
        else:
            raise ValueError(
                "Different input and output modes. Please refer to modes_in and modes_out."
            )

    @property
    def modes_in(self) -> List[int]:
        "Returns the tuple of input modes that are used by this CircuitPart."
        in_modes = set([w.mode for w in self.input_wires_L]).union(
            [w.mode for w in self.input_wires_R]
        )
        return list(sorted(list(in_modes)))

    @property
    def modes_out(self) -> List[int]:
        "Returns the tuple of output modes that are used by this CircuitPart."
        out_modes = set([w.mode for w in self.output_wires_L]).union(
            [w.mode for w in self.output_wires_R]
        )
        return list(sorted(list(out_modes)))

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, name={self.name}, \
            \n\tinput_wires_L={self.input_wires_L}, \n\tinput_wires_R={self.input_wires_R}, \
            \n\toutput_wires_L={self.output_wires_L}, \n\toutput_wires_R={self.output_wires_R})"

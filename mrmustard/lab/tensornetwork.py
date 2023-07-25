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

"""
This module implements the :class:`.Circuit` class which acts as a representation for quantum circuits.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union


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


class TNTensor:
    """
    Represents a tensor in a tensor network.

    Attributes:
        id: An integer unique identifier for this tensor.
        name: A string name for this tensor, used for human-readable references.
        input_wires_L: A list of left-flavored input wires connected to this tensor.
        input_wires_R: A list of right-flavored input wires connected to this tensor.
        output_wires_L: A list of left-flavored output wires connected to this tensor.
        output_wires_R: A list of right-flavored output wires connected to this tensor.
    """

    _id_counter: int = 0

    def __init__(
        self,
        name: str,
        modes_output_L: List[int],
        modes_input_L: List[int],
        modes_output_R: List[int],
        modes_input_R: List[int],
        data: Any = None,
    ):
        r"""
        Initializes a Tensor instance.

        Arguments:
            name: A string name for this tensor.
            modes_output_L: A list of left-flavored output wire modes.
            modes_input_L: A list of left-flavored input wire modes.
            modes_output_R: A list of right-flavored output wire modes.
            modes_input_R: A list of right-flavored input wire modes.
            data: An optional arbitrary object associated with this tensor.
        """
        self.id: int = TNTensor._id_counter
        TNTensor._id_counter += 1
        self.name: str = name
        self.output_wires_L: List[Wire] = [Wire(False, "L", mode) for mode in modes_output_L]
        self.input_wires_L: List[Wire] = [Wire(True, "L", mode) for mode in modes_input_L]
        self.output_wires_R: List[Wire] = [Wire(False, "R", mode) for mode in modes_output_R]
        self.input_wires_R: List[Wire] = [Wire(True, "R", mode) for mode in modes_input_R]
        self.data: Any = data

    @property
    def input_modes(self) -> List[Wire]:
        r"""Returns a list of all input modes of this tensor."""
        return list(set([w.mode for w in self.input_wires_L + self.input_wires_R]))

    @property
    def output_modes(self) -> List[Wire]:
        r"""Returns a list of all output wires connected to this tensor."""
        return list(set([w.mode for w in self.output_wires_L + self.output_wires_R]))


class TensorNetwork:
    """
    Represents a tensor network, maintaining a collection of tensors and their connections.

    Attributes:
        graph: A networkx.Graph object representing the connections between wires.
        tensors: A dictionary mapping tensor ids to Tensor objects.
        name_to_id: A dictionary mapping tensor names to tensor ids.
    """

    def __init__(self):
        """Initializes a TensorNetwork instance."""
        self.graph: nx.Graph = nx.Graph()
        self.tensors: dict = {}
        self.name_to_id: dict = {}

    def add_tensor(self, tensor: Tensor) -> None:
        """
        Adds a tensor to the network.

        Arguments:
            tensor: The Tensor object to add.
        """
        self.name_to_id[tensor.name] = tensor.id
        self.tensors[tensor.id] = tensor
        self.graph.add_nodes_from(
            [
                (
                    wire.id,
                    {
                        "tensor": tensor.id,
                        "flavor": wire.flavor,
                        "mode": wire.mode,
                        "data": wire.data,
                    },
                )
                for wire in tensor.input_wires + tensor.output_wires
            ]
        )

    def get_tensor(self, identifier: Union[int, str]) -> Tensor:
        """
        Retrieves a tensor from the network by its name or id.

        Arguments:
            identifier: A string or integer identifying the tensor to retrieve.

        Returns:
            The matching Tensor object.
        """
        if isinstance(identifier, int):
            return self.tensors.get(identifier)
        elif isinstance(identifier, str):
            tensor_id = self.name_to_id.get(identifier)
            return self.tensors.get(tensor_id)

    def connect_wires(self, wire1_id: int, wire2_id: int) -> None:
        """
        Connects two wires in the network.

        Arguments:
            wire1_id: The id of the first wire.
            wire2_id: The id of the second wire.

        Raises:
            ValueError: If the wires have the same parity, different flavors, or different modes.
        """
        wire1 = self.graph.nodes[wire1_id]
        wire2 = self.graph.nodes[wire2_id]

        if wire1_id % 2 == wire2_id % 2:
            raise ValueError("Error: Wires have the same parity")
        elif wire1["flavor"] != wire2["flavor"]:
            raise ValueError("Error: Wires have different flavors")
        elif wire1["mode"] != wire2["mode"]:
            raise ValueError("Error: Wires have different modes")
        else:
            self.graph.add_edge(wire1_id, wire2_id)

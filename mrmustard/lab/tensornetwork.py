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

from typing import Union

import networkx as nx

from mrmustard.lab.abstract.circuitpart import CircuitPart


class TensorNetwork:
    r"""
    Represents a tensor network, maintaining a collection of tensors and their connections.

    Attributes:
        graph: A networkx.Graph object representing the connections between wires.
        tensors: A dictionary mapping tensor ids to Tensor objects.
        name_to_id: A dictionary mapping tensor names to tensor ids.
    """

    def __init__(self):
        r"""Initializes a TensorNetwork instance."""
        self.graph: nx.Graph = nx.Graph()
        self.tensors: dict = {}
        self.name_to_id: dict = {}

    def add_tensor(self, tensor: CircuitPart) -> None:
        r"""
        Adds a tensor to the network by adding its wires to the graph.

        Arguments:
            tensor: The Tensor object to add.
        """
        self.name_to_id[tensor.name] = tensor.id
        self.tensors[tensor.id] = tensor
        self.graph.add_nodes_from(
            [
                (
                    wire.id,
                    {"tensor": tensor.id},
                    {"wire": wire},
                )
                for wire in tensor.all_wires
            ]
        )

    def get_tensor(self, identifier: Union[int, str]) -> CircuitPart:
        r"""
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

    def connect_wires(self, wire1_id, wire2_id):
        r"""
        Connects two wires in the network.
        They can belong to different tensors or to the same tensor.

        Arguments:
            wire1_id (int): The id of the first wire.
            wire2_id (int): The id of the second wire

        Raises:
            ValueError: If the wires are not input/output, different duality, or different modes.
        """
        if self.can_connect_wires(wire1_id, wire2_id):
            self.graph.add_edge(wire1_id, wire2_id)

    @classmethod
    def from_circuit(self, circuit):  # these must all be transformations with current Circuit
        for op in circuit.ops:
            pass


class TensorNetworkCircuit(TensorNetwork):
    r"""A restricted version of a TensorNetwork used to enforce certain rules when
    constructing a tensor network of a circuit. For example, only input and output modes can
    be contracted, and the duality and modes must match."""

    def __init__(self):
        r"""Initializes a TensorNetworkCircuit instance."""
        super().__init__()

    def can_connect_wires(self, wire1_id: int, wire2_id: int) -> bool:
        r"""
        Checks whether two wires can be connected.

        Arguments:
            wire1_id: The id of the first wire.
            wire2_id: The id of the second wire.

        Returns:
            Whether the wires can be connected.
        """
        wire1 = self.graph.nodes[wire1_id].wire
        wire2 = self.graph.nodes[wire2_id].wire

        mode = wire1.mode == wire2.mode
        equal_duality = wire1.duality == wire2.duality
        in_out = wire1.is_input != wire2.is_input

        partial_trace = mode and not equal_duality and not in_out
        op = mode and equal_duality and in_out

        return op or partial_trace

    def connect_wires(self, tensor_id_1, tensor_id_2, mode, duality):
        r"""
        Connects two wires in the network.
        They can belong to different tensors or to the same tensor.

        Arguments:
            tensor_id_1 (int): The id of the first tensor.
            tensor_id_2 (int): The id of the second tensor (can be the same as the first tensor)
            mode (int): The mode of the wires to connect.
            duality (str): The duality of the wires to connect.

        Raises:
            ValueError: If the wires are not input/output, different duality, or different modes.
        """
        wire1_id = self.get_wire_id(tensor_id_1, mode, duality)
        wire2_id = self.get_wire_id(tensor_id_2, mode, duality)
        if self.can_connect_wires(wire1_id, wire2_id):
            self.graph.add_edge(wire1_id, wire2_id)

    def get_wire_id(self, tensor_id, mode, duality):
        r"""
        Gets the id of a wire in the network.

        Arguments:
            tensor_id: The id of the tensor.
            mode: The mode of the wire.
            duality: The duality of the wire.

        Returns:
            The id of the wire.
        """
        for wire_id, wire in self.graph.nodes(data=True):
            if wire["tensor"] == tensor_id and wire["mode"] == mode and wire["duality"] == duality:
                return wire_id

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
                        "duality": wire.duality,
                        "mode": wire.mode,
                        "data": wire.data,
                    },
                )
                for wire in tensor.input_wires_L
                + tensor.output_wires_L
                + tensor.input_wires_R
                + tensor.output_wires_R
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

    def connect_wires_by_id(
        self, wire1_id: int, wire2_id: int, check_physical: bool = True
    ) -> None:
        r"""
        Connects two wires in the network.

        Arguments:
            wire1_id: The id of the first wire.
            wire2_id: The id of the second wire.
            check_physical (bool): whether to allow only physical connections (default True).

        Raises:
            ValueError: If the wires are not input/output, different duality, or different modes.
        """
        wire1 = self.graph.nodes[wire1_id]
        wire2 = self.graph.nodes[wire2_id]

        if check_physical:
            if wire1_id % 2 == wire2_id % 2:
                raise ValueError("Error: Wires are not input/output pairs")
            elif wire1["duality"] != wire2["duality"]:
                raise ValueError("Error: Wires have different duality")
            elif wire1["mode"] != wire2["mode"]:
                raise ValueError("Error: Wires are on different modes")

        self.graph.add_edge(wire1_id, wire2_id)

    def connect_wires(self, tensor_id_1, tensor_id_2, mode, duality):
        r"""
        Connects two wires in the network.

        Arguments:
            tensor_id_1: The id of the first tensor.
            tensor_id_2: The id of the second tensor.
            mode: The mode of the wires to connect.
            duality: The duality of the wires to connect.

        Raises:
            ValueError: If the wires are not input/output, different duality, or different modes.
        """
        wire1_id = self.get_wire_id(tensor_id_1, mode, duality)
        wire2_id = self.get_wire_id(tensor_id_2, mode, duality)
        self.connect_wires_by_id(wire1_id, wire2_id)

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

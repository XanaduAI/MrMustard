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

import matplotlib.pyplot as plt
import networkx as nx

from mrmustard import settings
from mrmustard.lab.abstract.circuitpart import CircuitPart, Wire
from mrmustard.math import Math

math = Math()


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
        self.tensors: dict[int, CircuitPart] = {}
        self.wires: dict[int, Wire] = {}
        self.name_to_tensor_id: dict[str, int] = {}  # for quick retrieval of tensors by name
        self.wire_to_tensor: dict[int, int] = {}

    def add_tensor(self, tensor: CircuitPart) -> None:
        r"""
        Adds a tensor to the network by adding its wires to the graph.

        Arguments:
            tensor: The Tensor object to add.
        """
        self.name_to_tensor_id[tensor.name] = tensor.id
        self.tensors[tensor.id] = tensor

        for wire in tensor.all_wires:
            self.graph.add_node(wire.id, tensor_id=tensor.id)
            self.wires[wire.id] = wire
            self.wire_to_tensor[wire.id] = tensor.id

        # for visual purposes, we keep the wires together by adding edges between them, but these do nothing.
        for wire1 in tensor.all_wires:
            for wire2 in tensor.all_wires:
                if wire1.id != wire2.id:
                    self.graph.add_edge(wire1.id, wire2.id, kind="visual")

    def free_wires(self) -> list[Wire]:
        r"""
        Returns the free wires in the network.

        Returns:
            A list of wires that are not going to be contracted.
        """
        # return wires that have no edge of kind "inner_product"
        return [
            wire
            for wire in self.wires.values()
            if not any(
                self.graph.edges[edge]["kind"] == "inner_product"
                for edge in self.graph.edges(wire.id)
            )
        ]

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

    def can_connect_wires(self, wire1_id: int, wire2_id: int) -> bool:
        r"""
        Checks whether two wires can be connected.

        Arguments:
            wire1_id: The id of the first wire.
            wire2_id: The id of the second wire.

        Returns:
            Whether the wires can be connected.
        """
        return True  # override in TensorNetworkCircuit

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

    def __repr__(self) -> str:
        return f"TensorNetwork(graph={self.graph}, tensors=\n{self.tensors})"

    def draw(self, show_cutoffs=False, show_ids=False):
        edge_colors = [
            "blue" if self.graph.edges[e]["kind"] == "inner_product" else "red"
            for e in self.graph.edges
        ]
        positions = {}
        for center, tensor in enumerate(self.tensors.values()):
            for wire in tensor.all_wires:
                positions[wire.id] = (
                    2 * center - (wire.is_input - 0.5),
                    -0.5 * (wire.mode + 0.5 * (wire.LR == "R")),
                )

        node_colors = [self.graph.nodes[n]["tensor_id"] for n in self.graph.nodes]
        labels = dict()
        for n in self.graph.nodes:
            wire = self.wires[n]
            labels[n] = self.tensors[wire.owner_id].short_name
            if show_cutoffs:
                labels[n] += f"\n{wire.cutoff}" if hasattr(wire, "cutoff") else ""
            if show_ids:
                labels[n] += f"\nid={wire.id}" if hasattr(wire, "id") else ""

        widths = [
            2 if self.graph.edges[e]["kind"] == "inner_product" else 4 for e in self.graph.edges
        ]
        node_sizes = (
            1000  # [1200 if self.graph.nodes[n]["tensor_id"] else 1000 for n in self.graph.nodes]
        )

        # Normalize color values to between 0 and 1
        node_norm = [
            (float(i) - min(node_colors)) / (max(node_colors) - min(node_colors))
            for i in node_colors
        ]

        # use figsize = (num_tensors, num_modes)
        plt.figure(figsize=(len(self.tensors), len(set([w.mode for w in self.wires.values()]))))

        nx.draw_networkx_nodes(
            self.graph,
            pos=positions,
            node_color=node_norm,
            node_size=node_sizes,
            cmap=plt.cm.Pastel1,  # use colormap that avoids dark colors
            vmin=0,
            vmax=1,
        )

        nx.draw_networkx_edges(
            self.graph,
            pos=positions,
            width=widths,
            edge_color=edge_colors,
            alpha=0.6,
        )

        nx.draw_networkx_labels(
            self.graph,
            pos=positions,
            labels=labels,
            font_size=10,
            font_color="black",
            font_weight="bold",
        )

        plt.show()


ID = int


class TensorNetworkCircuit(TensorNetwork):
    r"""A restricted version of a TensorNetwork used to enforce certain rules when
    constructing a tensor network of a circuit. For example, only input and output modes can
    be contracted, and the duality and modes must match, unless it's a partial trace operation."""

    def __init__(self):
        r"""Initializes a TensorNetworkCircuit instance."""
        super().__init__()
        self.circuit_input_wires_L = []
        self.circuit_output_wires_L = []
        self.circuit_input_wires_R = []
        self.circuit_output_wires_R = []

    def add_tensor(self, tensor: CircuitPart) -> None:
        super().add_tensor(tensor)

        # we automatically connect the tensor to the circuit if possible
        id_to_remove: list[ID] = []
        for wire_in in tensor.input_wires_L:
            for wire_out in self.circuit_output_wires_L:
                if self.can_connect_wires(wire_in.id, wire_out.id):
                    self.graph.add_edge(wire_in.id, wire_out.id, kind="inner_product")
                    id_to_remove.append(wire_out.id)
                    break
            else:
                # if the wire wasn't connected
                self.circuit_input_wires_L.append(wire_in)
        self.circuit_output_wires_L = [
            wire for wire in self.circuit_output_wires_L if wire.id not in id_to_remove
        ]
        self.circuit_output_wires_L.extend(tensor.output_wires_L)

        id_to_remove = []
        for wire_in in tensor.input_wires_R:
            for wire_out in self.circuit_output_wires_R:
                if self.can_connect_wires(wire_in.id, wire_out.id):
                    self.graph.add_edge(wire_in.id, wire_out.id, kind="inner_product")
                    id_to_remove.append(wire_out.id)
                    break
            else:
                self.circuit_input_wires_R.append(wire_in)
        self.circuit_output_wires_R = [
            wire for wire in self.circuit_output_wires_R if wire.id not in id_to_remove
        ]
        self.circuit_output_wires_R.extend(tensor.output_wires_R)

    def can_connect_wires(self, wire1_id: int, wire2_id: int) -> bool:
        r"""
        Checks whether two wires can be connected.

        Arguments:
            wire1_id: The id of the first wire.
            wire2_id: The id of the second wire.

        Returns:
            Whether the wires can be connected.
        """
        wire1 = self.wires[wire1_id]
        wire2 = self.wires[wire2_id]

        mode = wire1.mode == wire2.mode
        equal_duality = wire1.LR == wire2.LR
        in_out = wire1.is_input != wire2.is_input

        partial_trace = mode and not equal_duality and not in_out
        op = mode and equal_duality and in_out

        return op or partial_trace

    def _add_fock_cutoffs(self):
        # add from states first
        _largest = 0
        for wire in self.wires.values():
            owner = self.tensors[wire.owner_id]
            try:
                wire.cutoff = owner.shape[owner.wire_order(wire.id)]
                _largest = max(_largest, wire.cutoff)
            except AttributeError:  # transformations don't have a shape attribute
                wire.cutoff = None

        # if a node is connected to a node with a cutoff via an inner product it inherits the cutoff:
        for wire in self.wires.values():
            if wire.cutoff is None:
                for neighbor in self.graph.neighbors(wire.id):
                    if self.graph.edges[wire.id, neighbor]["kind"] == "inner_product":
                        wire.cutoff = self.wires[neighbor].cutoff
                        break

        # set all remaining cutoffs to the largest cutoff
        for wire in self.wires.values():
            if wire.cutoff is None:
                wire.cutoff = max(_largest, settings.TN_DEFAULT_BOND_CUTOFF)

    def _write_contraction_ids(self):
        id = 0
        for edge in self.graph.edges:
            if self.graph.edges[edge]["kind"] == "inner_product":
                self.wires[edge[0]].contraction_id = id
                self.wires[edge[1]].contraction_id = id
                id += 1
        # write all the rest
        for wire in self.wires.values():
            if not hasattr(wire, "contraction_id"):
                wire.contraction_id = id
                id += 1

    def _get_opt_einsum_args(self):
        for tensor_id in self.tensors:
            array = self.tensors[tensor_id].fock
            ids = [w.contraction_id for w in self.tensors[tensor_id].all_wires]
            yield array, ids

    def contract(self):
        return math.einsum(*[el for pair in self._get_opt_einsum_args() for el in pair])

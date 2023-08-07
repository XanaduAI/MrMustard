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

from collections import defaultdict
from itertools import product
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx

from mrmustard import settings
from mrmustard.lab.abstract.circuitpart import CircuitPart
from mrmustard.math import Math

math = Math()


class TensorNetwork(CircuitPart):
    r"""
    Represents a tensor network, maintaining a collection of tensors indices and their connections.
    """

    def __init__(self):
        r"""Initializes a TensorNetwork instance."""
        self.graph: nx.Graph = nx.Graph()
        self.tensors: dict[int, CircuitPart] = dict()
        self._two_sided = False
        super().__init__(
            name="TN",
            modes_output_ket=[],
            modes_input_ket=[],
            modes_output_bra=[],
            modes_input_bra=[],
        )

    @property
    def two_sided(self) -> bool:
        "redefine the two_sided property of CircuitPart"
        if self._two_sided:
            return True
        else:
            return super().two_sided

    def _rebuild_TN(self):
        r"""Rebuilds the tensor network."""
        self.graph = nx.Graph()
        super().__init__(name="TN")  # resets the modes too
        for tensor in self.tensors.values():
            self._add_fake_edges(tensor)
        self.connect()

    def add_tensor(self, tensor: CircuitPart):
        r"""
        Adds a tensor to the network by adding a node to the graph for each of its wires.

        Arguments:
            tensor: The Tensor object to add.
        """
        if self.one_sided and tensor.two_sided:
            self._two_sided = True  # we force the TN to be two-sided
            self._rebuild_TN()  # rebuild the TN with both ket/bra available
        self.tensors[tensor.id] = tensor
        for id in tensor.ids:
            self.graph.add_node(id, tensor_id=tensor.id)
        self._add_fake_edges(tensor.ids)

    def connect(self, id1: Optional[int] = None, id2: Optional[int] = None):
        r"""
        Connects two wires specified by id. If called with no arguments it connects all the tensors
        in order, as if they were in a circuit. If self.LR = "L" it only connects
        the tensors on the left side, if self.LR = "R" it only connects the tensors
        on the right side and if self.LR = "LR" it connects on both sides.

        Arguments:
            id1: The id of the first wire.
            id2: The id of the second wire.
        """
        if id1 is None and id2 is None:
            for tensor in self.tensors.values():
                self.connect_tensor_default(tensor)
        else:
            self.graph.add_edge(id1, id2, kind="inner_product")

    def connect_tensor_default(self, tensor):
        r"""Connects tensor to the current TN as it would happen in a circuit."""
        # ket side
        for mode, id in tensor.input.ket.items():
            for mode_tn, id_tn in self.output.ket.items():
                if mode == mode_tn:
                    self.connect(id, id_tn)
                    self.output.ket.pop(mode_tn)
                    break
            else:
                self.input.ket[mode] = id
        for mode, id in tensor.output.ket.items():
            self.output.ket[mode] = id

        # bra side
        if self.two_sided and tensor.one_sided:
            tensor = tensor.adjoint  # tensor^dagger
        for mode, id in tensor.input.bra.items():
            for mode_tn, id_tn in self.output.bra.items():
                if mode == mode_tn:
                    self.connect(id, id_tn)
                    self.output.bra.pop(mode_tn)
                    break
            else:
                self.input.bra[mode] = id
        for mode, id in tensor.output.bra.items():
            self.output.bra[mode] = id

    def _add_fake_edges(self, ids):
        r"""Adds fake edges between all the wires of a tensor.
        For visualization purposes only."""
        for id1, id2 in product(ids, ids):
            if id1 != id2:
                self.graph.add_edge(id1, id2, kind="fake")

    def __repr__(self) -> str:
        return f"TensorNetwork(graph={self.graph}, tensors=\n{self.tensors})"

    def draw(self, show_cutoffs=False, show_ids=False):
        positions = dict()
        for x, tensor in enumerate(self.tensors.values()):
            for mode, id in tensor.input.ket.items():
                positions[id] = (2 * x - 0.5, -2 * mode)
            for mode, id in tensor.input.bra.items():
                positions[id] = (2 * x - 0.5, -2 * mode - 0.5)
            for mode, id in tensor.output.ket.items():
                positions[id] = (2 * x + 0.5, -2 * mode)
            for mode, id in tensor.output.bra.items():
                positions[id] = (2 * x + 0.5, -2 * mode - 0.5)
        edge_colors = [
            "blue" if self.graph.edges[e]["kind"] == "inner_product" else "red"
            for e in self.graph.edges
        ]
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
            node_size=1000,
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

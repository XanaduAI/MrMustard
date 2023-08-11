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

from collections import ChainMap, defaultdict
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx

from mrmustard import settings
from mrmustard.lab.abstract.circuitpart import CircuitPart
from mrmustard.math import Math

math = Math()

ID = int  # just a type alias


class TNTensor(CircuitPart):
    r"""Wrapper for MM objects that allows to add them multiple times to a TN.
    It offers the same CircuitPart functionality as the tensor it wraps,
    but with unique wire ids.
    """

    def __init__(self, circuit_part: CircuitPart):
        # The TNTensor and its wires get unique ids from the CircuitPart init.
        # Note we use output.ket etc so that views work too.
        super().__init__(
            name=circuit_part.name,
            modes_output_ket=circuit_part.output.ket.keys(),
            modes_input_ket=circuit_part.input.ket.keys(),
            modes_output_bra=circuit_part.output.bra.keys(),
            modes_input_bra=circuit_part.input.bra.keys(),
        )
        self.mm_obj = circuit_part
        self._id_map = {id_self: id_mm for id_self, id_mm in zip(self.ids, self.mm_obj.ids)}
        for id in self.ids:
            self[id]["dimension"] = self._dimension(id)

    @property
    def fock(self):
        return math.hermite_renormalized(*self.mm_obj.bargmann(), shape=self.shape)

    @property
    def shape(self):
        return tuple(self[id]["dimension"] for id in self.ids)

    def _dimension(self, id: ID):
        i = self.mm_obj.modes.index(self.mm_obj.mode_from_id(id))
        try:
            return self.mm_obj.cutoffs[i]
        except AttributeError:
            return None


class TensorNetwork:
    r"""
    Represents a tensor network, maintaining a collection of tensors indices and their connections.
    """

    def __init__(self):
        r"""Initializes a TensorNetwork instance."""
        self.graph: nx.Graph = nx.Graph()
        super().__init__(name="TN")

    def __getitem__(self, id: ID) -> dict:
        "get wire dict by id"
        return self.graph.nodes[id]

    @property
    def input(self) -> dict:
        return {
            "ket": {
                wire["mode"]: id
                for id, wire in self.graph.nodes(data=True)
                if wire["direction"] == "in" and wire["type"] == "ket" and not wire["connected"]
            },
            "bra": {
                wire["mode"]: id
                for id, wire in self.graph.nodes(data=True)
                if wire["direction"] == "in" and wire["type"] == "bra" and not wire["connected"]
            },
        }

    @property
    def output(self) -> dict:
        return {
            "ket": {
                wire["mode"]: id
                for id, wire in self.graph.nodes(data=True)
                if wire["direction"] == "out" and wire["type"] == "ket" and not wire["connected"]
            },
            "bra": {
                wire["mode"]: id
                for id, wire in self.graph.nodes(data=True)
                if wire["direction"] == "out" and wire["type"] == "bra" and not wire["connected"]
            },
        }

    def add_tensor(self, tensor: TNTensor):
        r"""
        Adds a tensor to the network by adding a node to the graph for each of its wires.

        Arguments:
            tensor: The Tensor object to add.
        """
        for mode, id in tensor.input.ket.items():
            self.graph.add_node(
                id,
                owner=tensor,
                mode=mode,
                direction="in",
                type="ket",
                connected=False,
                dimension=None,
            )
        for mode, id in tensor.input.bra.items():
            self.graph.add_node(
                id,
                owner=tensor,
                mode=mode,
                direction="in",
                type="bra",
                connected=False,
                dimension=None,
            )
        for mode, id in tensor.output.ket.items():
            self.graph.add_node(
                id,
                owner=tensor,
                mode=mode,
                direction="out",
                type="ket",
                connected=False,
                dimension=None,
            )
        for mode, id in tensor.output.bra.items():
            self.graph.add_node(
                id,
                owner=tensor,
                mode=mode,
                direction="out",
                type="bra",
                connected=False,
                dimension=None,
            )
        self._add_fake_edges(tensor)

    def connect(self, id1: Optional[ID] = None, id2: Optional[ID] = None):
        r"""
        Connects two wires specified by id.
        If called with no arguments it connects all the tensors in order,
        as if they were in a circuit.

        Arguments:
            id1: The id of the first wire.
            id2: The id of the second wire.
        """
        if id1 is None and id2 is None:
            self._connect_network()
        else:
            self._connect_wires(id1, id2)

    def _connect_wires(self, id1: ID, id2: ID):
        # if a dimension exist, use it for both wires and favor the largest
        c1 = self[id1]["dimension"]
        c2 = self[id2]["dimension"]
        dimension = max([c1, c2], key=lambda x: x if x is not None else -1)
        self[id1]["dimension"] = dimension
        self[id2]["dimension"] = dimension

        self.graph.add_edge(id1, id2, kind="inner_product")
        self[id1]["connected"] = True
        self[id2]["connected"] = True

    def _connect_network(self):
        for tensor in self.tensors.values():
            self.connect_tensor_default(tensor)

    def connect_tensor_default(self, tensor: TNTensor):
        r"""Connects tensor to the current TN as it would happen in a circuit."""
        # ket side  # TODO use set intersections
        common_modes = set(tensor.input.ket).intersection(self.output.ket)
        for mode, id in tensor.input.ket.items():
            if mode in common_modes:
                self.connect(self.output.ket[mode], id)
                self.output.ket.pop(mode)
            else:
                self.input.ket[mode] = id
        for mode, id in tensor.output.ket.items():
            self.output.ket[mode] = id

        # bra side
        if (self.has_ket and self.has_bra) and (tensor.only_ket or tensor.only_bra):
            tensor = tensor.adjoint
        elif (self.only_ket and self.only_bra) and (tensor.has_ket and tensor.has_bra):
            self = self.adjoint  # probably wrong
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

    def _add_fake_edges(self, tensor: TNTensor):
        r"""Adds fake edges between all the wires of a tensor.
        For visualization purposes only.

        Arguments:
            id: The id of the tensor.
        """
        for id1, id2 in product(tensor.ids, tensor.ids):
            if id1 != id2:
                self.graph.add_edge(id1, id2, kind="fake")

    def __repr__(self) -> str:
        return f"TensorNetwork(graph={self.graph}, tensors=\n{self.tensors})"

    def draw(self, show_dimensions=True, show_ids=False):
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
        node_colors = [self.graph.nodes[n]["owner_id"] for n in self.graph.nodes]

        labels = dict()
        for id, wire in self.graph.nodes.items():
            labels[id] = self.tensor_by_wire[id].short_name
            if show_dimensions:
                labels[id] += f"\n{wire.dimension}" if hasattr(wire, "dimension") else ""
            if show_ids:
                labels[id] += f"\nid={wire.id}" if hasattr(wire, "id") else ""

        widths = [
            2 if self.graph.edges[e]["kind"] == "inner_product" else 4 for e in self.graph.edges
        ]

        # Normalize color values to between 0 and 1
        node_norm = [
            (float(i) - min(node_colors))
            / (bool(max(node_colors) == min(node_colors)) + max(node_colors) - min(node_colors))
            for i in node_colors
        ]

        # use figsize = (num_tensors, num_modes)
        num_modes = len(set.union(*[tensor.all_modes for tensor in self.tensors.values()]))

        plt.figure(figsize=(len(self.tensors), num_modes + 1))

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

    def _set_wire_dimension(self):
        # set all remaining dimensions to the default
        for id in self.graph.nodes:
            if self.tensor_by_wire[id][id]["dimension"] is None:
                self.tensor_by_wire[id][id]["dimension"] = settings.TN_DEFAULT_BOND_DIMENSION

    def _set_contraction_ids(self):
        id = 0
        # first add same ids to inner products
        for edge in self.graph.edges:
            if self.graph.edges[edge]["kind"] == "inner_product":
                self.tensor_by_wire[edge[0]][edge[0]]["contraction_id"] = id
                self.tensor_by_wire[edge[1]][edge[1]]["contraction_id"] = id
                id += 1
        # then add unique ids to the rest
        for id in self.graph.nodes:
            if "contraction_id" not in self[id]:
                self[id]["contraction_id"] = id
                id += 1

    def _get_opt_einsum_args(self):
        for tensor in self.tensors.values():
            yield tensor.fock, list(self.id_data("contraction_ids"))

    def contract(self):
        self._set_wire_dimension()
        self._set_contraction_ids()
        return math.einsum(*[el for pair in self._get_opt_einsum_args() for el in pair])

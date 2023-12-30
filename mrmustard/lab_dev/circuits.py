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

"""
A class to quantum circuits.
"""

from __future__ import annotations

from typing import Sequence
from functools import reduce
from operator import add
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from mrmustard.physics.representations import Representation, Bargmann

from .circuit_components import CircuitComponent


__all__ = ["Circuit"]


class Circuit:
    r"""
    A quantum circuit.

    Quantum circuits store a list of ``CircuitComponent``s. They can be generated directly by
    specifying a list of components, or alternatively, indirectly using the
    operators ``>>`` and ``<<`` on sequences of ``CircuitComponent``s.

    .. code-block::

        # direct initialization
        g1 = Dgate(1, modes=[0, 1])
        g2 = Dgate(1, modes=[0,])
        g3 = Dgate(1, modes=[1,])
        g4 = Dgate(1, modes=[0, 1, 9])

        circ = Circuit([g1, g2, g2, g3, g4])
        circ.draw();

    .. code-block::

        # indirect initialization
        g1 = Dgate(1, modes=[0, 1])
        g2 = Dgate(1, modes=[0,])
        g3 = Dgate(1, modes=[1,])
        g4 = Dgate(1, modes=[0, 1, 9])

        circ = g1 >> g2 >> g2 >> g3 >> g4
        circ.draw();
    """

    def __init__(self, components: Sequence[CircuitComponent] = []) -> None:
        self.components = list(components)
        self._connect(self.components)

    @staticmethod
    def _connect(components: Sequence[CircuitComponent]):
        r"""Connects all components (sets the same id of connected wire pairs).
        Supports mode reinitialization."""
        for i,c in enumerate(components):
            ket_modes = set(c.wires.output.ket.modes)
            bra_modes = set(c.wires.output.bra.modes)
            for c_ in components[i+1:]:
                common_ket = ket_modes.intersection(c_.wires.input.ket.modes)
                common_bra = bra_modes.intersection(c_.wires.input.bra.modes)
                c.wires[common_ket].output.ket.ids = c_.wires[common_ket].input.ket.ids
                c.wires[common_bra].output.bra.ids = c_.wires[common_bra].input.bra.ids
                ket_modes -= common_ket
                bra_modes -= common_bra
                if not ket_modes and not bra_modes:
                    break
    
    @staticmethod
    def _create_einsum_dict(components: Sequence[CircuitComponent]) -> dict[str, Representation]:
        r"""Creates the einsum dict from the components."""
        einsum: dict[str, Representation] = {}  
        ids = {id:None for c in components for id in c.wires.ids}
        ids_map = {id:i for i,id in enumerate(ids.keys())}  # remap to 0,1,2...
        strings_ints = [[ids_map[id] for id in c.wires.ids] for c in components]
        strings = ["".join([chr(i+97) for i in s]) for s in strings_ints]
        for s,c in zip(strings, components):
            einsum[s] = c.representation
        return einsum

    def _contract_bargmann(self, components: Sequence[CircuitComponent]):
        r"""Contracts the circuit assuming it is made of CV (Bargmann) stuff."""
        # 0. create the einsum dict and the char-id and id-char dicts
        einsum_dict = self._create_einsum_dict(self.components)
        char_id_dict = {char:self.components[i].wires.ids[j] for i,string in enumerate(einsum_dict) for j,char in enumerate(string)}
        id_char_dict = {id:char for char,id in char_id_dict.items()}

        # 1. create the einsum string for the whole contraction
        all_chars = [char for s in einsum_dict for char in s]
        ids = [char_id_dict[char] for char in set(all_chars) if all_chars.count(char) == 1]
        out_bra = reduce(add, [c.wires.bra.output.subset(ids) for c in components])
        in_bra = reduce(add, [c.wires.bra.input.subset(ids) for c in components])
        out_ket = reduce(add, [c.wires.ket.output.subset(ids) for c in components])
        in_ket = reduce(add, [c.wires.ket.input.subset(ids) for c in components])
        out_string = "".join([id_char_dict[id] for id in out_bra.ids + in_bra.ids + out_ket.ids + in_ket.ids])
        einsum_string = ",".join(einsum_dict.keys()) + "->" + out_string

        # 2. use einsum string in path_info to get pair-wise contractions
        import opt_einsum as oe
        shapes = [(2,)*len(s) for s in self.einsum_dict.keys()]
        path_info = oe.contract_path(einsum_string, *shapes, shapes=True, optimize='auto')

        # 3. update the dict as path_info indicates until the whole circuit is contracted
        for _, _, current, _, _ in path_info[1].contraction_list:
            idx_A, idx_B, idx_reorder = self.parse_einsum(current)
            A, (B, out) = current.split(",")[0], current.split(",")[1].split("->")
            new_r = einsum_dict[A][idx_A] @ einsum_dict[B][idx_B]
            new_r = new_r.reorder(idx_reorder)
            einsum_dict[out] = new_r

        # 4. return the result
        return CircuitComponent("", einsum_dict[out_string], modes_out_bra=out_bra.modes, modes_in_bra=in_bra.modes, modes_out_ket=out_ket.modes, modes_in_ket=in_ket.modes)

    @staticmethod
    def parse_einsum(string: str):
        parts, result = string.split('->')
        parts = parts.split(',')
        repeated = set(parts[0]).intersection(parts[1])
        remaining = set(parts[0]).union(parts[1]) - repeated
        return [parts[0].index(i) for i in repeated], [parts[1].index(i) for i in repeated], [result.index(i) for i in remaining]


    def __rshift__(self, other: Circuit | CircuitComponent) -> Circuit:
        r"""
        Returns a ``Circuit`` that contains all the components of ``self`` and ``other``.
        """
        other = Circuit([other]) if isinstance(other, CircuitComponent) else other
        return Circuit(self.components + other.components)

    def __getitem__(self, i: int) -> CircuitComponent:
        r"""
        The component in position ``i`` of this circuit's components.
        """
        return self.components[i]

    def draw(self, layout: str = "spring_layout", figsize: tuple[int, int] = (10, 6)):
        r"""Draws the components in this circuit in the style of a tensor network.

        Args:
            layout: The layout method. Must be one of the methods in ``nx.drawing.layout``.
            figsize: The size of the returned figure.

        Returns:
            A figure showing the tensor network.
        """
        self._connect(self.components)
        try:
            fn_layout = getattr(nx.drawing.layout, layout)
        except AttributeError:
            msg = f"Invalid layout {layout}."
            # pylint: disable=raise-missing-from
            raise ValueError(msg)

        # initialize empty lists and dictionaries used to store metadata
        component_labels = {}
        mode_labels = {}
        node_size = []
        node_color = []

        # initialize three graphs--one to store nodes and edges, two to keep track of arrows
        graph = nx.Graph()
        arrows_in = nx.Graph()
        arrows_out = nx.Graph()

        for i, component in enumerate(self.components):
            component_id = component.name + str(i)
            graph.add_node(component_id)
            component_labels[component_id] = component.name
            mode_labels[component_id] = ""
            node_size.append(150)
            node_color.append("red")
            wires_in = list(zip(component.wires.input.modes, component.wires.input.ids))
            wires_out = list(zip(component.wires.output.modes, component.wires.output.ids))
            wires = wires_in + wires_out
            for mode, wire in wires:
                if wire not in graph.nodes:
                    node_size.append(0)
                    node_color.append("white")
                    component_labels[wire] = ""
                    mode_labels[wire] = mode

                graph.add_node(wire)
                graph.add_edge(wire, component_id)
                if (mode, wire) in wires_in:
                    arrows_in.add_edge(component_id, wire)
                else:
                    arrows_out.add_edge(component_id, wire)

        pos = fn_layout(graph)
        pos_labels = {k: v + np.array([0.0, 0.05]) for k, v in pos.items()}

        fig = plt.figure(figsize=figsize)
        nx.draw_networkx_nodes(
            graph, pos, edgecolors="gray", alpha=0.9, node_size=node_size, node_color=node_color
        )
        nx.draw_networkx_edges(graph, pos, edge_color="lightgreen", width=4, alpha=0.6)
        nx.draw_networkx_edges(
            arrows_in,
            pos,
            edge_color="darkgreen",
            width=0.5,
            arrows=True,
            arrowsize=10,
            arrowstyle="<|-",
        )
        nx.draw_networkx_edges(
            arrows_out,
            pos,
            edge_color="darkgreen",
            width=0.5,
            arrows=True,
            arrowsize=10,
            arrowstyle="-|>",
        )
        nx.draw_networkx_labels(
            graph,
            pos=pos_labels,
            labels=component_labels,
            font_size=12,
            font_color="black",
            font_family="serif",
        )
        nx.draw_networkx_labels(
            graph,
            pos=pos_labels,
            labels=mode_labels,
            font_size=12,
            font_color="black",
            font_family="FreeMono",
        )

        plt.title("Mr Mustard Circuit")
        return fig
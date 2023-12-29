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

    def __init__(self, components: Sequence[CircuitComponent] = [], connections: dict[int,int] = {}) -> None:
        self.components = list(components)
        self.components_by_id = {id:i for i,c in enumerate(self.components) for id in c.wires.ids}
        self.connections = self._connect(self.components, connections)

    @staticmethod
    def _connect(components: Sequence[CircuitComponent], connections: dict[int,int]) -> dict[int,int]:
        r"""Connects all components (sets their wires.connection[id] value) and returns the updated dictionary of connections.
        Supports mode reinitialization."""
        for i,c in enumerate(components):
            ket_modes = set(m for m in c.wires.output.ket.modes if c.wires.output.ket[m].ids[0] not in connections)
            bra_modes = set(m for m in c.wires.output.bra.modes if c.wires.output.bra[m].ids[0] not in connections)
            for c_ in components[i+1:]:
                common_ket = ket_modes & c_.wires.input.ket.modes
                common_bra = bra_modes & c_.wires.input.bra.modes
                for m in common_ket:
                    id = c.wires[m].output.ket.ids[0]
                    if id not in connections:
                        c.wires.connections[id] = connections[id] = c_.wires[m].input.ket.ids[0]
                for m in common_bra:
                    id = c.wires[m].output.bra.ids[0]
                    if id not in connections:
                        c.wires.connections[id] = connections[id] = c_.wires[m].input.bra.ids[0]
                ket_modes -= common_ket
                bra_modes -= common_bra
                if not ket_modes and not bra_modes:
                    break
        return connections

    def _contract_CV(self):
        r"""Contracts the circuit assuming it is made of CV (Bargmann) stuff."""
        r = self.components[0].representation
        order = []
        for c in self.components[1:]:
            r = r & c.representation  # tensor product 
            order += [c.wires.index(id) for id in self.connections if id in c.wires.ids]

        

    def __rshift__(self, other: Circuit | CircuitComponent) -> Circuit:
        r"""
        Returns a ``Circuit`` that contains all the components of ``self`` and ``other``.
        """
        other = Circuit([other]) if isinstance(other, CircuitComponent) else other
        return Circuit(self.components + other.components, self.connections | other.connections)

    def __getitem__(self, idx: int) -> CircuitComponent:
        r"""
        The component in position ``idx`` of this circuit's components.
        """
        return self.components[idx]

    def draw(self, layout: str = "spring_layout", figsize: tuple[int, int] = (10, 6)):
        r"""Draws the components in this circuit in the style of a tensor network.

        Args:
            layout: The layout method. Must be one of the methods in ``nx.drawing.layout``.
            figsize: The size of the returned figure.

        Returns:
            A figure showing the tensor network.
        """
        components = connect(self.components)
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

        for idx, component in enumerate(components):
            component_id = component.name + str(idx)
            graph.add_node(component_id)
            component_labels[component_id] = component.name
            mode_labels[component_id] = ""
            node_size.append(150)
            node_color.append("red")

            wires_in = [
                (m, w)
                for m, w in list(component.wires.in_ket.items())
                + list(component.wires.in_bra.items())
                if w
            ]
            wires_out = [
                (m, w)
                for m, w in list(component.wires.out_ket.items())
                + list(component.wires.out_bra.items())
                if w
            ]
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


class Network:
    r"""
    A network of components.
    
    Supports methods like 
    """
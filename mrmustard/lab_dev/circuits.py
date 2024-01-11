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

from typing import Iterable, Sequence, Union

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from .circuit_components import CircuitComponent, connect

__all__ = [
    "Circuit",
]


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

    def __init__(self, components=Union[CircuitComponent, Sequence[CircuitComponent]]) -> None:
        if not isinstance(components, Iterable):
            components = list[components]

        self._components = [c.light_copy() for c in components]

    @property
    def components(self) -> Sequence[CircuitComponent]:
        r"""
        The components in this circuit.
        """
        return self._components

    def __rshift__(self, other: Union[CircuitComponent, Circuit]) -> Circuit:
        r"""
        Returns a ``Circuit`` that contains all the components of ``self``, as well as
        a light copy of ``other`` (or light copies of ``other.components`` if ``other`` is
        a ``Circuit``).
        """
        if isinstance(other, CircuitComponent):
            other = Circuit([other])

        ret = Circuit(self.components + other.components)
        return ret

    def __getitem__(self, idx: int) -> CircuitComponent:
        r"""
        The component in position ``idx`` of this circuit's components.
        """
        return self._components[idx]

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

            wires = component.wires
            wires_in = [
                (m, w)
                for m, w in list(zip(wires.modes, wires.input.ket.ids))
                + list(zip(wires.modes, wires.input.bra.ids))
                if w
            ]
            wires_out = [
                (m, w)
                for m, w in list(zip(wires.modes, wires.output.ket.ids))
                + list(zip(wires.modes, wires.output.bra.ids))
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

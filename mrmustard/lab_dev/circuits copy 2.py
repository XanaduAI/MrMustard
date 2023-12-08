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
A base class for the components of quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from .circuit_components import CircuitComponent
from .wires import Wires


def to_circuit(components: Optional[Sequence[CircuitComponent]] = None) -> Circuit:
    r"""
    Returns a circuit with given components.

    It does not check whether these components are connected, hence it should be used with
    care (and used only by Mr Mustard developers).

    Arguments:
        components: The components of the returned circuit.
    """
    ret = Circuit()
    ret._components = components
    return ret


class Circuit:
    r"""
    A quantum circuit.

    Quantum circuits can be generated using the operators ``>>`` and ``<<`` to connect
    sequences of ``CircuitComponent``s.

    .. code-block::

        g1 = Dgate(x=0.5, y=0.3)
        g2 = Sgate(0.1)
        circuit == g1 >> g2
    """

    def __init__(
        self,
    ) -> None:
        self._components = []

    @property
    def components(self) -> Sequence[CircuitComponent]:
        return self._components

    def __rshift__(self, other: Union[CircuitComponent, Circuit]) -> Circuit:
        r"""
        Returns a ``Circuit`` that contains all the components of ``self``, as well as
        a light copy of ``other`` (or light copies of ``other.components`` if ``other`` is
        a ``Circuit``) correctly connected.
        """
        if isinstance(other, CircuitComponent):
            other = to_circuit([other.light_copy()])

        ret = self.components
        for component in other.components:
            connected_component = (ret[-1] >> component)[1]
            ret = to_circuit(ret + [connected_component])
        return ret

    def __getitem__(self, idx: int) -> CircuitComponent:
        return self._components[idx]

    def draw(self, layout: str = "spring_layout", figsize: tuple[int, int] = (10, 6)):
        r"""Draws the components in this circuit in the style of a tensor network.

        Args:
            layout: The layout method. Must be one of the methods in ``nx.drawing.layout``.
            figsize: The size of the returned figure.

        Returns:
            A figure showing the tensor network.
        """
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

        for idx, component in enumerate(self.components):
            component_id = component.name + str(idx)
            graph.add_node(component_id)
            component_labels[component_id] = component.name
            mode_labels[component_id] = ""
            node_size.append(150)
            node_color.append("red")

            wires_in = (
                component.wires.in_ket.items
                + component.wires.in_bra.items
            )
            wires_out = (
                component.wires.out_ket.items
                + component.wires.out_bra.items
            )
            wires = wires_in + wires_out
            for wire in wires:
                wire_id = wire.id
                if wire_id not in graph.nodes:
                    node_size.append(0)
                    node_color.append("white")
                    component_labels[wire_id] = ""
                    mode_labels[wire_id] = wire.mode

                graph.add_node(wire_id)
                graph.add_edge(wire_id, component_id)
                if wire in wires_in:
                    arrows_in.add_edge(component_id, wire_id)
                else:
                    arrows_out.add_edge(component_id, wire_id)

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
        plt.show()
        return fig

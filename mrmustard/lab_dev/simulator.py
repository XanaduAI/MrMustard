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
Simulators for quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .circuit_components import CircuitComponent
from .circuits import Circuit

__all__ = ["Simulator"]


class Simulator:
    r"""
    A simulator for quantum circuits.

    The simulation is carried out by contracting the components of ``circuit`` in pairs, until only
    one component is left and returned. If no ``path`` is given, the order in which these contractions
    are performed is chosen automatically and may be suboptimal (explain when that may be). The ``path``
    input allows customising the contraction order and potentially speed up the simulation.

    When a ``path`` of the type ``[(i, j), (l, m), ...]`` is given, the simulator creates a dictionary
    of the type ``{0: c_0, ..., N: c_N}``, where ``[c_0, .., c_N]`` is the ``circuit.component`` list.
    Then:

    * The two components ``c_i`` and ``c_j`` in positions ``i`` and ``j`` are contracted. ``c_i`` is
        replaced by the resulting component ``c_j >> c_j``, while ``c_j`` popped.
    * The two components ``c_l`` and ``c_m`` in positions ``l`` and ``m`` are contracted. ``c_l`` is
        replaced by the resulting component ``c_l >> c_m``, while ``c_l`` is popped.
    * Et cetera.

    When all the contractions are performed, only one component remains in the dictionary, and this
    component is returned.
    """

    def run(
        self, circuit: Circuit, path: Optional[Sequence[tuple[int, int]]] = None
    ) -> CircuitComponent:
        r"""
        Runs the simulations of the given circuit.

        Arguments:
            circuit: The circuit to simulate.
            path: A list of tuples indicating the contraction path.

        Returns:
            A circuit component representing the entire circuit.
        """
        ret = {i: c for i, c in enumerate(circuit)}

        for idx0, idx1 in path:
            ret[idx0] = ret[idx0] >> ret.pop(idx1)

        return list(ret.values())[0]


# class Circuit:
#     r"""
#     A quantum circuit.

#     Quantum circuits store a list of ``CircuitComponent``s. They can be generated directly by
#     specifying a list of components, or alternatively, indirectly using the
#     operators ``>>`` and ``<<`` on sequences of ``CircuitComponent``s.

#     .. code-block::

#         # direct initialization
#         g1 = Dgate(1, modes=[0, 1])
#         g2 = Dgate(1, modes=[0,])
#         g3 = Dgate(1, modes=[1,])
#         g4 = Dgate(1, modes=[0, 1, 9])

#         circ = Circuit([g1, g2, g2, g3, g4])
#         circ.draw();

#     .. code-block::

#         # indirect initialization
#         g1 = Dgate(1, modes=[0, 1])
#         g2 = Dgate(1, modes=[0,])
#         g3 = Dgate(1, modes=[1,])
#         g4 = Dgate(1, modes=[0, 1, 9])

#         circ = g1 >> g2 >> g2 >> g3 >> g4
#         circ.draw();
#     """

#     def __init__(
#         self,
#         components: Union[CircuitComponent, Sequence[CircuitComponent]],
#     ) -> None:
#         if not isinstance(components, Iterable):
#             components = list[components]

#         self._components = [c.light_copy() for c in components]
#         self._path = []

#     @property
#     def components(self) -> Sequence[CircuitComponent]:
#         r"""
#         The components in this circuit.
#         """
#         return self._components

#     @property
#     def path(self) -> Union[list[tuple[int, int]], None]:
#         r"""
#         An list describing a contraction path for this circuit.

#         When a path is attached to a circuit, the ``Simulator`` follows the given path to perform
#         the contractions. 
        
#         In more detail, when a circuit with components ``[c_0, .., c_N]`` has a path of the type
#         ``[(i, j), (l, m), ...]``, the simulator creates a dictionary of the type
#         ``{0: c_0, ..., N: c_N}``. Then:

#         * The two components ``c_i`` and ``c_j`` in positions ``i`` and ``j`` are contracted. ``c_i`` is
#             replaced by the resulting component ``c_j >> c_j``, while ``c_j`` popped.
#         * The two components ``c_l`` and ``c_m`` in positions ``l`` and ``m`` are contracted. ``c_l`` is
#             replaced by the resulting component ``c_l >> c_m``, while ``c_l`` is popped.
#         * Et cetera.

#         When all the contractions are performed, only one component remains in the dictionary, and this
#         component is returned.
#         """
#         return self._path

#     @path.setter
#     def path(self, value: list[tuple[int, int]]) -> None:
#         r"""
#         Setter for ``path``.
#         """
#         self._path = value

#     def check_path(self):
#         r"""
#         Returns a dictionary mapping the available contraction indices to the corresponding
#         components.

#         Raises:
#             ValueError: If ``circuit.path`` contains invalid contractions.
#         """
#         remaining = {i: Circuit([c]) for i, c in enumerate(self.components)}
#         for idx0, idx1 in self.path:
#             try:
#                 left = remaining[idx0].components
#                 right = remaining.pop(idx1).components
#                 remaining[idx0] = Circuit(left + right)
#             except KeyError:
#                 wrong_key = idx0 if idx0 not in remaining else idx1
#                 msg = f"index {wrong_key} in pair ({idx0}, {idx1}) is invalid."
#                 raise ValueError(msg)
        
#         for idx, circ in remaining.items():
#             print(f"\nindex: {idx}")
#             print(f"{circ}\n")

#     def generate_path(self, strategy: Optional[str] = None) -> None:
#         r"""
#         Generates a path and attaches it to this circuit.
        
#         The available strategies are:
#             * ``left_to_right``: The first two components are contracted together, then the
#                 resulting component is contracted with the third one, et cetera.
#             * ``opt_einsum``: Uses ``opt_einsum`` to generate a ore optimal path.

#         Args:
#             strategy: The strategy used to generate the path.
#         """
#         strategy = strategy or "left_to_right"
        
#         if strategy == "left_to_right":
#             self.path = [(0, i) for i in range(1, len(self))]
#         else:
#             msg = f"Strategy ``{strategy}`` is not available."
#             raise ValueError(msg)

#     def __rshift__(self, other: Union[CircuitComponent, Circuit]) -> Circuit:
#         r"""
#         Returns a ``Circuit`` that contains all the components of ``self``, as well as
#         a light copy of ``other`` (or light copies of ``other.components`` if ``other`` is
#         a ``Circuit``).
#         """
#         if isinstance(other, CircuitComponent):
#             other = Circuit([other])

#         ret = Circuit(self.components + other.components)
#         return ret

#     def __getitem__(self, idx: int) -> CircuitComponent:
#         r"""
#         The component in position ``idx`` of this circuit's components.
#         """
#         return self._components[idx]

#     def draw(self, layout: str = "spring_layout", figsize: tuple[int, int] = (10, 6)):
#         r"""Draws the components in this circuit in the style of a tensor network.

#         Args:
#             layout: The layout method. Must be one of the methods in ``nx.drawing.layout``.
#             figsize: The size of the returned figure.

#         Returns:
#             A figure showing the tensor network.
#         """
#         components = self.components
#         for component in components:
#             if component.wires.bra:
#                 components = add_bra(components)
#                 break
#         components = connect(components)

#         try:
#             fn_layout = getattr(nx.drawing.layout, layout)
#         except AttributeError:
#             msg = f"Invalid layout {layout}."
#             # pylint: disable=raise-missing-from
#             raise ValueError(msg)

#         # initialize empty lists and dictionaries used to store metadata
#         component_labels = {}
#         mode_labels = {}
#         node_size = []
#         node_color = []

#         # initialize three graphs--one to store nodes and edges, two to keep track of arrows
#         graph = nx.Graph()
#         arrows_in = nx.Graph()
#         arrows_out = nx.Graph()

#         for idx, component in enumerate(components):
#             component_id = component.name + str(idx)
#             graph.add_node(component_id)
#             component_labels[component_id] = component.name
#             mode_labels[component_id] = ""
#             node_size.append(150)
#             node_color.append("red")

#             wires = component.wires
#             wires_in = [
#                 (m, w)
#                 for m, w in list(zip(wires.modes, wires.input.ket.ids))
#                 + list(zip(wires.modes, wires.input.bra.ids))
#                 if w
#             ]
#             wires_out = [
#                 (m, w)
#                 for m, w in list(zip(wires.modes, wires.output.ket.ids))
#                 + list(zip(wires.modes, wires.output.bra.ids))
#                 if w
#             ]
#             wires = wires_in + wires_out
#             for mode, wire in wires:
#                 if wire not in graph.nodes:
#                     node_size.append(0)
#                     node_color.append("white")
#                     component_labels[wire] = ""
#                     mode_labels[wire] = mode

#                 graph.add_node(wire)
#                 graph.add_edge(wire, component_id)
#                 if (mode, wire) in wires_in:
#                     arrows_in.add_edge(component_id, wire)
#                 else:
#                     arrows_out.add_edge(component_id, wire)

#         pos = fn_layout(graph)
#         pos_labels = {k: v + np.array([0.0, 0.05]) for k, v in pos.items()}

#         fig = plt.figure(figsize=figsize)
#         nx.draw_networkx_nodes(
#             graph, pos, edgecolors="gray", alpha=0.9, node_size=node_size, node_color=node_color
#         )
#         nx.draw_networkx_edges(graph, pos, edge_color="lightgreen", width=4, alpha=0.6)
#         nx.draw_networkx_edges(
#             arrows_in,
#             pos,
#             edge_color="darkgreen",
#             width=0.5,
#             arrows=True,
#             arrowsize=10,
#             arrowstyle="<|-",
#         )
#         nx.draw_networkx_edges(
#             arrows_out,
#             pos,
#             edge_color="darkgreen",
#             width=0.5,
#             arrows=True,
#             arrowsize=10,
#             arrowstyle="-|>",
#         )
#         nx.draw_networkx_labels(
#             graph,
#             pos=pos_labels,
#             labels=component_labels,
#             font_size=12,
#             font_color="black",
#             font_family="serif",
#         )
#         nx.draw_networkx_labels(
#             graph,
#             pos=pos_labels,
#             labels=mode_labels,
#             font_size=12,
#             font_color="black",
#             font_family="FreeMono",
#         )

#         plt.title("Mr Mustard Circuit")
#         return fig
    
#     def __len__(self):
#         r"""
#         The number of components in this circuit.
#         """
#         return len(self.components)
    
#     def __repr__(self) -> str:
#         r"""
#         A string-based graphic representation of this circuit.
#         """
#         components = self.components
#         modes = set(sorted([m for c in components for m in c.modes]))
        
#         # create a dictionary mapping modes to heigth in the drawing, where heigth ``0``
#         # corrsponds to the top mode, heigth ``1`` to the second top mode, etc.
#         heigths = [h for h in range(len(modes))]
#         modes_to_heigth = {m: h for m, h in zip(modes, heigths)}

#         modes_str = [f"{mode}: " for mode in modes]
#         modes_str = [s.rjust(max(len(s) for s in modes_str), " ") for s in modes_str]

#         # generate a dictionary with the graphical representation, heigth by heigth
#         repr = {h: modes_str[h] for h in range(len(modes_str))}

#         # generate a dictionary to map x-axis coordinates to the components drawn at those
#         # coordinates
#         layers = defaultdict(list)
#         x = 0
#         for c1 in components:
#             # if a component would overlap, increase the x-axis coordinate
#             span_c1 = set(range(min(c1.modes), max(c1.modes) + 1))
#             for c2 in layers[x]:
#                 span_c2 = set(range(min(c2.modes), max(c2.modes) + 1))
#                 if span_c1.intersection(span_c2):
#                     x += 1
#                     break
#             # add component to the dictionary
#             layers[x].append(c1)

#         for layer in layers.values():
#             for h in heigths:
#                 repr[h] += "──"

#             layer_str = {h: "" for h in heigths}
#             for c in layer:
#                 # add symbols indicating the extent of a given object
#                 min_heigth = min(modes_to_heigth[m] for m in c.modes)
#                 max_heigth = max(modes_to_heigth[m] for m in c.modes)
#                 if max_heigth - min_heigth > 0:
#                     layer_str[min_heigth] = "╭"
#                     layer_str[max_heigth] = "╰"
#                     for h in range(min_heigth + 1, max_heigth):
#                         layer_str[h] = "├" if h in c.modes else "|"  # other option: ┼

#                 # add control for controlled gates
#                 control = []
#                 if c.__class__.__qualname__ in ["BSgate", "MZgate", "CZgate", "CXgate"]:
#                     control = [c.modes[0]]
#                 label = c.name
#                 param_string = c.parameter_set.to_string(settings.CIRCUIT_DECIMALS)
#                 # if param_string == "":
#                 #     param_string = str(len(c.modes))
#                 label += "" if not param_string else "(" + param_string + ")"

#                 for m in c.modes:
#                     layer_str[modes_to_heigth[m]] += "•" if m in control else label

#             max_label_len = max(len(s) for s in layer_str.values())
#             for h in heigths:
#                 repr[h] += layer_str[h].ljust(max_label_len, "─")

#         return "\n".join(list(repr[modes_to_heigth[m]] for m in modes))

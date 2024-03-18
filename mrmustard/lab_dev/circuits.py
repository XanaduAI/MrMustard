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

from collections import defaultdict
from typing import Optional, Sequence, Union

from mrmustard import math
from .circuit_components import CircuitComponent

__all__ = ["Circuit"]


class Circuit:
    r"""
    A quantum circuit.

    Quantum circuits store a list of ``CircuitComponent``s.

    .. code-block::

        >>> from mrmustard.lab_dev import BSgate, Sgate, Vacuum, Circuit

        >>> vac = Vacuum([0, 1, 2])
        >>> s01 = Sgate([0, 1], r=[0.1, 0.2])
        >>> bs01 = BSgate([0, 1])
        >>> bs12 = BSgate([1, 2])

        >>> components = [vac, s01, bs01, bs12]
        >>> circ = Circuit(components)
        >>> assert circ.components == components

    New components (or entire circuits) can be appended using the ``>>`` operator.

    .. code-block::

        >>> from mrmustard.lab_dev import BSgate, Sgate, Vacuum, Circuit

        >>> vac = Vacuum([0, 1, 2])
        >>> s01 = Sgate([0, 1], r=[0.1, 0.2])
        >>> bs01 = BSgate([0, 1])
        >>> bs12 = BSgate([1, 2])

        >>> circ1 = Circuit() >> vac >> s01
        >>> circ2 = Circuit([bs01]) >> bs12
        >>> assert circ1 >> circ2 == Circuit([vac, s01, bs01, bs12])

    Args:
        components: A list of circuit components.
    """

    def __init__(self, components: Optional[Sequence[CircuitComponent]] = None) -> None:
        self._components = components or []
        self._path = []

    @property
    def components(self) -> Sequence[CircuitComponent]:
        r"""
        The components in this circuit.
        """
        return self._components
    
    @property
    def path(self) -> list[tuple[int, int]]:
        r"""
        A list describing the desired contraction path for this circuit.

        When a path specified, the ``Simulator`` follows it the given path to perform
        the contractions. 
        
        In more detail, when a circuit with components ``[c_0, .., c_N]`` has a path of the type
        ``[(i, j), (l, m), ...]``, the simulator creates a dictionary of the type
        ``{0: c_0, ..., N: c_N}``. Then:

        * The two components ``c_i`` and ``c_j`` in positions ``i`` and ``j`` are contracted. ``c_i`` is
            replaced by the resulting component ``c_j >> c_j``, while ``c_j`` popped.
        * The two components ``c_l`` and ``c_m`` in positions ``l`` and ``m`` are contracted. ``c_l`` is
            replaced by the resulting component ``c_l >> c_m``, while ``c_l`` is popped.
        * Et cetera.

        When all the contractions are performed, only one component remains in the dictionary, and this
        component is returned.
        """
        return self._path

    def __eq__(self, other: Circuit) -> bool:
        return self.components == other.components

    def __getitem__(self, idx: int) -> CircuitComponent:
        r"""
        The component in position ``idx`` of this circuit's components.
        """
        return self._components[idx]

    def __len__(self):
        r"""
        The number of components in this circuit.
        """
        return len(self.components)

    def __rshift__(self, other: Union[CircuitComponent, Circuit]) -> Circuit:
        r"""
        Returns a ``Circuit`` that contains all the components of ``self`` as well as
        ``other`` if ``other`` is a ``CircuitComponent``, or ``other.components`` if
        ``other`` is a ``Circuit``).
        """
        if isinstance(other, CircuitComponent):
            other = Circuit([other])
        return Circuit(self.components + other.components)

    # pylint: disable=too-many-branches,too-many-statements
    def __repr__(self) -> str:
        r"""
        A string-based representation of this component.
        """

        def component_to_str(comp: CircuitComponent) -> str:
            r"""
            Generates a list string-based representation for the given component.

            If ``comp`` is not a controlled gate, the list contains as many elements modes as in
            ``comp.modes``. For example, if ``comp=Sgate([0, 1, 5], r=[0.1, 0.2, 0.5])``, it returns
            ``['Sgate(0.1,0.0)', 'Sgate(0.2,0.0)', 'Sgate(0.5,0.0)']``.

            If ``comp`` is a controlled gate, the list contains the string that needs to be added to
            the target mode. For example, if``comp=BSgate([0, 1], 1, 2)``, it returns
            ``['BSgate(0.0,0.0)']``.

            Args:
                comp: A circuit component.
            """
            cc_name = comp.name or "CC"
            if not comp.wires.input:
                cc_name = f"◖{cc_name}◗"
            if not comp.wires.output:
                cc_name = f"|{cc_name})="

            if comp.parameter_set.names:
                values = []
                for name in comp.parameter_set.names:
                    param = comp.parameter_set.constants.get(
                        name
                    ) or comp.parameter_set.variables.get(name)
                    new_values = math.atleast_1d(param.value)
                    if len(new_values) == 1 and cc_name not in control_gates:
                        new_values = math.tile(new_values, (len(comp.modes),))
                    values.append(
                        new_values.numpy() if math.backend.name == "tensorflow" else new_values
                    )
                return [cc_name + str(l).replace(" ", "") for l in list(zip(*values))]
            # some components have an empty parameter set
            return [cc_name for _ in range(len(comp.modes))]

        if len(self) == 0:
            return ""

        components = self.components
        modes = set(sorted([m for c in components for m in c.modes]))
        n_modes = len(modes)

        # update this when new controlled gates are added
        control_gates = ["BSgate", "MZgate", "CZgate", "CXgate"]

        # create a dictionary ``lines`` mapping modes to the heigth of the corresponding line
        #  in the drawing, where:
        # - heigth ``0`` is the mode with smallest index and drawn on the top line
        # - height ``1`` is the second mode from the top
        # - etc.
        lines = {m: h for h, m in enumerate(modes)}

        # create a dictionary ``wires`` that maps height ``h`` to "──" if the line contains
        # a mode, or to "  " if the line does not contain a mode
        wires = {h: "  " for h in range(n_modes)}

        # generate a dictionary to map x-axis coordinates to the components drawn at
        # those coordinates
        layers = defaultdict(list)
        x = 0
        for c1 in components:
            # if a component would overlap, increase the x-axis coordinate
            span_c1 = set(range(min(c1.modes), max(c1.modes) + 1))
            for c2 in layers[x]:
                span_c2 = set(range(min(c2.modes), max(c2.modes) + 1))
                if span_c1.intersection(span_c2):
                    x += 1
                    break
            # add component to the dictionary
            layers[x].append(c1)

        # store the returned drawing in a dictionary mapping heigths to strings
        drawing_dict = {height: "" for height in range(n_modes)}

        # loop through the layers and add the components to ``drawing_dict``
        for layer in layers.values():
            for comp in layer:
                # there are two types of components: the controlled gates, and all the other ones
                if comp.name in control_gates:
                    control = min(lines[m] for m in comp.modes)
                    target = max(lines[m] for m in comp.modes)

                    # update ``wires`` and start the line with "──"
                    wires[control] = "──"
                    wires[target] = "──"
                    drawing_dict[control] += "──"
                    drawing_dict[target] += "──"

                    drawing_dict[control] += "╭"
                    drawing_dict[target] += "╰"
                    for h in range(target + 1, control):
                        drawing_dict[h] += "├" if h in comp.modes else "|"

                    drawing_dict[control] += "•"
                    drawing_dict[target] += component_to_str(comp)[0]
                else:
                    labels = component_to_str(comp)
                    for i, m in enumerate(comp.modes):
                        # update ``wires`` and start the line with "──" or "  "
                        if comp.wires.input.modes:
                            wires[lines[m]] = "──"
                        drawing_dict[lines[m]] += wires[lines[m]]

                        # draw the label
                        drawing_dict[lines[m]] += labels[i]

                        # update ``wires`` again
                        if comp.wires.output.modes:
                            wires[lines[m]] = "──"
                        else:
                            wires[lines[m]] = "  "

            # ensure that all the strings in the final drawing have the same lenght
            max_len = max(len(v) for v in drawing_dict.values())
            for h in range(n_modes):
                drawing_dict[h] = drawing_dict[h].ljust(max_len, wires[h][0])

                # add a special character to mark the end of the layer
                drawing_dict[h] += "//"

        # break the drawing in chunks of length <90 characters that can be
        # drawn on top of each other
        for h in range(n_modes):
            splits = drawing_dict[h].split("//")
            drawing_dict[h] = [splits[0]]
            for split in splits[1:]:
                if len(drawing_dict[h][-1] + split) < 90:
                    drawing_dict[h][-1] += split
                else:
                    drawing_dict[h].append(split)
        n_chunks = len(drawing_dict[0])

        # every chunk starts with a recap of the modes
        chunk_start = [f"mode {mode}:   " for mode in modes]
        chunk_start = [s.rjust(max(len(s) for s in chunk_start), " ") for s in chunk_start]

        # generate the drawing
        ret = ""
        for chunk_idx in range(n_chunks):
            for height in range(n_modes):
                ret += "\n" + chunk_start[height]
                if n_chunks > 1 and chunk_idx != 0:
                    ret += "--- "
                ret += drawing_dict[height][chunk_idx]
                if n_chunks > 1 and chunk_idx != n_chunks - 1:
                    ret += " ---"
            ret += "\n\n"

        return ret

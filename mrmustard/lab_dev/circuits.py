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
from typing import Iterable, Sequence, Union

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from mrmustard import math, settings
from .circuit_components import CircuitComponent
from .states import State

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

    New components (or entire circuits) can be appended by using the ``>>`` operator.

    .. code-block::

        >>> from mrmustard.lab_dev import BSgate, Sgate, Vacuum, Circuit

        >>> vac = Vacuum([0, 1, 2])
        >>> s01 = Sgate([0, 1], r=[0.1, 0.2])
        >>> bs01 = BSgate([0, 1])
        >>> bs12 = BSgate([1, 2])

        >>> circ1 = Circuit([vac]) >> s01
        >>> circ2 = Circuit([bs01, bs12])
        >>> assert circ1 >> circ2 == Circuit([vac, s01, bs01, bs12])

    Args:
        components: A list of circuit components.
    """

    def __init__(self, components=Sequence[CircuitComponent]) -> None:
        self._components = components

    @property
    def components(self) -> Sequence[CircuitComponent]:
        r"""
        The components in this circuit.
        """
        return self._components

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

    def __repr__(self) -> str:
        r"""
        A string-based graphic representation of this circuit.
        """

        def component_to_str(comp: CircuitComponent) -> str:
            r"""
            Get list of labels for the component's parameters.

            Args:
                comp: A circuit component.
            """
            if comp.parameter_set.names:
                values = []
                for name in comp.parameter_set.names:
                    param = comp.parameter_set.constants.get(
                        name
                    ) or comp.parameter_set.variables.get(name)
                    new_values = math.atleast_1d(param.value)
                    if len(new_values) == 1 and comp.name not in control_gates:
                        new_values = math.tile(new_values, len(comp.modes))
                    values.append(list(new_values))
                return [comp.name + str(l).replace(" ", "") for l in list(zip(*values))]
            return [comp.name for _ in range(len(comp.modes))]

        components = self.components
        modes = set(sorted([m for c in components for m in c.modes]))
        n_modes = len(modes)

        # update this when new controlled gates are added
        control_gates = ["BSgate", "MZgate", "CZgate", "CXgate"]

        # create a dictionary mapping modes to heigth in the drawing, where heigth ``0``
        # corresponds to the mode indexed by the smallest index (e.g., mode ``0``) and
        # is drawn on the top line, heigth ``1`` to the second mode from the top, etc.
        modes_to_heigth = {m: h for m, h in zip(modes, range(n_modes))}

        # initialize the start of the
        start = [f"{mode}: " for mode in modes]
        start = [s.rjust(max(len(s) for s in start), " ") for s in start]

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
        repr = {height: "" for height in range(n_modes)}

        # loop through the layers and add to the dictionary
        for layer in layers.values():
            # layers always start with "──"
            for h in range(n_modes):
                repr[h] += "──"

            # there are two types of components: the controlled gates, and all the other ones
            for comp in layer:
                if comp.name in control_gates:
                    control = min(modes_to_heigth[m] for m in comp.modes)
                    target = max(modes_to_heigth[m] for m in comp.modes)
                    repr[control] += "╭"
                    repr[target] += "╰"
                    for h in range(target + 1, control):
                        repr[h] += "├" if h in comp.modes else "|"

                    repr[modes_to_heigth[control]] += "•"
                    repr[modes_to_heigth[target]] += component_to_str(comp)[0]
                else:
                    labels = component_to_str(comp)
                    for i, m in enumerate(comp.modes):
                        repr[modes_to_heigth[m]] += labels[i]

            # ensure that all the strings in the final drawing have the same lenght
            max_len = max(len(v) for v in repr.values())
            for h in range(n_modes):
                repr[h] = repr[h].ljust(max_len, "─")

        # break the drawing in chunks of length <90 characters that can be
        # drawn on top of each other
        from textwrap import wrap
        for h in range(n_modes):
            repr[h] = wrap(repr[h], 90)
        n_chunks = len(repr[0])

        # every chunk starts with a recap of the modes
        chunk_start = [f"mode {mode}:   " for mode in modes]
        chunk_start = [s.rjust(max(len(s) for s in chunk_start), " ") for s in chunk_start]

        ret = ""
        for chunk_idx in range(n_chunks):
            for height in range(n_modes):
                ret += "\n" + chunk_start[height]
                if n_chunks > 1 and chunk_idx != 0:
                    ret += "--- "
                ret += repr[height][chunk_idx] 
                if n_chunks > 1 and chunk_idx != n_chunks - 1:
                    ret += " ---"
            ret += "\n\n"

        return ret


        "\n".join(list(repr[modes_to_heigth[m]] for m in modes))


        lhs = [f"{mode}: " for mode in modes]
        lhs = [s.rjust(max(len(s) for s in lhs), " ") for s in lhs]

        return "\n".join(list(repr[modes_to_heigth[m]] for m in modes))


        #     for c in layer:
        #         # add symbols indicating the extent of a given object
        #         min_heigth = min(modes_to_heigth[m] for m in c.modes)
        #         max_heigth = max(modes_to_heigth[m] for m in c.modes)
        #         if max_heigth - min_heigth > 0:
        #             repr[min_heigth] += "╭" if c.name in control_gates else ""
        #             repr[max_heigth] += "╰" if c.name in control_gates else ""
        #             for h in range(min_heigth + 1, max_heigth):
        #                 repr[h] += ("├" if h in c.modes else "|") if c.name in control_gates else ""

        #         # add control for controlled gates
        #         control_m = []
        #         if c.name in control_gates:
        #             control_m = [c.modes[0]]

        #         # get list of labels for the component's parameters
        #         if c.parameter_set.names:
        #             values = []
        #             for name in c.parameter_set.names:
        #                 param = c.parameter_set.constants.get(
        #                     name
        #                 ) or c.parameter_set.variables.get(name)
        #                 new_values = math.atleast_1d(param.value)
        #                 if len(new_values) == 1 and c.name not in control_gates:
        #                     new_values = math.tile(new_values, len(c.modes))
        #                 values.append(list(new_values))
        #                 labels = [c.name + str(l) for l in list(zip(*values))]
        #         else:
        #             labels = [c.name for _ in range(len(c.modes))]

        #         # add str
        #         if c.name in control_gates:
        #             for m in c.modes:
        #                 repr[modes_to_heigth[m]] += "•" if m in control_m else labels[0]
        #         else:
        #             for i, m in enumerate(c.modes):
        #                 repr[modes_to_heigth[m]] += labels[i]

        #     max_len = max(len(v) for v in repr.values())
        #     for h in range(n_modes):
        #         repr[h] = repr[h].ljust(max_len, "─")

        # return "\n".join(list(repr[modes_to_heigth[m]] for m in modes))

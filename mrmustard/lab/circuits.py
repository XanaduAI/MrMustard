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
A class to simulate quantum circuits.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pydoc import locate

from mrmustard import math, settings
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.path import optimal_path
from mrmustard.utils.serialize import save

__all__ = ["Circuit"]


class Circuit:
    r"""
    A quantum circuit. It is a sequence of uncontracted components, which leaves the
    possibility of contracting them in different orders. The order in which the components
    are contracted is specified by the ``path`` attribute.

    Different orders of contraction lead to the same result, but the cost of the contraction
    can vary significantly. The ``optimize`` method optimizes the Fock shapes and the contraction
    path of the circuit, while the ``contract`` method contracts the components in the order
    specified by the ``path`` attribute.

    .. code-block::

        >>> from mrmustard.lab import BSgate, S2gate, Vacuum, Circuit

        >>> vac = Vacuum((0,1,2))
        >>> s01 = S2gate((0, 1), r=0.1) >> S2gate((0, 1), r=0.2)
        >>> bs01 = BSgate((0, 1))
        >>> bs12 = BSgate((1, 2))

        >>> components = [vac, s01, bs01, bs12]
        >>> circ = Circuit(components)
        >>> assert circ.components == components

    New components (or entire circuits) can be appended using the ``>>`` operator.

    .. code-block::

        >>> from mrmustard.lab import BSgate, S2gate, Vacuum, Circuit

        >>> vac = Vacuum((0,1,2))
        >>> s01 = S2gate((0, 1), r=0.1) >> S2gate((0, 1), r=0.2)
        >>> bs01 = BSgate((0, 1))
        >>> bs12 = BSgate((1, 2))

        >>> circ1 = Circuit() >> vac >> s01
        >>> circ2 = Circuit([bs01]) >> bs12
        >>> assert circ1 >> circ2 == Circuit([vac, s01, bs01, bs12])

    Args:
        components: A list of circuit components.
    """

    def __init__(
        self,
        components: Sequence[CircuitComponent] | None = None,
    ) -> None:
        self.components = [c._light_copy() for c in components] if components else []
        self.path: list[tuple[int, int]] = [
            (0, i) for i in range(1, len(self.components))
        ]  # default path (likely not optimal)

    @classmethod
    def deserialize(cls, data: dict) -> Circuit:
        r"""Deserialize a Circuit."""
        comps, path = data.pop("components"), data.pop("path")

        for k, v in data.items():
            kwarg, i = k.split(":")
            comps[int(i)][kwarg] = v

        classes: list[CircuitComponent] = [locate(c.pop("class")) for c in comps]
        circ = cls([c._deserialize(comp_data) for c, comp_data in zip(classes, comps)])
        circ.path = [tuple(p) for p in path]
        return circ

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        (ret.components,) = children
        (ret.path,) = aux_data
        return ret

    def contract(self) -> CircuitComponent:
        r"""
        Contracts the components in this circuit in the order specified by the ``path`` attribute.

        Returns:
            The result of contracting the circuit.

        Raises:
            ValueError: If ``circuit`` has an incomplete path.
        """
        if len(self.path) != len(self) - 1:
            msg = f"``circuit.path`` needs to specify {len(self) - 1} contractions, found "
            msg += (
                f"{len(self.path)}. Please run the ``.optimize()`` method or set the path manually."
            )
            raise ValueError(msg)

        ret = dict(enumerate(self.components))
        for idx0, idx1 in self.path:
            ret[idx0] = ret[idx0] >> ret.pop(idx1)

        return next(iter(ret.values()))

    def check_contraction(self, n: int) -> None:
        r"""
        An auxiliary function that helps visualize the contraction path of the circuit.

        Shows the remaining components and the corresponding contraction indices after n
        of the contractions in ``self.path``.

        .. code-block::

                >>> from mrmustard.lab import BSgate, Sgate, Vacuum, Circuit

                >>> vac = Vacuum((0,1,2))
                >>> s0 = Sgate(0, r=0.1)
                >>> bs01 = BSgate((0, 1))
                >>> bs12 = BSgate((1, 2))

                >>> circ = Circuit([vac, s0, bs01, bs12])

                >>> # ``circ`` has no path: all the components are available, and indexed
                >>> # as they appear in the list of components
                >>> circ.check_contraction(0)  # no contractions
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗
                mode 1:     ◖Vac◗
                mode 2:     ◖Vac◗
                <BLANKLINE>
                <BLANKLINE>
                → index: 1
                mode 0:   ──S(0.1,0.0)
                <BLANKLINE>
                <BLANKLINE>
                → index: 2
                mode 0:   ──╭•──────────
                mode 1:   ──╰BS(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                → index: 3
                mode 1:   ──╭•──────────
                mode 2:   ──╰BS(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>

                >>> # start building the path manually
                >>> circ.path = ((0, 1), (2, 3), (0, 2))

                >>> circ.check_contraction(1)  # after 1 contraction
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗──S(0.1,0.0)
                mode 1:     ◖Vac◗────────────
                mode 2:     ◖Vac◗────────────
                <BLANKLINE>
                <BLANKLINE>
                → index: 2
                mode 0:   ──╭•──────────
                mode 1:   ──╰BS(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                → index: 3
                mode 1:   ──╭•──────────
                mode 2:   ──╰BS(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>

                >>> circ.check_contraction(2)  # after 2 contractions
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗──S(0.1,0.0)
                mode 1:     ◖Vac◗────────────
                mode 2:     ◖Vac◗────────────
                <BLANKLINE>
                <BLANKLINE>
                → index: 2
                mode 0:   ──╭•────────────────────────
                mode 1:   ──╰BS(0.0,0.0)──╭•──────────
                mode 2:                 ──╰BS(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>

                >>> circ.check_contraction(3)  # after 3 contractions
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗──S(0.1,0.0)──╭•────────────────────────
                mode 1:     ◖Vac◗──────────────╰BS(0.0,0.0)──╭•──────────
                mode 2:     ◖Vac◗────────────────────────────╰BS(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>


        Raises:
            ValueError: If ``circuit.path`` contains invalid contractions.
        """
        remaining = {i: Circuit([c]) for i, c in enumerate(self.components)}
        for idx0, idx1 in self.path[:n]:
            try:
                left = remaining[idx0].components
                right = remaining.pop(idx1).components
                remaining[idx0] = Circuit(left + right)
            except KeyError as e:
                wrong_key = idx0 if idx0 not in remaining else idx1
                msg = f"index {wrong_key} in pair ({idx0}, {idx1}) is invalid."
                raise ValueError(msg) from e

        msg = "\n"
        for idx, circ in remaining.items():
            msg += f"→ index: {idx}"
            msg += f"{circ}\n"

        print(msg)

    def optimize(
        self,
        n_init: int = 100,
        with_BF_heuristic: bool = True,
        verbose: bool = True,
    ) -> None:
        r"""
        Optimizes the Fock shapes and the contraction path of this circuit.
        It allows one to exclude the 1BF and 1FB heuristic in case contracting 1-wire Fock/Bagmann
        components with multimode Bargmann/Fock components leads to a higher total cost.

        Args:
            n_init: The number of random contractions to find an initial cost upper bound.
            with_BF_heuristic: If True (default), the 1BF/1FB heuristics are included in the optimization process.
            verbose: If True (default), the progress of the optimization is shown.
        """
        self.path = optimal_path(
            self.components,
            n_init=n_init,
            with_BF_heuristic=with_BF_heuristic,
            verbose=verbose,
        )

    def serialize(self, filestem: str | None = None):
        r"""
        Serialize a Circuit.

        Args:
            filestem: An optional name to give the resulting file saved to disk.
        """
        components, data = list(zip(*[c._serialize() for c in self.components]))
        kwargs = {
            "arrays": {f"{k}:{i}": v for i, arrs in enumerate(data) for k, v in arrs.items()},
            "path": self.path,
            "components": components,
        }
        return save(type(self), filename=filestem, **kwargs)

    def _tree_flatten(self):  # pragma: no cover
        children = (self.components,)
        aux_data = (self.path,)
        return (children, aux_data)

    def __eq__(self, other: Circuit) -> bool:
        if not isinstance(other, Circuit):
            return False
        return self.components == other.components

    def __getitem__(self, idx: int) -> CircuitComponent:
        r"""
        The component in position ``idx`` of this circuit's components.
        """
        return self.components[idx]

    def __iter__(self):
        r"""
        An iterator over the components in this circuit.
        """
        return iter(self.components)

    def __len__(self):
        r"""
        The number of components in this circuit.
        """
        return len(self.components)

    def __rshift__(self, other: CircuitComponent | Circuit) -> Circuit:
        r"""
        Returns a ``Circuit`` that contains all the components of ``self`` as well as
        ``other`` if ``other`` is a ``CircuitComponent``, or ``other.components`` if
        ``other`` is a ``Circuit``).
        """
        if isinstance(other, CircuitComponent):
            other = Circuit([other])
        return Circuit(self.components + other.components)

    def __repr__(self) -> str:  # noqa: C901
        r"""
        A string-based representation of this component.
        """

        def component_to_str(comp: CircuitComponent) -> list[str]:
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
            cc_name = comp.short_name
            parallel = isinstance(cc_name, list)
            if not comp.wires.input:
                cc_names = [
                    f"◖{cc_name[i] if parallel else cc_name}◗" for i in range(len(comp.modes))
                ]
            elif not comp.wires.output:
                cc_names = [
                    f"|{cc_name[i] if parallel else cc_name})=" for i in range(len(comp.modes))
                ]
            elif cc_name not in control_gates:
                cc_names = [
                    f"{cc_name[i] if parallel else cc_name}" for i in range(len(comp.modes))
                ]
            else:
                cc_names = [f"{cc_name}"]

            if comp.parameters.names and settings.DRAW_CIRCUIT_PARAMS:
                values = []
                for name in comp.parameters.names:
                    param = comp.parameters.constants.get(name) or comp.parameters.variables.get(
                        name,
                    )
                    new_values = math.atleast_nd(param.value, 1)
                    if len(new_values) == 1 and cc_name not in control_gates:
                        new_values = math.tile(new_values, (len(comp.modes),))
                    values.append(math.asnumpy(new_values))
                return [
                    cc_names[i] + str(val).replace(" ", "") for i, val in enumerate(zip(*values))
                ]
            return cc_names

        if len(self) == 0:
            return ""

        modes = set(m for c in self.components for m in c.modes)  # noqa: C401
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
        wires = dict.fromkeys(range(n_modes), "  ")

        # generate a dictionary to map x-axis coordinates to the components drawn at
        # those coordinates
        layers = defaultdict(list)
        x = 0
        for c1 in self.components:
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
        drawing_dict = dict.fromkeys(range(n_modes), "")

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

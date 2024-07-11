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

# pylint: disable=too-many-branches

"""
A class to quantum circuits.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence, Union
from mrmustard import math, settings
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.transformations import BSgate

__all__ = ["Circuit"]


class Circuit:
    r"""
    A quantum circuit. It is a sequence of uncontracted components, which leaves the
    possibility of contracting them in different orders. The order in which the components
    are contracted is specified by the ``path`` attribute.

    Different orders of contraction lead to the same result, but the cost of the contraction
    can vary significantly. The ``path`` attribute is used to specify the order in which the
    components are contracted.

    We supply an automatic path generator that uses a graph representation of the circuit
    and a branch-and-bound algorithm to find the optimal path. The path can also be set manually.

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
        self._components = [c._light_copy() for c in components] if components else []
        self._path = []

        # a dictionary to keep track of the underlying graph, mapping the ``ids`` of output wires
        # to the ``ids`` of the input wires that they are being contracted with. It is initialized
        # automatically (once and for all) when a path is validated for the first time.
        self._graph: dict[int, int] = {}

    @property
    def indices_dict(self) -> dict[int, dict[int, dict[int, int]]]:
        r"""
        A dictionary that maps the index of each component to all the components it is connected to.
        For each connected component, the value is a dictionary with a key-value pair for each component
        connected to the first one, where the key is the index of this component and the value is
        a dictionary with all the wire index pairs that are being contracted between the two components.

        For example, if components[i] is connected to components[j] and they are contracting two wires
        at index pairs (a, b) and (c, d), then indices_dict[i][j] = {a: b, c:d}.

        This dictionary is used to propagate the shapes of the components in the circuit.
        """
        if not hasattr(self, "_idxdict"):
            self._idxdict = self._indices_dict()
        return self._idxdict

    def _indices_dict(self):
        ret = dict()
        for i, opA in enumerate(self.components):
            out_idx = set(opA.wires.output.indices)
            indices: dict[int, dict[int, int]] = dict()
            for j, opB in enumerate(self.components[i + 1 :]):
                ovlp_bra = opA.wires.output.bra.modes & opB.wires.input.bra.modes
                ovlp_ket = opA.wires.output.ket.modes & opB.wires.input.ket.modes
                if not (ovlp_bra or ovlp_ket):
                    continue
                iA = (
                    opA.wires.output.bra[ovlp_bra].indices
                    + opA.wires.output.ket[ovlp_ket].indices
                )
                iB = (
                    opB.wires.input.bra[ovlp_bra].indices
                    + opB.wires.input.ket[ovlp_ket].indices
                )
                if not out_idx.intersection(iA):  # output remaining
                    continue
                assert list(iA) == sorted(iA)
                assert list(iB) == sorted(iB)
                assert len(iA) == len(iB)
                indices[i + j + 1] = dict(zip(iA, iB))
                out_idx -= set(iA)
                if not out_idx:
                    break
            ret[i] = indices
        return ret

    def propagate_shapes(self):
        r"""Propagates the shape information so that the shapes of the components are better
        than those provided by the auto_shape attribute.

        .. code-block::

        >>> from mrmustard.lab_dev import BSgate, Dgate, Coherent, Circuit, SqueezedVacuum

        >>> circ = Circuit([Coherent([0], x=1.0), Dgate([0], 0.1)])
        >>> assert [op.auto_shape() for op in circ] == [(5,), (100,100)]
        >>> circ.propagate_shapes()
        >>> assert [op.auto_shape() for op in circ] == [(5,), (100, 5)]

        >>> circ = Circuit([SqueezedVacuum([0,1], r=[0.5,-0.5]), BSgate([0,1], 0.9)])
        >>> assert [op.auto_shape() for op in circ] == [(6, 6), (100, 100, 100, 100)]
        >>> circ.propagate_shapes()
        >>> assert [op.auto_shape() for op in circ] == [(6, 6), (12, 12, 6, 6)]
        """

        for component in self:
            component.fock_shape = list(component.auto_shape())

        # update the fock_shapes until convergence
        changes = self._update_shapes()
        while changes:
            changes = self._update_shapes()

    def _update_shapes(self) -> bool:
        r"""Updates the shapes of the components in the circuit graph by propagating the known shapes.
        If two wires are connected and one of them has shape n and the other None, the shape of the
        wire with None is updated to n. If both wires have a shape, the minimum is taken.

        For a BSgate, we apply the rule that the sum of the shapes of the inputs must be equal to the sum of
        the shapes of the outputs.

        It returns True if any shape was updated, False otherwise.
        """
        changes = False
        # inherit shapes from neighbors if needed
        for i, component in enumerate(self.components):
            for j, indices in self.indices_dict[i].items():
                for a, b in indices.items():  # for every connected pair between i and j
                    s_ia = self.components[i].fock_shape[a]
                    s_jb = self.components[j].fock_shape[b]
                    if not s_ia and not s_jb:
                        continue
                    changes = changes or s_ia != s_jb
                    s = min(s_ia or 1e42, s_jb or 1e42)
                    self.components[i].fock_shape[a] = s
                    self.components[j].fock_shape[b] = s

        # propagate through BSgates (remember shape = photon number + 1)
        for i, component in enumerate(self.components):
            if isinstance(component, BSgate):
                a, b, c, d = component.fock_shape  # out0, out1, in0, in1
                if c and d:
                    if not a or a > c + d:
                        a = c + d
                        changes = True
                    if not b or b > c + d:
                        b = c + d
                        changes = True
                if a and b:
                    if not c or c > a + b:
                        c = a + b
                        changes = True
                    if not d or d > a + b:
                        d = a + b
                        changes = True

                self.components[i].fock_shape = [a, b, c, d]

        return changes

    @property
    def components(self) -> Sequence[CircuitComponent]:
        r"""
        The components in this circuit.
        """
        return self._components

    @property
    def path(self) -> list[tuple[int, int]]:
        r"""
        A list describing the desired contraction path followed by the ``Simulator``.
        """
        return self._path

    @path.setter
    def path(self, value: list[tuple[int, int]]) -> None:
        r"""
        A function to set the path.

        In addition to setting the path, it validates it using the ``validate_path`` method.

        Args:
            path: The path.
        """
        self.validate_path(value)
        self._path = value

    def lookup_path(self) -> None:
        r"""
        An auxiliary function that helps building the contraction path for this circuit.

        Shows the remaining components and the corresponding contraction indices.

        .. code-block::

                >>> from mrmustard.lab_dev import BSgate, Sgate, Vacuum, Circuit

                >>> vac = Vacuum([0, 1, 2])
                >>> s01 = Sgate([0, 1], r=[0.1, 0.2])
                >>> bs01 = BSgate([0, 1])
                >>> bs12 = BSgate([1, 2])

                >>> circ = Circuit([vac, s01, bs01, bs12])

                >>> # ``circ`` has no path: all the components are available, and indexed
                >>> # as they appear in the list of components
                >>> circ.lookup_path()
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗
                mode 1:     ◖Vac◗
                mode 2:     ◖Vac◗
                <BLANKLINE>
                <BLANKLINE>
                → index: 1
                mode 0:   ──Sgate(0.1,0.0)
                mode 1:   ──Sgate(0.2,0.0)
                <BLANKLINE>
                <BLANKLINE>
                → index: 2
                mode 0:   ──╭•──────────────
                mode 1:   ──╰BSgate(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                → index: 3
                mode 1:   ──╭•──────────────
                mode 2:   ──╰BSgate(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>

                >>> # start building the path
                >>> circ.path = [(0, 1)]
                >>> circ.lookup_path()
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗──Sgate(0.1,0.0)
                mode 1:     ◖Vac◗──Sgate(0.2,0.0)
                mode 2:     ◖Vac◗────────────────
                <BLANKLINE>
                <BLANKLINE>
                → index: 2
                mode 0:   ──╭•──────────────
                mode 1:   ──╰BSgate(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                → index: 3
                mode 1:   ──╭•──────────────
                mode 2:   ──╰BSgate(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>

                >>> circ.path = [(0, 1), (2, 3)]
                >>> circ.lookup_path()
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗──Sgate(0.1,0.0)
                mode 1:     ◖Vac◗──Sgate(0.2,0.0)
                mode 2:     ◖Vac◗────────────────
                <BLANKLINE>
                <BLANKLINE>
                → index: 2
                mode 0:   ──╭•────────────────────────────────
                mode 1:   ──╰BSgate(0.0,0.0)──╭•──────────────
                mode 2:                     ──╰BSgate(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>

                >>> circ.path = [(0, 1), (2, 3), (0, 2)]
                >>> circ.lookup_path()
                <BLANKLINE>
                → index: 0
                mode 0:     ◖Vac◗──Sgate(0.1,0.0)──╭•────────────────────────────────
                mode 1:     ◖Vac◗──Sgate(0.2,0.0)──╰BSgate(0.0,0.0)──╭•──────────────
                mode 2:     ◖Vac◗────────────────────────────────────╰BSgate(0.0,0.0)
                <BLANKLINE>
                <BLANKLINE>
                <BLANKLINE>


        Raises:
            ValueError: If ``circuit.path`` contains invalid contractions.
        """
        remaining = {i: Circuit([c]) for i, c in enumerate(self.components)}
        for idx0, idx1 in self.path:
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

    def make_path(self, strategy: str = "l2r") -> None:
        r"""
        Automatically generates a path for this circuit.

        The available strategies are:
            * ``l2r``: The two left-most components are contracted together, then the
                resulting component is contracted with the third one from the left, et cetera.
            * ``r2l``: The two right-most components are contracted together, then the
                resulting component is contracted with the third one from the right, et cetera.

        Args:
            strategy: The strategy used to generate the path.
        """
        if strategy == "l2r":
            self.path = [(0, i) for i in range(1, len(self))]
        elif strategy == "r2l":
            self.path = [(i, i + 1) for i in range(len(self) - 2, -1, -1)]
        else:
            msg = f"Strategy ``{strategy}`` is not available."
            raise ValueError(msg)

    def validate_path(self, path) -> None:
        r"""
        A convenience function to check whether a given contraction path is valid for this circuit.

        Uses the wires' ``ids`` to understand what pairs of wires would be contracted, if the
        simulation was carried from left to right. Next, it checks whether ``path`` is an equivalent
        contraction path, meaning that it instructs to contract the same wires as a ``l2r`` path.

        Args:
            path: A candidate contraction path.

        Raises:
            ValueError: If the given path is not equivalent to a left-to-right path.
        """
        wires = [c.wires for c in self.components]

        # if at least one of the ``Wires`` has wires on the bra side, add the adjoint
        # to all the other ``Wires``
        add_adjoints = len(set(bool(w.bra) for w in wires)) != 1
        if add_adjoints:
            wires = [(w @ w.adjoint)[0] if bool(w.bra) is False else w for w in wires]

        # if the circuit has no graph, compute it
        if not self._graph:
            # a dictionary to store the ``ids`` of the dangling wires
            ids_dangling_wires = {
                m: {"ket": None, "bra": None} for w in wires for m in w.modes
            }

            # populate the graph
            for w in wires:
                # if there is a dangling wire, add a contraction
                for m in w.input.ket.modes:  # ket side
                    if ids_dangling_wires[m]["ket"]:
                        self._graph[ids_dangling_wires[m]["ket"]] = w.input.ket[m].ids[
                            0
                        ]
                        ids_dangling_wires[m]["ket"] = None
                for m in w.input.bra.modes:  # bra side
                    if ids_dangling_wires[m]["bra"]:
                        self._graph[ids_dangling_wires[m]["bra"]] = w.input.bra[m].ids[
                            0
                        ]
                        ids_dangling_wires[m]["bra"] = None

                # update the dangling wires
                for m in w.output.ket.modes:  # ket side
                    if w.output.ket[m].ids:
                        if ids_dangling_wires[m]["ket"]:
                            raise ValueError("Dangling wires cannot be overwritten.")
                        ids_dangling_wires[m]["ket"] = w.output.ket[m].ids[0]
                for m in w.output.bra.modes:  # bra side
                    if w.output.bra[m].ids:
                        if ids_dangling_wires[m]["bra"]:
                            raise ValueError("Dangling wires cannot be overwritten.")
                        ids_dangling_wires[m]["bra"] = w.output.bra[m].ids[0]

        # use ``self._graph`` to validate the path
        remaining = dict(enumerate(wires))
        for i1, i2 in path:
            overlap_ket = remaining[i1].output.ket.modes & remaining[i2].input.ket.modes
            for m in overlap_ket:
                key = remaining[i1].output.ket[m].ids[0]
                val = remaining[i2].input.ket[m].ids[0]
                if self._graph[key] != val:
                    raise ValueError(f"The contraction ``{(i1, i2)}`` is invalid.")

            overlap_bra = remaining[i1].output.bra.modes & remaining[i2].input.bra.modes
            for m in overlap_bra:
                key = remaining[i1].output.bra[m].ids[0]
                val = remaining[i2].input.bra[m].ids[0]
                if self._graph[key] != val:
                    raise ValueError(f"The contraction ``{i1, i2}`` is invalid.")

            remaining[i1] = (remaining[i1] @ remaining.pop(i2))[0]

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

    def __iter__(self):
        r"""
        An iterator over the components in this circuit.
        """
        return iter(self.components)

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

            if comp.parameter_set.names and settings.CIRCUIT_DRAW_PARAMS:
                values = []
                for name in comp.parameter_set.names:
                    param = comp.parameter_set.constants.get(
                        name
                    ) or comp.parameter_set.variables.get(name)
                    new_values = math.atleast_1d(param.value)
                    if len(new_values) == 1 and cc_name not in control_gates:
                        new_values = math.tile(new_values, (len(comp.modes),))
                    values.append(
                        new_values.numpy()
                        if math.backend.name == "tensorflow"
                        else new_values
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
        chunk_start = [
            s.rjust(max(len(s) for s in chunk_start), " ") for s in chunk_start
        ]

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

# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
import numbers
from collections import defaultdict
from contextlib import suppress
from copy import copy
from math import isfinite
from typing import Literal

from mrmustard import math
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.lab.computational_graph import ComputationalGraph
from mrmustard.lab.states import DM, Ket, State
from mrmustard.physics.wires import QuantumWire, ReprEnum
from mrmustard.utils.typing import Scalar

__all__ = ["Circuit"]

# update this when new controlled gates are added
control_gates = ["BSgate", "CXgate", "CZgate", "MZgate"]


def _component_to_str(comp: CircuitComponent, draw_params: bool = True) -> list[str]:
    r"""Generates a list string-based representation for the given component.

    If ``comp`` is not a controlled gate, the list contains as many elements modes as in
    ``comp.modes``. For example, if ``CircuitComponent(modes=(0,1,2))``, it returns
    ``['CircuitComponent(modes=(0,))', 'CircuitComponent(modes=(1,))', 'CircuitComponent(modes=(2,))']``.

    If ``comp`` is a controlled gate, the list contains the string that needs to be added to
    the target mode. For example, if ``comp=BSgate((0, 1), 1, 2)``, it returns
    ``['BSgate(0.0,0.0)']``.

    Args:
        comp: A circuit component.
        draw_params: Whether to draw the parameters of the component.

    Returns:
        A list of strings representing the component.
    """
    cc_name = comp.short_name
    class_name = comp.__class__.__name__
    parallel = isinstance(cc_name, list)
    if not comp.wires.input:
        cc_names = [f"◖{cc_name[i] if parallel else cc_name}◗" for i in range(len(comp.modes))]
    elif not comp.wires.output:
        cc_names = [f"|{cc_name[i] if parallel else cc_name})=" for i in range(len(comp.modes))]
    elif class_name not in control_gates:
        cc_names = [f"{cc_name[i] if parallel else cc_name}" for i in range(len(comp.modes))]
    else:
        cc_names = [f"{cc_name}"]

    if comp.parameters.names and draw_params:
        values = []
        for name in comp.parameters.names:
            param = comp.parameters.constants.get(name) or comp.parameters.variables.get(
                name,
            )
            new_values = math.atleast_nd(param.value, 1)
            if len(new_values) == 1 and class_name not in control_gates:
                new_values = math.tile(new_values, (len(comp.modes),))
            values.append(math.asnumpy(new_values))
        return [cc_names[i] + _format_param_tuple(val) for i, val in enumerate(zip(*values))]
    return cc_names


def _draw_components(components: list[CircuitComponent], draw_params: bool = True) -> str:  # noqa: C901
    r"""A string-based representation of a list of components.

    Args:
        components: The list of components to draw.
        draw_params: Whether to draw the parameters of the components.

    Returns:
        A string-based representation of the list of components.
    """
    if len(components) == 0:
        return ""

    modes = {m for c in components for m in c.modes}
    n_modes = len(modes)

    # create a dictionary ``lines`` mapping modes to the height of the corresponding line
    #  in the drawing, where:
    # - height ``0`` is the mode with smallest index and drawn on the top line
    # - height ``1`` is the second mode from the top
    # - etc.
    lines = {m: h for h, m in enumerate(sorted(modes))}

    # create a dictionary ``wires`` that maps height ``h`` to "──" if the line contains
    # a mode, or to "  " if the line does not contain a mode
    wires = dict.fromkeys(range(n_modes), "  ")

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
    drawing_dict = dict.fromkeys(range(n_modes), "")

    # loop through the layers and add the components to ``drawing_dict``
    for layer in layers.values():
        for comp in layer:
            # there are two types of components: the controlled gates, and all the other ones
            if comp.__class__.__name__ in control_gates:
                control = min(lines[m] for m in comp.modes)
                target = max(lines[m] for m in comp.modes)

                # update ``wires`` and start the line with "──"
                wires[control] = "──"
                wires[target] = "──"
                drawing_dict[control] += "──"
                drawing_dict[target] += "──"

                drawing_dict[control] += "╭"
                drawing_dict[target] += "╰"

                drawing_dict[control] += "•"
                drawing_dict[target] += _component_to_str(comp, draw_params)[0]
            else:
                labels = _component_to_str(comp, draw_params)
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

        # ensure that all the strings in the final drawing have the same length
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


def _format_draw_scalar(x) -> str:
    """Format one parameter for drawing; trim floats to two decimals only when str() shows more."""
    if hasattr(x, "item"):
        with suppress(ValueError, AttributeError):
            x = x.item()
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, numbers.Integral):
        return str(int(x))
    if isinstance(x, numbers.Real):
        xf = float(x)
        s = str(xf).replace(" ", "")
        if not isfinite(xf):
            return s
        if "." in s and "e" not in s.lower():
            frac = s.split(".", 1)[1].rstrip("0")
            if len(frac) > 2:
                return f"{xf:.2f}"
        return s
    return str(x).replace(" ", "")


def _format_param_tuple(val: tuple) -> str:
    parts = [_format_draw_scalar(x) for x in val]
    if len(parts) == 1:
        return f"({parts[0]},)"
    return "(" + ",".join(parts) + ")"


def _get_wires(components: list[CircuitComponent]) -> list[tuple[str, str]]:
    r"""Builds the list of wire tuples for a ``ComputationalGraph.add_wires`` call
    given a list of components.

    Args:
        components: The list of components to parse (including adjoints).

    Returns:
        The list of wire connections to add to the computational graph.
    """
    ket_paths = {}
    bra_paths = {}

    for c in components:
        ket_modes = set()
        bra_modes = set()
        for q in c.wires.quantum_wires:
            (ket_modes if q.is_ket else bra_modes).add(q.mode)
        for m in ket_modes:
            ket_paths.setdefault(m, []).append(c)
        for m in bra_modes:
            bra_paths.setdefault(m, []).append(c)

    wires = []
    for mode, path in ket_paths.items():
        for prev, nxt in itertools.pairwise(path):
            wires.append((f"{prev.name}[ok{mode}]", f"{nxt.name}[ik{mode}]"))
    for mode, path in bra_paths.items():
        for prev, nxt in itertools.pairwise(path):
            wires.append((f"{prev.name}[ob{mode}]", f"{nxt.name}[ib{mode}]"))

    return wires


def _include_missing_adjoints(components: list[CircuitComponent]) -> list[CircuitComponent]:
    r"""Returns a copy of ``components`` with any missing adjoint components inserted based on
    the logic of :py:meth:`~mrmustard.lab.circuit_components.CircuitComponent.__rshift__`.

    Args:
        components: The list of components to parse and add adjoints to.

    Returns:
        A list of components including any missing adjoint components.
    """
    ret = []
    pending = []
    acc_bra = False
    acc_ket = False

    for comp in components:
        has_bra = bool(comp.wires.bra)
        has_ket = bool(comp.wires.ket)
        comp_complete = has_bra and has_ket
        comp_incomplete = (has_bra or has_ket) and not comp_complete
        acc_complete = acc_bra and acc_ket
        acc_incomplete = (acc_bra or acc_ket) and not acc_complete

        if acc_incomplete and comp_complete:
            ret.extend(p.adjoint for p in pending)
            pending.clear()
            ret.append(comp)
        elif acc_complete and comp_incomplete:
            ret.append(comp.adjoint)
            ret.append(comp)
        else:
            ret.append(comp)
            if comp_incomplete:
                pending.append(comp)

        acc_bra = acc_bra or has_bra
        acc_ket = acc_ket or has_ket

    return ret


class Circuit:
    r"""A quantum optical circuit.

    A circuit is defined by a collection of ``CircuitComponent``\ s.
    Each mode is expected to begin with a ``State`` followed by a
    series of ``Transformation``\ s.

    Args:
        components: A list of components in the circuit.
    """

    def __init__(self, components: list[CircuitComponent]):
        self._validate_components(components)
        self._components = components
        self._cached_comp_graph = None
        self._cached_fock_config = None

    @property
    def components(self) -> list[CircuitComponent]:
        return self._components

    def draw(self, draw_params: bool = True) -> str:
        r"""A string-based representation of this circuit.

        Args:
            draw_params: Whether to draw the parameters of the components.

        Returns:
            A string-based representation of the circuit.
        """
        return _draw_components(self.components, draw_params)

    def expectation(
        self,
        operator: CircuitComponent,
    ) -> Scalar:
        r"""Compute the expectation value of an operator with respect to the circuit.

        Args:
            operator: The operator to compute the expectation value of. Expected
                to be a unitary-like component (input and output ket wires on
                each of its modes).

        Returns:
            The expectation value of the operator with respect to the circuit.

        Raises:
            ValueError: If the operator acts on a mode that is no longer open
                in the circuit (e.g. already measured out).
        """
        comp_graph = copy(self.to_comp_graph())

        # snapshot the open output wires per mode before adding the operator
        last_ket: dict[int, str] = {}
        last_bra: dict[int, str] = {}
        for name, wires in comp_graph.uncontracted_wires.items():
            for w in wires:
                if not isinstance(w, QuantumWire) or not w.is_out:
                    continue
                if w.is_ket:
                    last_ket[w.mode] = name
                else:
                    last_bra[w.mode] = name

        comp_graph.add_component(operator, operator.name)

        # connect operator to graph and trace the operator's modes
        for mode in operator.modes:
            if mode not in last_ket or mode not in last_bra:
                raise ValueError(
                    f"Operator acts on mode {mode}, which has no open output wires in the circuit."
                )
            comp_graph.add_wire(f"{last_ket[mode]}[ok{mode}]", f"{operator.name}[ik{mode}]")
            comp_graph.add_wire(f"{operator.name}[ok{mode}]", f"{last_bra[mode]}[ob{mode}]")

        # trace out the remaining modes
        op_modes = set(operator.modes)
        for mode, ket_name in last_ket.items():
            if mode in op_modes:
                continue
            comp_graph.add_wire(f"{ket_name}[ok{mode}]", f"{last_bra[mode]}[ob{mode}]")

        return comp_graph.run()

    def run(
        self,
        output_shape: tuple[int, ...] | None = None,
        representation: Literal[ReprEnum.BARGMANN, ReprEnum.FOCK] | None = None,
    ) -> State | Scalar:
        r"""Runs the circuit.

        The return type depends on whether the underlying computational graph
        is fully specified or not.

        Args:
            output_shape: If the circuit returns a ``State``, the shape of the output in the Fock basis.
            representation: If the circuit returns a ``State``, the representation of the output.

        Returns:
            The result of the circuit.
        """
        comp_graph = self.to_comp_graph()
        _, output = comp_graph._mm_einsum_args()

        if self._cached_fock_config is None:
            self._cached_fock_config = comp_graph.fock_config()

        fock_config = self._cached_fock_config

        if output_shape is not None:
            for idx, label in enumerate(output):
                fock_config[label] = output_shape[idx]

        try:
            result = comp_graph.run(fock_config=fock_config)
        except RuntimeError:
            result = comp_graph.to_component(fock_config=fock_config)

            if representation and representation != result.wires.representations[0]:
                result = (
                    result.to_fock() if representation == ReprEnum.FOCK else result.to_bargmann()
                )

            if not result.wires.ket or not result.wires.bra:
                result = Ket.from_ansatz(modes=result.modes, ansatz=result.ansatz)
            else:
                result = DM.from_ansatz(modes=result.modes, ansatz=result.ansatz)
        return result

    def to_comp_graph(self) -> ComputationalGraph:
        r"""Compiles this circuit into a ``ComputationalGraph``.

        Returns:
            A computational graph representing this circuit.
        """
        if self._cached_comp_graph:
            return self._cached_comp_graph
        components_w_adjoint = _include_missing_adjoints(self.components)
        names = [c.name for c in components_w_adjoint]
        comp_graph = ComputationalGraph()
        comp_graph.add_components(components_w_adjoint, names)
        comp_graph.add_wires(_get_wires(components_w_adjoint))
        self._cached_comp_graph = comp_graph
        return comp_graph

    def trace_out(self, modes: int | tuple[int, ...]) -> Circuit:
        r"""Trace out the specified modes.

        Args:
            modes: The modes to trace out.

        Returns:
            A new ``Circuit`` with ``modes`` traced out.
        """
        modes = (modes,) if isinstance(modes, int) else modes
        ret = copy(self)
        comp_graph = ret.to_comp_graph()

        # snapshot the open output wires per mode before adding the operator
        last_ket: dict[int, str] = {}
        last_bra: dict[int, str] = {}
        for name, wires in comp_graph.uncontracted_wires.items():
            for w in wires:
                if not isinstance(w, QuantumWire) or not w.is_out:
                    continue
                if w.is_ket:
                    last_ket[w.mode] = name
                else:
                    last_bra[w.mode] = name

        # trace out
        for mode, ket_name in last_ket.items():
            if mode not in modes:
                continue
            comp_graph.add_wire(f"{ket_name}[ok{mode}]", f"{last_bra[mode]}[ob{mode}]")
        ret._cached_fock_config = None
        return ret

    def _validate_components(self, components: list[CircuitComponent]) -> None:
        r"""Validates the list of components.

        Args:
            components: The list of components to validate.

        Raises:
            ValueError: If the first component on a particular mode is not a ``State``,
                or if two components share the same ``name``.
        """
        seen_modes: set[int] = set()
        seen_names: set[str] = set()
        for comp in components:
            new_modes = [m for m in comp.modes if m not in seen_modes]
            if new_modes and not isinstance(comp, State):
                raise ValueError(
                    f"The first component on mode(s) {new_modes} must be a `State`, "
                    f"got `{type(comp).__name__}`."
                )
            if comp.name in seen_names:
                raise ValueError(
                    f"Duplicate component name `{comp.name}`: each component in the circuit "
                    "must have a unique `name`.",
                )
            seen_names.add(comp.name)
            seen_modes.update(comp.modes)

    def __copy__(self) -> Circuit:
        new_circuit = Circuit(self.components)
        new_circuit._cached_comp_graph = copy(self._cached_comp_graph)
        new_circuit._cached_fock_config = copy(self._cached_fock_config)
        return new_circuit

    def __repr__(self) -> str:
        return self.draw()

    def __len__(self) -> int:
        return len(self.components)

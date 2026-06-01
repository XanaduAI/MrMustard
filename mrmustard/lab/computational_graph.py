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

import re

import rustworkx as rx

from mrmustard import settings
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.lab.states import DM, Ket, State
from mrmustard.lab.transformations import BSgate, Channel, Map, Operation, Unitary
from mrmustard.physics.ansatz import Ansatz
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.mm_einsum import mm_einsum
from mrmustard.physics.wires import ClassicalWire, QuantumWire, ReprEnum, Wires
from mrmustard.utils.typing import Scalar

__all__ = ["ComputationalGraph"]

# the built-in `CircuitComponent` base classes; concrete components (e.g. `Coherent`)
# are recognized as built-ins by being strict subclasses of one of these.
BUILT_IN_COMPONENT_TYPES = (DM, Ket, Unitary, Operation, Channel, Map)


def optimize_fock_dims(  # noqa: C901
    fock_dims: dict[int, int],
    components: list[CircuitComponent],
    inputs: list[list[int]],
    iteration: int = 0,
    verbose: bool = False,
) -> dict[int, int]:
    r"""A recursive function to determine the optimal Fock dimensions for all wire labels in an ``mm_einsum`` contraction.

    Args:
        fock_dims: A dictionary mapping wire labels to their respective Fock dimensions.
        components: A list of components.
        inputs: A list of the ``mm_einsum`` input labels for each component.
        iteration: The iteration number.
        verbose: Whether to print the progress.

    Returns:
        A dictionary mapping wire labels to their respective Fock dimensions.
    """
    snapshot_before = tuple(sorted(fock_dims.items()))

    for labels, component in zip(inputs, components, strict=True):
        shape = component.auto_shape() if isinstance(component, State) else component.manual_shape
        for label, dim in zip(labels, shape, strict=True):
            current_dim = fock_dims.get(label)
            if dim is not None:
                fock_dims[label] = dim if current_dim is None else min(current_dim, dim)

    for labels, component in zip(inputs, components, strict=True):
        # enforce beamsplitter symmetry for photon number conservation
        # if we know the total number of photons going in (out) a beamsplitter then
        # the number of photons coming out (in) must match
        if isinstance(component, BSgate):
            a, b, c, d = (fock_dims.get(lb) for lb in labels)
            if c and d:
                max_dim = c + d
                a = max_dim if not a or a > max_dim else a
                b = max_dim if not b or b > max_dim else b
            if a and b:
                max_dim = a + b
                c = max_dim if not c or c > max_dim else c
                d = max_dim if not d or d > max_dim else d
            fock_dims.update(zip(labels, (a, b, c, d), strict=True))

    if tuple(sorted(fock_dims.items())) == snapshot_before:
        if verbose:
            print(f"Final outcome: {fock_dims}")
        return fock_dims

    if verbose:
        print(f"Iteration {iteration}: {fock_dims}")

    return optimize_fock_dims(fock_dims, components, inputs, iteration + 1, verbose)


class ComputationalGraph:
    r"""A computational graph is a directed graph that represents the computation of a quantum circuit.
    Nodes are comprised of components and edges are wires between nodes.
    """

    def __init__(self) -> None:
        self._ansatz_dict = {}
        self._cached_mm_einsum_args = None
        self._graph = rx.PyDiGraph()
        self._name_to_idx = {}
        self._uncontracted_wires = {}

    @property
    def ansatz_dict(self) -> dict[type, AnsatzFactory]:
        r"""The ansatz dictionary for the computational graph."""
        return self._ansatz_dict

    @property
    def graph(self) -> rx.PyDiGraph:
        r"""The underlying ``rustworkx`` graph."""
        return self._graph

    @property
    def mm_einsum_str(self) -> str:
        r"""The ``mm_einsum`` string for the computational graph.

        Raises:
            ValueError: If the computational graph contains more than 26 unique indices.
        """
        labels, output = self._mm_einsum_args()

        unique_labels = sorted({lbl for wire_labels in labels for lbl in wire_labels})
        if len(unique_labels) > 26:
            raise ValueError(
                f"Computational graph contains too many ({len(unique_labels)}) unique indices to generate an `mm_einsum` string!"
            )

        # Convert the unique integer labels into 'a'-'z'
        label_to_char = {uid: chr(97 + i) for i, uid in enumerate(unique_labels)}

        # Generate the mm_einsum string
        equation = ",".join("".join(label_to_char[lbl] for lbl in wl) for wl in labels)

        # Generate the output string in standard order
        equation += "->"
        equation += "".join(label_to_char[w] for w in output)
        return equation

    @property
    def name_to_idx(self) -> dict[str, int]:
        r"""A dictionary mapping node names to their indices in the graph."""
        return self._name_to_idx

    @property
    def uncontracted_wires(self) -> dict[str, list[QuantumWire | ClassicalWire]]:
        r"""A dictionary mapping node names to their uncontracted wires."""
        return self._uncontracted_wires

    def add_component(self, component: CircuitComponent, name: str) -> int:
        r"""Add a component to the computational graph.

        Args:
            component: The component to add.
            name: The name of the component.

        Returns:
            The index of the added component.
        """
        idx = self.graph.add_node(component)
        self._name_to_idx[name] = idx
        self._uncontracted_wires[name] = component.wires.standard_order
        self._populate_ansatz_dict(component)
        self._cached_mm_einsum_args = None
        return idx

    def add_components(self, components: list[CircuitComponent], names: list[str]) -> list[int]:
        r"""Add components to the computational graph from a list of components and names.

        Args:
            components: A list of circuit components.
            names: A list of names for the components.

        Returns:
            A list of indices of the added components.

        Raises:
            ValueError: If the number of components and names are not the same.
        """
        idxs = []
        for component, name in zip(components, names, strict=True):
            idx = self.add_component(component, name)
            idxs.append(idx)
        return idxs

    def add_wire(self, component_a: str, component_b: str) -> None:
        r"""Add a wire between two nodes in the graph.
        Nodes must be of the form ``component_name[wire_spec]``.

        ``wire_spec`` is a string consisting of three characters:
            - `i` / `o` for input / output.
            - `k` / `b` for ket / bra.
            - `0` / `1` / `2`, etc. for mode.

        For example, `ib0` means an input bra on mode 0 whereas
        `ok1` means an output ket on mode 1.

        Args:
            component_a: The name and wire of the first component.
            component_b: The name and wire of the second component.

        Returns:
            None

        Raises:
            ValueError: If a wire is not supported.
        """
        name_a, wire_a = self._parse_wire_string(component_a)
        name_b, wire_b = self._parse_wire_string(component_b)

        for name, wire_index in ((name_a, wire_a), (name_b, wire_b)):
            supported_wires = self._uncontracted_wires[name]
            if wire_index not in {w.index for w in supported_wires}:
                raise ValueError(f"Invalid wire {wire_index} for component {name}!")
            self._uncontracted_wires[name] = [w for w in supported_wires if w.index != wire_index]

        idx_a = self._name_to_idx[name_a]
        idx_b = self._name_to_idx[name_b]
        self.graph.add_edge(idx_a, idx_b, (component_a, component_b))
        self._cached_mm_einsum_args = None

    def add_wires(self, wires: list[tuple[str, str]]) -> None:
        r"""Add wires between components in the graph from a list of wire tuples.
        See :meth:`~mrmustard.lab.computational_graph.ComputationalGraph.add_wire` for more details.

        Args:
            wires: A list of wire tuples.

        Returns:
            None
        """
        for wire in wires:
            self.add_wire(wire[0], wire[1])

    def fock_config(
        self,
        representation_config: dict[str, ReprEnum] | None = None,
        verbose: bool = False,
    ) -> dict[int, int]:
        r"""Returns an optimized Fock configuration given a representation configuration
        by using a recursive optimization strategy.

        Args:
            representation_config: A representation configuration.
            verbose: Whether to print the progress.

        Returns:
            A dictionary mapping wire labels to their respective Fock dimensions.
        """
        representation_config = (
            representation_config
            if representation_config is not None
            else self.representation_config()
        )

        labels, output = self._mm_einsum_args()
        components = list(self.graph.nodes())
        fock_dims = optimize_fock_dims({}, components, labels, verbose=verbose)

        fock_config = {}
        for comp_name, representation in representation_config.items():
            if representation == ReprEnum.FOCK:
                idx = self.name_to_idx[comp_name]
                fock_config.update({lbl: fock_dims[lbl] for lbl in labels[idx]})
        if fock_config:
            for label in output:
                fock_config[label] = fock_dims.get(label, settings.DEFAULT_FOCK_SIZE)
        return fock_config

    def representation_config(self) -> dict[str, ReprEnum]:
        r"""Returns a default representation configuration for the computational graph.

        Currently, this is the representation of the component instance
        (selected via :meth:`~mrmustard.lab.circuit_components.CircuitComponent.wires.representations`)
        however optimizations are possible in the future.

        Returns:
            A dictionary mapping component names to their representations.
        """
        representation_config = {}
        for comp_name, idx in self.name_to_idx.items():
            component = self.graph[idx]
            representation_config[comp_name] = component.wires.representations[0]
        return representation_config

    def run(
        self,
        representation_config: dict[str, ReprEnum] | None = None,
        fock_config: dict[int, int] | None = None,
        path_config: list[tuple[int, int]] | None = None,
    ) -> Scalar:
        r"""Runs the computational graph.

        Args:
            representation_config: A representation configuration.
            fock_config: A Fock configuration.
            path_config: A contraction path configuration.

        Returns:
            The result of the computational graph.

        Raises:
            RuntimeError: If the graph is underspecified.
        """
        for component_name, s_wires in self.uncontracted_wires.items():
            if s_wires:
                raise RuntimeError(
                    f"Graph is underspecified. Component {component_name} has uncontracted wires."
                )
        return self.to_component(
            representation_config=representation_config,
            fock_config=fock_config,
            path_config=path_config,
        ).ansatz.scalar

    def to_component(
        self,
        representation_config: dict[str, ReprEnum] | None = None,
        fock_config: dict[int, int] | None = None,
        path_config: list[tuple[int, int]] | None = None,
    ) -> CircuitComponent:
        r"""Converts the computational graph to a ``CircuitComponent``.

        Args:
            representation_config: A representation configuration.
            fock_config: A Fock configuration.
            path_config: A contraction path configuration.

        Returns:
            A ``CircuitComponent`` representing the computational graph in its current state.
        """
        representation_config = (
            representation_config
            if representation_config is not None
            else self.representation_config()
        )
        fock_config = (
            fock_config
            if fock_config is not None
            else self.fock_config(representation_config=representation_config)
        )
        component_ansatz = self._get_component_ansatz(representation_config, fock_config)
        labels, output = self._mm_einsum_args()
        operands = [x for pair in zip(component_ansatz, labels, strict=True) for x in pair]
        operands.append(output)

        ansatz = mm_einsum(
            *operands,
            fock_dims=fock_config,
            contraction_path=path_config,
        )

        wires = Wires.from_wires(
            quantum=[w.copy(new_id=True) for ws in self.uncontracted_wires.values() for w in ws]
        )
        wires._reindex()
        return CircuitComponent._from_attributes(ansatz=ansatz, wires=wires)

    def _get_component_ansatz(
        self, representation_config: dict[str, ReprEnum], fock_config: dict[int, int]
    ) -> list[Ansatz]:
        r"""Gets the ansatz for each component in the computational graph for the given configuration.

        Args:
            representation_config: A representation configuration.
            fock_config: A Fock configuration.

        Returns:
            A list of ansatz for each component in the computational graph.
        """
        labels, _ = self._mm_einsum_args()
        components = self.graph.nodes()
        ansatz_dict = self.ansatz_dict

        idx_to_name = {idx: name for name, idx in self.name_to_idx.items()}

        component_ansatz = []
        for idx, comp in enumerate(components):
            comp_type = type(comp)
            comp_name = idx_to_name[idx]
            if comp_type in ansatz_dict:
                representation = representation_config[comp_name]
                ansatz_factory = ansatz_dict[comp_type]
                parameters = comp.parameters
                shape = (
                    tuple(fock_config[label] for label in labels[idx])
                    if representation == ReprEnum.FOCK
                    else ()
                )
                ansatz = ansatz_factory(**parameters, representation=representation, shape=shape)
            else:
                ansatz = comp.ansatz

            component_ansatz.append(ansatz)
        return component_ansatz

    def _mm_einsum_args(self) -> tuple[list[list[int]], list[int]]:
        r"""Generates the operands (excluding Ansatze) for an `mm_einsum` call in sublist style."""
        if self._cached_mm_einsum_args is None:
            components = self.graph.nodes()
            connections = self.graph.edges()

            # [sc-113556] This can be removed once wires have labels
            # Assign integer labels to all wires; connections will merge labels so that
            # connected wires share the same label, freeing them for reuse.
            next_label = 0
            labels: list[list[int]] = []
            wire_id_to_label: dict[int, int] = {}
            for component in components:
                wire_labels = []
                for w in component.wires.quantum:
                    wire_labels.append(next_label)
                    wire_id_to_label[w.id] = next_label  # use the wire id to avoid conflicts
                    next_label += 1
                labels.append(wire_labels)

            # Update labels based on connections
            for connection in connections:
                name_a, idx_a = self._parse_wire_string(connection[0])
                posn_a = self.name_to_idx[name_a]

                name_b, idx_b = self._parse_wire_string(connection[1])
                posn_b = self.name_to_idx[name_b]

                labels[posn_b][idx_b] = labels[posn_a][idx_a]

            output_wires = sorted(
                (w for wires in self.uncontracted_wires.values() for w in wires),
                key=lambda w: w._order(),
            )
            output = [wire_id_to_label[w.id] for w in output_wires]
            self._cached_mm_einsum_args = (labels, output)

        return self._cached_mm_einsum_args

    def _parse_wire_string(self, wire_string: str) -> tuple[str, int]:
        r"""Parses a wire string of the form ``node_name[wire_spec]``
        into a tuple of the form ``(node_name, wire_idx)``.

        ``wire_spec`` is a string consisting of three characters:
            - `i` / `o` for input / output.
            - `k` / `b` for ket / bra.
            - `0` / `1` / `2`, etc. for mode.

        For example, `ib0` means an input bra on mode 0 whereas
        `ok1` means an output ket on mode 1.

        Args:
            wire_string: The wire string to parse.

        Returns:
            A tuple of the form ``(node_name, wire_idx)``.

        Raises:
            ValueError: If the string format is invalid.
            ValueError: If the component or specified wire is not found.
        """
        match = re.match(r"^(.+)\[([io][bk][0-9]+)\]$", wire_string)
        if not match:
            raise ValueError(f"Invalid string format: {wire_string}!")

        name, wire_spec = match.groups()
        is_out = wire_spec[0] == "o"
        is_ket = wire_spec[1] == "k"
        mode = int(wire_spec[2:])

        node_idx = self.name_to_idx.get(name)
        if node_idx is None:
            raise ValueError(f"Component {name} not found!")
        for w in self.graph[node_idx].wires:
            if w.is_out == is_out and w.is_ket == is_ket and w.mode == mode:
                return name, w.index

        raise ValueError(f"Wire {wire_spec} not found!")

    def _populate_ansatz_dict(self, component: CircuitComponent) -> None:
        r"""Populates the ansatz dictionary for a component.

        Note:
            Generic ``CircuitComponent``s are not added to the ansatz dictionary.

        Args:
            component: The component to populate the ansatz dictionary with.

        Returns:
            None
        """
        comp_type = type(component)
        if (
            comp_type not in self._ansatz_dict
            and comp_type not in BUILT_IN_COMPONENT_TYPES
            and issubclass(comp_type, BUILT_IN_COMPONENT_TYPES)
        ):
            self._ansatz_dict[comp_type] = component.ansatz_factory

    def __copy__(self) -> ComputationalGraph:
        r"""Returns a copy of the computational graph."""
        new_graph = ComputationalGraph()
        new_graph._ansatz_dict = self._ansatz_dict
        new_graph._cached_mm_einsum_args = self._cached_mm_einsum_args
        new_graph._graph = self._graph.copy()
        new_graph._name_to_idx = self._name_to_idx.copy()
        new_graph._uncontracted_wires = self._uncontracted_wires.copy()
        return new_graph

    def __repr__(self) -> str:
        idx_to_name = {idx: name for name, idx in self.name_to_idx.items()}
        comp_names = [idx_to_name[idx] for idx in range(len(self.graph.nodes()))]

        labels, output = self._mm_einsum_args()
        label_strs = [str(lbl) for lbl in labels]

        col_widths = [max(len(n), len(l)) for n, l in zip(comp_names, label_strs, strict=True)]
        names_row = "   ".join(n.ljust(w) for n, w in zip(comp_names, col_widths, strict=True))
        labels_row = "   ".join(l.ljust(w) for l, w in zip(label_strs, col_widths, strict=True))

        repr_str = [
            "ComputationalGraph",
            f"components: {names_row}",
            f"labels:     {labels_row}  ->  {output}",
        ]
        return "\n".join(repr_str)

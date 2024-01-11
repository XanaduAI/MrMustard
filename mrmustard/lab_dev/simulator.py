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

from typing import Sequence

from ..physics.representations import Bargmann, Representation
from .circuit_components import CircuitComponent
from .circuits import Circuit
from .wires import Wire, Wires

__all__ = ["Simulator", "SimulatorBargmann"]


class Simulator:
    r"""
    A simulator for ``Circuit`` objects.
    """

    def _contract_bargmann(self, components: Sequence[CircuitComponent]) -> CircuitComponent:
        r"""
        Contracts a sequence of Bargmann components.
        """
        # 0. create the einsum dict and the char-id and id-char dicts
        einsum_dict = self._create_einsum_dict(self.components)
        char_id_dict = {
            char: self.components[i].wires.ids[j]
            for i, string in enumerate(einsum_dict)
            for j, char in enumerate(string)
        }
        id_char_dict = {id: char for char, id in char_id_dict.items()}

    def run(self, circuit: Circuit) -> CircuitComponent:
        r"""
        Simulates the given circuit.

        Arguments:
            circuit: The circuit to simulate.
        """
        ...


@staticmethod
def _create_einsum_dict(components: Sequence[CircuitComponent]) -> dict[str, Representation]:
    r"""Creates the einsum dict from the components."""
    einsum: dict[str, Representation] = {}
    ids = {id: None for c in components for id in c.wires.ids}
    ids_map = {id: i for i, id in enumerate(ids.keys())}  # remap to 0,1,2...
    strings_ints = [[ids_map[id] for id in c.wires.ids] for c in components]
    strings = ["".join([chr(i + 97) for i in s]) for s in strings_ints]
    for s, c in zip(strings, components):
        einsum[s] = c.representation
    return einsum


def _contract(self, components: Sequence[CircuitComponent]):
    r"""Contracts the circuit."""
    # 0. create the einsum dict and the char-id and id-char dicts
    einsum_dict = self._create_einsum_dict(self.components)
    char_id_dict = {
        char: self.components[i].wires.ids[j]
        for i, string in enumerate(einsum_dict)
        for j, char in enumerate(string)
    }
    id_char_dict = {id: char for char, id in char_id_dict.items()}

    # 1. create the einsum string for the whole contraction
    all_chars = [char for s in einsum_dict for char in s]
    ids = [char_id_dict[char] for char in set(all_chars) if all_chars.count(char) == 1]
    out_bra = reduce(add, [c.wires.bra.output.subset(ids) for c in components])
    in_bra = reduce(add, [c.wires.bra.input.subset(ids) for c in components])
    out_ket = reduce(add, [c.wires.ket.output.subset(ids) for c in components])
    in_ket = reduce(add, [c.wires.ket.input.subset(ids) for c in components])
    out_string = "".join(
        [id_char_dict[id] for id in out_bra.ids + in_bra.ids + out_ket.ids + in_ket.ids]
    )
    einsum_string = ",".join(einsum_dict.keys()) + "->" + out_string

    # 2. use einsum string in path_info to get pair-wise contractions
    import opt_einsum as oe

    shapes = [(2,) * len(s) for s in einsum_dict.keys()]
    path_info = oe.contract_path(einsum_string, *shapes, shapes=True, optimize="auto")

    # 3. update the dict as path_info indicates until the whole circuit is contracted
    for _, _, current, _, _ in path_info[1].contraction_list:
        idx_A, idx_B, idx_reorder = self.parse_einsum(current)
        A, (B, out) = current.split(",")[0], current.split(",")[1].split("->")
        new_r = einsum_dict[A][idx_A] @ einsum_dict[B][idx_B]
        new_r = new_r.reorder(idx_reorder)
        einsum_dict[out] = new_r

    # 4. return the result
    return CircuitComponent(
        "",
        einsum_dict[out_string],
        modes_out_bra=out_bra.modes,
        modes_in_bra=in_bra.modes,
        modes_out_ket=out_ket.modes,
        modes_in_ket=in_ket.modes,
    )


@staticmethod
def parse_einsum(string: str):
    parts, result = string.split("->")
    parts = parts.split(",")
    repeated = set(parts[0]).intersection(parts[1])
    remaining = set(parts[0]).union(parts[1]) - repeated
    return (
        [parts[0].index(i) for i in repeated],
        [parts[1].index(i) for i in repeated],
        [result.index(i) for i in remaining],
    )

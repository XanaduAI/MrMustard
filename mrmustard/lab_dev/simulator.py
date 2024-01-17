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

from opt_einsum import contract_path
from opt_einsum.parser import get_symbol
from typing import Sequence

from mrmustard import math
from ..physics.representations import Representation
from .circuit_components import CircuitComponent, connect
from .circuits import Circuit

__all__ = [
    "Simulator",
]


class Simulator:
    r"""
    A simulator for ``Circuit`` objects.
    """

    @staticmethod
    def _get_oe_subscripts(components: Sequence[CircuitComponent]) -> dict[str, Representation]:
        r"""
        Returns a list of subscripts for every component, together with a dictionary that maps
        each wire id to the corresponding subscript.
        """
        all_ids = set([id for c in components for id in c.wires.ids])
        ids_to_subs = {id: get_symbol(i) for i, id in enumerate(all_ids)}
        subs = ["".join([ids_to_subs[id] for id in c.wires.ids]) for c in components]
        return subs, ids_to_subs

    def _contract(self, components: Sequence[CircuitComponent]) -> CircuitComponent:
        r"""
        Contracts a sequence of Bargmann components.

        Arguments:
            components: The components to contract.

        Returns:
            The resulting circuit component.
        """
        # get a list of subscripts for every component
        subs, ids_to_subs = self._get_oe_subscripts(components)
        subs_to_component = {sub: c for (sub, c) in zip(subs, components)}

        # get the path for opt_einsum
        path = ",".join(subs)

        # use opt_einsum to get a list of pair-wise contractions
        shapes = [(2,) * len(sub) for sub in subs]
        path_info = contract_path(path, *shapes, shapes=True, optimize="auto")
        contractions = [ctr for _, _, ctr, _, _, in path_info[1].contraction_list]

        # initialize a dictionary mapping the unsorted subscripts (i.e., those received from
        # opt_einsum) to othe sorted ones (i.e., those with output subscripts on the left and with
        # input subscripts on the right)
        u_to_s_subscripts = {}
        for contraction in contractions:
            terms, result_u = contraction.split("->")
            term1, term2 = terms.split(",")
            u_to_s_subscripts[term1] = term1
            u_to_s_subscripts[term2] = term2
            u_to_s_subscripts[result_u] = result_u

        # perform the contractions in the order given by opt_einsum
        for contraction in contractions:
            terms_u, result_u = contraction.split("->")
            term1_u, term2_u = terms_u.split(",")

            # pop the sorted term1 and term2
            term1_s = u_to_s_subscripts.pop(term1_u)
            term2_s = u_to_s_subscripts.pop(term2_u)

            # store the "repeated" indices (those that appear in both terms)
            repeated = [s for s in term1_s if s in term2_s]

            # ensure that term1 is the term whose contracted subscripts are on the input side,
            # swapping term1 and term2 if needed
            if repeated and term1_s.index(repeated[0]) > term2_s.index(repeated[0]):
                term1_s, term2_s = term2_s, term1_s

            # pop the two circuit components involved in the contraction
            component1 = subs_to_component.pop(term1_s)
            component2 = subs_to_component.pop(term2_s)

            # calculate the ``Wires`` of the circuit component resulting from the contraction
            wires_out = component1.wires.add_connected(component2.wires)

            # calculate the ``Representation`` of the circuit component resulting from the contraction
            idx1 = [term1_s.index(i) for i in repeated]
            idx2 = [term2_s.index(i) for i in repeated]
            representation = component2.representation[idx2] @ component1.representation[idx1]

            # reorder the representation
            all_modes = component2.modes + component1.modes
            modes = list(set(all_modes))
            idx_reorder = math.concat(
                [[2 * all_modes.index(m), 2 * all_modes.index(m) + 1] for m in modes], axis=-1
            )
            representation = representation.reorder(idx_reorder)

            # initialize the circuit component resulting from the contraction
            component_out = CircuitComponent.from_attributes(
                name="", wires=wires_out, representation=representation
            )

            # store ``result_s`` and ``component_out`` in the relevant dictionaries
            result_s = "".join(
                [
                    ids_to_subs[i]
                    for i in wires_out.output.bra.ids
                    + wires_out.input.bra.ids
                    + wires_out.output.ket.ids
                    + wires_out.input.ket.ids
                ]
            )
            u_to_s_subscripts[result_u] = result_s
            subs_to_component[result_s] = component_out

        # return the remaining value of ``subs_to_component``
        ret = list(subs_to_component.values())[0]

        return ret

    def run(self, circuit: Circuit) -> CircuitComponent:
        r"""
        Simulates the given circuit.

        Note: This does not yet support measurements.

        Arguments:
            circuit: The circuit to simulate.

        Returns:
            A circuit component representing the entire circuit.
        """
        components = circuit.components
        if len(components) == 1:
            return components[0].light_copy()

        components = connect(circuit.components)
        return self._contract(components)

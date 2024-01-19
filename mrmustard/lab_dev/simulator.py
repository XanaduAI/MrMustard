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

from ..physics.representations import Representation
from .circuit_components import CircuitComponent, add_bra, connect
from .circuits import Circuit

__all__ = ["Simulator"]


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

        # initialize a dictionary mapping the "unsorted" subscripts (i.e., those received from
        # opt_einsum, which are not guaranteed to respect MrMustard's indexing convention) to the
        # "sorted" ones (i.e., the same subscripts, reordered to respect MrMustard's indexing
        # convention)
        u_to_s_subscripts = {}

        # perform the contractions in the order given by opt_einsum
        for contraction in contractions:
            # split `contraction` into unsorted subscripts
            terms_u, result_u = contraction.split("->")
            term1_u, term2_u = terms_u.split(",")

            # pop the sorted term1 and term2
            term1_s = u_to_s_subscripts.pop(term1_u, term1_u)
            term2_s = u_to_s_subscripts.pop(term2_u, term2_u)

            # pop the two circuit components involved in the contraction
            component1 = subs_to_component.pop(term1_s)
            component2 = subs_to_component.pop(term2_s)

            # ensure that component1 is the component whose contracted indices are on the output side,
            # swapping component1 with component2 if needed
            ids1 = component1.wires.input.ids
            ids2 = component2.wires.output.ids
            overlap = [i for i in ids1 if i in ids2]
            if overlap:
                term1_s, term2_s = term2_s, term1_s
                component1, component2 = component2, component1

            # calculate the ``Wires`` of the circuit component resulting from the contraction
            wires_out = component1.wires.add_connected(component2.wires)

            # get the string of sorted subscripts for the result of the contraction
            result_s = "".join(ids_to_subs[i] for i in wires_out.ids)

            # store the "repeated" indices (those that appear in both term1_s and term2_s)
            repeated = [s for s in term1_s if s in term2_s]

            # calculate the ``Representation`` of the circuit component resulting from the contraction
            idx1 = [term1_s.index(i) for i in repeated]
            idx2 = [term2_s.index(i) for i in repeated]
            representation = component2.representation[idx2] @ component1.representation[idx1]

            # reorder the representation
            all_subs = term2_s + term1_s
            for s in repeated:
                all_subs = all_subs.replace(s, "")
            idx_reorder = [all_subs.index(s) for s in result_s]
            representation = representation.reorder(idx_reorder)

            # initialize the circuit component resulting from the contraction
            component_out = CircuitComponent.from_attributes(
                name="", wires=wires_out, representation=representation
            )

            # store ``result_s`` and ``component_out`` in the relevant dictionaries
            u_to_s_subscripts[result_u] = result_s
            subs_to_component[result_s] = component_out

        # return the remaining value of ``subs_to_component``
        ret = list(subs_to_component.values())[0]

        return ret

    def run(self, circuit: Circuit, add_bras: bool = True) -> CircuitComponent:
        r"""
        Simulates the given circuit.

        Note: This does not yet support measurements.

        Arguments:
            circuit: The circuit to simulate.
            add_bras: If ``True``, adds the conjugate of every component that
                has wires only on the ket side.

        Returns:
            A circuit component representing the entire circuit.
        """
        components = circuit.components
        if len(components) == 1:
            return components[0].light_copy()

        if add_bras:
            components = add_bra(components)
        components = connect(components)
        return self._contract(components)

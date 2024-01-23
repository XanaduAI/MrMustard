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
        subs_to_rep = {sub: c.representation for (sub, c) in zip(subs, components)}

        # initialize a dictionary mapping the subscripts provided by opt_einsum to the subscripts
        # obtained when contracting
        opt_to_ctr_subscripts = {sub: sub for sub in subs}

        # get the path for opt_einsum
        path = ",".join(subs)

        # calculate the ``Wires`` of the returned component, alongside its substrings
        wires_out = components[0].wires
        for c in components[1:]:
            wires_out = wires_out.add_connected(c.wires)
        subs_out = "".join([ids_to_subs[id] for id in wires_out.ids])

        # use opt_einsum to get a list of pair-wise contractions
        shapes = [(2,) * len(sub) for sub in subs]
        path_info = contract_path(path, *shapes, shapes=True, optimize="auto")
        contractions = [ctr for _, _, ctr, _, _, in path_info[1].contraction_list]

        for ctr in contractions:
            # split `contraction` into subscripts, in the order provided by opt_einsum
            terms, result_opt = ctr.split("->")
            term1_opt, term2_opt = terms.split(",")
            term1, term2 = terms.split(",")

            # pop the subscripts of the terms undergoing the contraction
            term1 = opt_to_ctr_subscripts.pop(term1_opt)
            term2 = opt_to_ctr_subscripts.pop(term2_opt)

            # pop the two circuit components involved in the contraction
            rep1 = subs_to_rep.pop(term1)
            rep2 = subs_to_rep.pop(term2)

            # store the "repeated" indices that appear in both term1 and term2
            repeated = [s for s in term1 if s in term2]

            # multiply the two representations
            idx1 = [term1.index(i) for i in repeated]
            idx2 = [term2.index(i) for i in repeated]
            representation = rep1[idx1] @ rep2[idx2]

            # get the subscripts of the resulting representation
            result = "".join([s for s in term1 + term2 if s not in repeated])

            # store ``result`` and ``representation`` in the relevant dictionaries
            opt_to_ctr_subscripts[result_opt] = result
            subs_to_rep[result] = representation

        # grab the representation that remains in ``subs_to_rep``
        subs_out_u, representation_out = list(subs_to_rep.items())[0]

        # reorder the representation
        reorder_idx = [subs_out_u.index(i) for i in subs_out]
        representation_out = representation_out.reorder(reorder_idx)

        ret = CircuitComponent.from_attributes(
            name="",
            wires=wires_out,
            representation=representation_out,
        )

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

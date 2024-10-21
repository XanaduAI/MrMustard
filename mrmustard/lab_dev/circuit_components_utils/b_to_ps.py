# Copyright 2024 Xanadu Quantum Technologies Inc.

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
The class representing an operation that changes Bargmann into phase space.
"""
# pylint: disable=protected-access

from __future__ import annotations
from typing import Sequence

from mrmustard.physics import triples
from mrmustard.math.parameters import Constant

from ..transformations.base import Map
from ...physics.representations import Bargmann

__all__ = ["BtoPS"]


class BtoPS(Map):
    r"""The `s`-parametrized ``Dgate`` as a ``Map``.

    Used internally as a ``Channel`` for transformations between representations.

    Args:
        num_modes: The number of modes of this channel.
        s: The `s` parameter of this channel.
    """

    def __init__(
        self,
        modes: Sequence[int],
        s: float,
    ):
        super().__init__(
            modes_out=modes,
            modes_in=modes,
            representation=Bargmann.from_function(
                fn=triples.displacement_map_s_parametrized_Abc, s=s, n_modes=len(modes)
            ),
            name="BtoPS",
        )
        self._add_parameter(Constant(s, "s"))
        self.s = s

    @property
    def adjoint(self) -> BtoPS:
        bras = self.wires.bra.indices
        kets = self.wires.ket.indices
        rep = self.representation.reorder(kets + bras).conj()

        ret = BtoPS(self.modes, self.s)
        ret._representation = rep
        ret._wires = self.wires.adjoint
        ret._name = self.name + "_adj"
        return ret

    @property
    def dual(self) -> BtoPS:
        ok = self.wires.ket.output.indices
        ik = self.wires.ket.input.indices
        ib = self.wires.bra.input.indices
        ob = self.wires.bra.output.indices
        rep = self.representation.reorder(ib + ob + ik + ok).conj()

        ret = BtoPS(self.modes, self.s)
        ret._representation = rep
        ret._wires = self.wires.dual
        ret._name = self.name + "_dual"
        return ret

    def inverse(self) -> BtoPS:
        inv = super().inverse()
        ret = BtoPS(self.modes, self.s)
        ret._representation = inv.representation
        ret._wires = inv.wires
        ret._name = inv.name
        return ret

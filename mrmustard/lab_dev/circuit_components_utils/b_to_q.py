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
The class representing an operation that changes Bargmann into quadrature.
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import numbers

from mrmustard.physics import triples
from mrmustard.math.parameters import Constant

from ..transformations.base import Operation
from ...physics.ansatz import PolyExpAnsatz
from ...physics.representations import RepEnum
from ..circuit_components import CircuitComponent

__all__ = ["BtoQ"]


class BtoQ(Operation):
    r"""
    The Operation that changes the representation of an object from ``Bargmann`` into quadrature.
    By default it's defined on the output ket side. Note that beyond such gate we cannot place further
    ones unless they support inner products in quadrature representation.

    Args:
        modes: The modes of this channel.
        phi: The quadrature angle. 0 corresponds to the `x` quadrature, and :math:`\pi/2` to the `p` quadrature.
    """

    def __init__(
        self,
        modes: Sequence[int],
        phi: float = 0.0,
    ):
        repr = PolyExpAnsatz.from_function(
            fn=triples.bargmann_to_quadrature_Abc, n_modes=len(modes), phi=phi
        )
        super().__init__(
            modes_out=modes,
            modes_in=modes,
            representation=repr,
            name="BtoQ",
        )
        self._add_parameter(Constant(phi, "phi"))

    def __custom_rrshift__(self, other: CircuitComponent | complex) -> CircuitComponent | complex:
        if hasattr(other, "__custom_rrshift__"):
            return other.__custom_rrshift__(self)

        if isinstance(other, (numbers.Number, np.ndarray)):
            return self * other

        s_k = other.wires.ket
        s_b = other.wires.bra
        o_k = self.wires.ket
        o_b = self.wires.bra

        only_ket = (not s_b and s_k) and (not o_b and o_k)
        only_bra = (not s_k and s_b) and (not o_k and o_b)
        both_sides = s_b and s_k and o_b and o_k

        self_needs_bra = (not s_b and s_k) and (o_b and o_k)
        self_needs_ket = (not s_k and s_b) and (o_b and o_k)

        other_needs_bra = (s_b and s_k) and (not o_b and o_k)
        other_needs_ket = (s_b and s_k) and (not o_k and o_b)

        if only_ket or only_bra or both_sides:
            ret = other @ self
        elif self_needs_bra or self_needs_ket:
            ret = other.adjoint @ (other @ self)
        elif other_needs_bra or other_needs_ket:
            ret = (other @ self) @ self.adjoint
        else:
            msg = f"``>>`` not supported between {other} and {self} because it's not clear "
            msg += "whether or where to add bra wires. Use ``@`` instead and specify all the components."
            raise ValueError(msg)

        temp = dict.fromkeys(self.modes, RepEnum.QUADRATURE)
        ret._representation._wire_reps.update(temp)
        return self._rshift_return(ret)

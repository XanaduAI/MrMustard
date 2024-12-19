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

from mrmustard.physics import triples

from ..transformations.base import Operation
from ...physics.ansatz import PolyExpAnsatz
from ...physics.representations import RepEnum
from ..utils import make_parameter

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
        super().__init__(name="BtoQ")
        self.parameters.add_parameter(make_parameter(False, phi, "phi", (None, None)))
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.bargmann_to_quadrature_Abc, n_modes=len(modes), phi=self.parameters.phi
            ),
        ).representation
        for i in self.wires.input.indices:
            self.representation._idx_reps[i] = (RepEnum.BARGMANN, None)
        for i in self.wires.output.indices:
            self.representation._idx_reps[i] = (
                RepEnum.QUADRATURE,
                float(self.parameters.phi.value),
            )

    def inverse(self):
        ret = BtoQ(self.modes, self.parameters.phi)
        ret._representation = super().inverse().representation
        ret._representation._wires = ret.representation.wires.dual
        return ret

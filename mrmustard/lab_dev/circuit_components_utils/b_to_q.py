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
from mrmustard.math.parameters import Constant

from ..transformations.base import Operation
from ...physics.representations import Bargmann

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
        repr = Bargmann.from_function(
            fn=triples.bargmann_to_quadrature_Abc, n_modes=len(modes), phi=phi
        )
        super().__init__(
            modes_out=modes,
            modes_in=modes,
            representation=repr,
            name="BtoQ",
        )
        self._add_parameter(Constant(phi, "phi"))

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

from collections.abc import Sequence

from mrmustard.physics import triples
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexTensor

from ...physics.ansatz import PolyExpAnsatz
from ...physics.wires import ReprEnum
from ..transformations.base import Operation
from ..utils import make_parameter

__all__ = ["BtoQ"]


class BtoQ(Operation):
    r"""
    The ``Operation`` that changes the representation of an object from Bargmann (B) into quadrature (Q).
    By default it's defined on the output ket side.


    Args:
        modes: The modes of this channel.
        phi: The quadrature angle. 0 corresponds to the `x` quadrature, and :math:`\pi/2` to the `p` quadrature.


    Returns:
        An ``Operation`` type object that performs the change of representation.

    Note:
        Be cautious about contractions after change of representation as the Abc parametrization has altered.

    .. code-block::

        >>> from mrmustard import math
        >>> from mrmustard.lab import BtoQ, Ket, QuadratureEigenstate
        >>> psi = Ket.random([0])
        >>> assert math.allclose(psi >> QuadratureEigenstate(0, x=1).dual, (psi >> BtoQ(0)).ansatz(1))
    """

    def __init__(
        self,
        modes: int | tuple[int, ...],
        phi: float | Sequence[float] = 0.0,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(name="BtoQ")
        self.parameters.add_parameter(make_parameter(False, phi, "phi", (None, None)))

        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.bargmann_to_quadrature_Abc,
            n_modes=len(modes),
            phi=self.parameters.phi,
        )
        self._wires = Wires(modes_in_ket=set(modes), modes_out_ket=set(modes))
        for w in self.wires.input.sorted_wires:
            w.repr = ReprEnum.BARGMANN
        for w in self.wires.output.sorted_wires:
            w.repr = ReprEnum.QUADRATURE

    def inverse(self):
        if self.modes == ():
            return self
        ret = BtoQ(self.modes, self.parameters.phi)
        ret._ansatz = super().inverse().ansatz
        ret._wires = ret.wires.dual
        return ret

    def fock_array(self, shape: int | Sequence[int] | None = None) -> ComplexTensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not have a Fock representation.")

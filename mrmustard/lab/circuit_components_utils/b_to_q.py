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

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexTensor

from ...physics.wires import ReprEnum
from ..transformations.base import Operation
from .builtins import bargmann_to_quadrature

__all__ = ["BtoQ"]


class BtoQ(Operation):
    r"""
    The ``Operation`` that changes the representation of an object from Bargmann (B) into quadrature (Q).
    By default it's defined on the output ket side.

    >>> from mrmustard import math
    >>> from mrmustard.lab import BtoQ, GaussianKet, QuadratureEigenstate
    >>> psi = GaussianKet.random([0])
    >>> assert math.allclose(psi >> QuadratureEigenstate(0, x=1).dual, (psi >> BtoQ(0)).ansatz(1))

    Args:
        modes: The modes of this channel.
        phi: The quadrature angle. ``0`` corresponds to the `x` quadrature, and :math:`\pi/2` to the `p` quadrature.
    """

    short_name = "BtoQ"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        phi: float | Sequence[float] = 0.0,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (bargmann_to_quadrature, ("n_modes", "phi", "lin_sup"))
                },
                n_modes=len(modes),
            ),
            wires=Wires(modes_in_ket=set(modes), modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")
        for w in self.wires.input.standard_order:
            w.repr = ReprEnum.BARGMANN
        for w in self.wires.output.standard_order:
            w.repr = ReprEnum.QUADRATURE

    def inverse(self):
        if self.modes == ():
            return self
        ret = BtoQ(self.modes, self.parameters.phi)
        ret_inverse = super().inverse()
        ret._ansatz_factory = ret_inverse.ansatz_factory
        ret._wires = ret.wires.dual
        ret._wires._reindex()
        return ret

    def fock_array(self, shape: int | Sequence[int] | None = None) -> ComplexTensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not have a Fock representation.")

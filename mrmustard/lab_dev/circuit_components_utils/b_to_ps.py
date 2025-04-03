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

from __future__ import annotations

from mrmustard.physics import triples

from ..transformations.base import Map
from ...physics.ansatz import PolyExpAnsatz
from ...physics.wires import ReprEnum
from ..utils import make_parameter

__all__ = ["BtoPS"]


class BtoPS(Map):
    r"""The `s`-parametrized ``Dgate`` as a ``Map``.
    The `s`-parametrized ``Dgate`` as a ``Map``.

    Used internally as a ``Channel`` for transformations between representations.


    Args:
        modes: The modes on which the transformation is to be applied.
        s: The `s` parameter of this channel.


    Returns:
        a ``Channel`` type object that applies the Bargmann to Phase Space transformation
        on any input.

    .. details::
        This class represents the transformation from the Bargmann (B) representation
        to phase space (PS).

        Any operator, say O can be expressed in the displacement basis. Formally, we have that
        the s-parametrized phase space basis is characterized by the following operators

        .. math::

            D_s(\alpha) = exp(s|\alpha|^2/2) D(\alpha).
        The s-parametrized phase space representation of an object O, would therefore be

        .. math::

            mathrm{tr}(D_s(\alpha) O).

        Important s-parametrizations include:
        - s=1: returns the complex Fourier transform (or often
        called the symplectic Fourier transform) of Galuber-Sudarshan P function.

        - s=0: returns the characteristic function, which is equivalent to the complex Fourier
        transform of the Wigner function.

        - s=-1: returns the complex Fourier transform of the Q function.

    .. code-block::

        >>> from mrmustard.lab_dev import BtoPS, Ket
        >>> from mrmustard import math

        >>> chi = (Ket.random([0]) >> BtoPS([0], s=0)).ansatz
        >>> assert math.allclose(chi(0,0), 1.0)
    """

    def __init__(
        self,
        modes: int | tuple[int, ...],
        s: float,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(name="BtoPS")
        self.parameters.add_parameter(make_parameter(False, s, "s", (None, None)))
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.displacement_map_s_parametrized_Abc,
                s=self.parameters.s,
                n_modes=len(modes),
            ),
        ).representation
        for w in self.representation.wires.output.wires:
            w.repr = ReprEnum.CHARACTERISTIC
            w.repr_params_func = lambda: self.parameters.s

    def inverse(self):
        ret = BtoPS(self.modes, self.parameters.s)
        ret._representation = super().inverse().representation
        ret._representation._wires = ret.representation.wires.dual
        return ret

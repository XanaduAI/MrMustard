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

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import ComplexTensor

from ..transformations.base import Map
from .builtins import displacement_map_s_parametrized

__all__ = ["BtoChar"]


class BtoChar(Map):
    r"""
    The ``s``-parametrized ``Dgate`` as a ``Map``. Also known as the Fourier transform of the Stratonovich-Weyl kernel.
    See https://arxiv.org/abs/quant-ph/9707010.

    This is an unphysical component whose purpose is to modify the internal representation of another component.
    In particular it transforms between the Bargmann representation and the s-parametrized Characteristic functions.
    Note that it can be applied to a subset of modes.

    >>> from mrmustard.lab import BtoChar, GaussianKet
    >>> from mrmustard import math
    >>> chi = (GaussianKet.random([0]) >> BtoChar([0], s=0)).ansatz
    >>> assert math.allclose(chi(0,0), 1.0)

    Args:
        modes: The modes of this channel.
        s: The ``s`` parameter of this channel.

    .. details::

        This class represents the transformation from the Bargmann (B) representation
        to characteristic function (Char).

        Any operator, say O can be expressed in the displacement basis. Formally, we have that
        the s-parametrized phase space basis is characterized by the following operators

        .. math::

            D_s(\alpha) = exp(s|\alpha|^2/2) D(\alpha).

        The s-parametrized phase space representation of an object O, would therefore be

        .. math::

            \mathrm{Tr}(D_s(\alpha) O).

        Important s-parametrizations include:

        - s=1: returns the complex Fourier transform (or often
          called the symplectic Fourier transform) of Galuber-Sudarshan P function.

        - s=0: returns the characteristic function, which is equivalent to the complex Fourier
          transform of the Wigner function.

        - s=-1: returns the complex Fourier transform of the Q function.
    """

    short_name = "BtoCh"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        s: float,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (
                        displacement_map_s_parametrized,
                        ("s", "n_modes", "lin_sup"),
                    )
                },
                n_modes=len(modes),
            ),
            wires=Wires(
                modes_in_bra=set(modes),
                modes_out_bra=set(modes),
                modes_in_ket=set(modes),
                modes_out_ket=set(modes),
            ),
            name=self.__class__.__name__,
        )
        self.parameters["s"] = Parameter.from_cc_init(s, "float64", f"{self.name}/s")
        for w in self.wires.output.standard_order:
            w.repr = ReprEnum.CHARACTERISTIC

    def inverse(self):
        ret = BtoChar(self.modes, self.parameters.s)
        ret_inverse = super().inverse()
        ret._ansatz_factory = ret_inverse.ansatz_factory
        ret._wires = ret.wires.dual
        ret._wires._reindex()
        return ret

    def fock_array(self, shape: int | Sequence[int] | None = None) -> ComplexTensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not have a Fock representation.")

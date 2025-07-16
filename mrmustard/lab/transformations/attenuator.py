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
The class representing a noisy attenuator channel.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.wires import Wires

from ...physics import triples
from ...physics.ansatz import PolyExpAnsatz
from ..utils import make_parameter
from .base import Channel

__all__ = ["Attenuator"]


class Attenuator(Channel):
    r"""
    The noisy attenuator channel.


    Args:
        mode: The mode this gate is applied to.
        transmissivity: The transmissivity.
        transmissivity_trainable: Whether ``transmissivity`` is trainable.
        transmissivity_bounds: The bounds for ``transmissivity``.

    .. code-block::

        >>> from mrmustard import math
        >>> from mrmustard.lab import Attenuator

        >>> channel = Attenuator(mode=1, transmissivity=0.1)
        >>> assert channel.modes == (1,)
        >>> assert channel.parameters.transmissivity.value == 0.1

    .. details::

        The :math:`N`-mode attenuator is defined as

        .. math::
            X = \text{cos}(\theta)I_{2N} \text{ , }
            Y = \text{sin}^2(\theta)I_{2N} \text{ , and }
            d = O_{4N}\:,

        where the :math:`\theta=\text{arcos}(\sqrt{\bar{\eta}})`, :math:`\eta` is the transmissivity, and
        :math:`\text{diag}_N(\bar{\eta})` is the :math:`N\text{x}N` matrix with diagonal :math:`\bar{\eta}`.

        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= \begin{bmatrix}
                    O_N & \text{diag}_N(\sqrt{\bar{\eta}}) & O_N & O_N \\
                    \text{diag}_N(\sqrt{\bar{\eta}}) & O_N & O_N & \text{diag}_N(1-\sqrt{\bar{\eta}})\\
                    O_N & O_N & O_N & \text{diag}_N(\sqrt{\bar{\eta}})\\
                    O_N & \text{diag}_N(1-\sqrt{\bar{\eta}}) & \text{diag}_N(\sqrt{\bar{\eta}}) & O_N
                \end{bmatrix} \\ \\
            b &= O_{4N} \\ \\
            c &= 1\:.
    """

    short_name = "Att~"

    def __init__(
        self,
        mode: int | tuple[int],
        transmissivity: float | Sequence[float] = 1.0,
        transmissivity_trainable: bool = False,
        transmissivity_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="Att~")
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=transmissivity_trainable,
                value=transmissivity,
                name="transmissivity",
                bounds=transmissivity_bounds,
                dtype=math.float64,
            ),
        )

        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.attenuator_Abc,
            eta=self.parameters.transmissivity,
        )
        self._wires = Wires(
            modes_in_bra=set(mode),
            modes_out_bra=set(mode),
            modes_in_ket=set(mode),
            modes_out_ket=set(mode),
        )

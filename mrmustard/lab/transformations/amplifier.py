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
The class representing a noisy amplifier channel.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.wires import Wires

from ...physics import triples
from ...physics.ansatz import PolyExpAnsatz
from ..utils import make_parameter
from .base import Channel

__all__ = ["Amplifier"]


class Amplifier(Channel):
    r"""
    The noisy amplifier channel.


    Args:
        mode: The mode this gate is applied to.
        gain: The gain.
        gain_trainable: Whether ``gain`` is trainable.
        gain_bounds: The bounds for ``gain``.

    .. code-block::

        >>> import numpy as np
        >>> from mrmustard.lab import Amplifier, Coherent
        >>> from mrmustard import settings

        >>> amp = Amplifier(0, gain=4)
        >>> coh = Coherent(0, x=1.0, y=2.0)
        >>> _, mu, _ = (coh >> amp).phase_space(0)
        >>> assert np.allclose(mu*np.sqrt(2/settings.HBAR), np.array([4.0, 8.0]))

    .. details::

        The :math:`N`-mode attenuator is defined as

        .. math::
            X = /sqrt{/bar{g}}I_{2N} \text{ , }
            Y = (/bar{g}-1)I_{2N} \text{ , and }
            d = O_{4N}\:,

        where :math:`/bar{g}` is the gain and
        :math:`\text{diag}_N(\bar{g})` is the :math:`N\text{x}N` matrix with diagonal :math:`\bar{g}`.

        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= \begin{bmatrix}
                    O_N & \text{diag}_N(1/(\sqrt{\bar{g}}) & \text{diag}_N(1-1/\bar{g}) & O_N \\
                    \text{diag}_N(1/(\sqrt{\bar{g}}) & O_N & O_N & O_N \\
                    \text{diag}_N(1-1/\bar{g})  & O_N & O_N &  \text{diag}_N(1/(\bar{g})\\
                    O_N & O_N &  \text{diag}_N(1/(\sqrt{\bar{g}}) & O_N
                \end{bmatrix} \\ \\
            b &= O_{4N} \\ \\
            c &= 1//bar{g}\:.
    """

    def __init__(
        self,
        mode: int | tuple[int],
        gain: float | Sequence[float] = 1.0,
        gain_trainable: bool = False,
        gain_bounds: tuple[float | None, float | None] = (1.0, None),
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="Amp~")
        self.parameters.add_parameter(
            make_parameter(
                is_trainable=gain_trainable,
                value=gain,
                name="gain",
                bounds=gain_bounds,
                dtype=math.float64,
            ),
        )
        self._ansatz = PolyExpAnsatz.from_function(fn=triples.amplifier_Abc, g=self.parameters.gain)
        self._wires = Wires(
            modes_in_bra=set(mode),
            modes_out_bra=set(mode),
            modes_in_ket=set(mode),
            modes_out_ket=set(mode),
        )

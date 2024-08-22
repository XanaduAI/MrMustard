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

from typing import Sequence

from .base import Channel
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter, reshape_params

__all__ = ["Amplifier"]


class Amplifier(Channel):
    r"""
    The noisy amplifier channel.

    If ``gain`` is an iterable, its length must be equal to `1` or `N`. If it length is equal to `1`,
    all the modes share the same gain.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Amplifier, Coherent
        >>> from mrmustard import settings

        >>> amp = Amplifier([0], gain=4)
        >>> coh = Coherent([0], x=1.0, y=2.0)
        >>> _, mu, _ = (coh >> amp).phase_space(0)
        >>> assert np.allclose(mu[0]*np.sqrt(2/settings.HBAR), np.array([4.0, 8.0]))

    Args:
        modes: The modes this gate is applied to.
        gain: The gain.
        gain_trainable: Whether the gain is a trainable variable.
        gain_bounds: The bounds for the gain.

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
        modes: Sequence[int],
        gain: float | Sequence[float] | None = 1.0,
        gain_trainable: bool = False,
        gain_bounds: tuple[float | None, float | None] = (1.0, None),
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="Amp")
        (gs,) = list(reshape_params(len(modes), gain=gain))
        self._add_parameter(
            make_parameter(
                gain_trainable,
                gs,
                "gain",
                gain_bounds,
                None,
            )
        )
        self._representation = Bargmann.from_function(fn=triples.amplifier_Abc, g=self.gain)

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

from typing import Sequence

from .base import Channel
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter, reshape_params

__all__ = ["Attenuator"]


class Attenuator(Channel):
    r"""
    The noisy attenuator channel.

    If ``transmissivity`` is an iterable, its length must be equal to `1` or `N`. If it length is equal to `1`,
    all the modes share the same transmissivity.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Attenuator

        >>> channel = Attenuator(modes=[1, 2], transmissivity=0.1)
        >>> assert channel.modes == [1, 2]
        >>> assert np.allclose(channel.transmissivity.value, [0.1, 0.1])

    Args:
        modes: The modes this gate is applied to.
        transmissivity: The transmissivity.
        transmissivity_trainable: Whether the transmissivity is a trainable variable.
        transmissivity_bounds: The bounds for the transmissivity.

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

    short_name = "Att"

    def __init__(
        self,
        modes: Sequence[int],
        transmissivity: float | Sequence[float] | None = 1.0,
        transmissivity_trainable: bool = False,
        transmissivity_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="Att")
        (etas,) = list(reshape_params(len(modes), transmissivity=transmissivity))
        self._add_parameter(
            make_parameter(
                transmissivity_trainable,
                etas,
                "transmissivity",
                transmissivity_bounds,
                None,
            )
        )
        self._representation = Bargmann.from_function(
            fn=triples.attenuator_Abc, eta=self.transmissivity
        )

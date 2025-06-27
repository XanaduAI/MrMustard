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
from math import comb

import numpy as np

from mrmustard import settings

from ..states.ket import Ket
from ..utils import make_parameter
from .base import Channel

__all__ = ["FockAttenuator"]


class FockAttenuator(Channel):
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

    """

    short_name = "Att~"

    def __init__(
        self,
        mode: int,
        transmissivity: float | Sequence[float] = 1.0,
        transmissivity_trainable: bool = False,
        transmissivity_bounds: tuple[float | None, float | None] = (0.0, 1.0),
    ):
        super().__init__(name="Att~")
        self.parameters.add_parameter(
            make_parameter(
                transmissivity_trainable,
                transmissivity,
                "transmissivity",
                transmissivity_bounds,
                None,
            ),
        )
        self.mode = mode
        self.rng = np.random.default_rng(settings.SEED)

    def loss_kraus(self, lost: int, N: int, eta: float) -> np.ndarray:
        """
        Create a Kraus operator for loss with a given loss coefficient.

        Args:
            lost (int): The loss coefficient.
            N (int): The number of modes.

        Returns:
            np.ndarray: The Kraus operator for the loss channel.
        """
        kraus = np.zeros((N, N), dtype=np.complex64)
        for n in range(lost, N):
            kraus[n - lost, n] = (
                np.sqrt(comb(n, lost)) * (1 - eta) ** (lost / 2) * eta ** ((n - lost) / 2)
            )
        return kraus

    def __custom_rrshift__(self, other: Ket) -> Ket:
        r"""
        Returns the custom right-shift ansatz for the attenuator channel.

        Args:
            mode: The mode this gate is applied to.

        Returns:
            A ``Ket``.
        """
        if not isinstance(other, Ket):
            raise TypeError("FockAttenuator can only be right-shifted with a Ket.")

        N = other.fock_array().shape[0]
        kraus = [self.loss_kraus(l=i, N=N, eta=self.tr) for i in range(N)]

        probs = [np.linalg.norm(kraus[i] @ other.fock_array()) ** 2 for i in range(N)]
        probs /= np.sum(probs)

        chosen_kraus = self.rng.choice(range(N), p=probs)
        return Ket.from_fock(other.modes, kraus[chosen_kraus] @ other.fock_array()).normalize()

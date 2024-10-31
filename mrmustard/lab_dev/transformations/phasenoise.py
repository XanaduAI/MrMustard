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
The class representing a Phase noise channel.
"""

from __future__ import annotations
from typing import Sequence
from mrmustard import math, settings
from .base import Channel
from .attenuator import Attenuator
from ...physics.representations import Fock
from ..utils import make_parameter

__all__ = ["PhaseNoise"]


class PhaseNoise(Channel):
    r"""
    The Phase noise channel.

    This class represents the application of a random phase. The distributiuon of the phase
    is assumed to be a Gaussian with mean zero, and standard deviation `sigma`.

    Args:
        modes: The modes the channel is applied to
        sigma: the standard deviation of the random Gaussian noise (could be provided as a list, if the channel acts on more than one mode)

    ..details::
        The Fock representation is connected to the Fourier coefficients of the distribution.

    """

    short_name = "P~"
    # randomized : bool
    def __init__(
        self,
        modes: Sequence[int],
        phase_stdev: float | Sequence[float],
        phase_stdev_trainable: bool = False,
        phase_stdev_bounds: tuple[float | None, float | None] = (0.0, None),
    ):
        super().__init__(name="PhN")
        self._add_parameter(
            make_parameter(phase_stdev_trainable, phase_stdev, "phase_stdev", phase_stdev_bounds)
        )

    def __custom_rrshift__(self, other):
        r"""
        Custom rrshift
        """
        # check if Ket or DM: do the specific matmul
        # raise exception if not
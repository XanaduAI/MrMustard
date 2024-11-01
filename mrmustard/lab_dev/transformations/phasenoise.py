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
from mrmustard import math
from .base import Channel
from ..states import Ket, DM
from ..utils import make_parameter
import numpy as np

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
        self._representation = self.from_ansatz(
            modes_in=modes, modes_out=modes, ansatz=None
        ).representation

    def __custom_rrshift__(self, other):
        r"""
        Custom rrshift
        """
        if isinstance(other, Ket):
            other = other.dm()
        if isinstance(other, DM):
            array = math.asnumpy(other.fock_array())
            for mode in self.modes:
                for count, _ in enumerate(np.nditer(array)):
                    idx = np.zeros(len(array.shape))
                    temp = count
                    for l in range(len(idx)):
                        idx[-1 - l] = temp % array.shape[-1 - l]
                        temp = temp // array.shape[-1 - l]
                    array_index = tuple(idx.astype(int))
                    array[array_index] *= np.exp(
                        -0.5
                        * (idx[mode] - idx[other.n_modes + mode]) ** 2
                        * self.phase_stdev.value**2
                    )
            return DM.from_fock(other.modes, array, self.name)
        else:
            raise ValueError("The PhaseNoise object can only be applied to a DM or a Ket.")

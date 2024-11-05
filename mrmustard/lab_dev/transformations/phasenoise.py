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
from .base import Channel
from ..utils import make_parameter
import numpy as np
from mrmustard import math
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz
from mrmustard.physics.representations import Representation

__all__ = ["PhaseNoise"]


class PhaseNoise(Channel):
    r"""
    The Phase noise channel.

    This class represents the application of a random phase. The distributiuon of the phase
    is assumed to be a Gaussian with mean zero, and standard deviation `phase_stdev`.

    Args:
        modes: The modes the channel is applied to
        phase_stdev: The standard deviation of the random phase noise.

    ..details::
        The Fock representation is connected to the Fourier coefficients of the distribution.
    """

    short_name = "P~"

    def __init__(
        self,
        modes: Sequence[int],
        phase_stdev: float | Sequence[float],
        phase_stdev_trainable: bool = False,
        phase_stdev_bounds: tuple[float | None, float | None] = (0.0, None),
    ):
        super().__init__(name="PhaseNoise")
        self._add_parameter(
            make_parameter(phase_stdev_trainable, phase_stdev, "phase_stdev", phase_stdev_bounds)
        )
        self._representation = self.from_ansatz(
            modes_in=modes, modes_out=modes, ansatz=None
        ).representation

    def __custom_rrshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Since PhaseNoise admits a particularly nice form in the Fock basis, we have implemented its right-shift operation separately.

        Args:
            other: the component other than the PhaseNoise object that is present in the contraction

        Output:
            the result of the contraction.
        """

        if not other.wires.bra or not other.wires.ket:
            other = other @ other.adjoint
        array = math.asnumpy(other.fock_array())
        mode_indices = np.indices(array.shape)
        for mode in self.modes:
            phase_factors = np.exp(
                -0.5
                * (mode_indices[mode] - mode_indices[other.n_modes + mode]) ** 2
                * self.phase_stdev.value**2
            )
            array *= phase_factors
        return CircuitComponent(Representation(ArrayAnsatz(array, False), other.wires), self.name)

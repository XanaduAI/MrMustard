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

import numpy as np

from mrmustard import math
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz
from mrmustard.physics.wires import Wires

from ..utils import make_parameter
from .base import Channel

__all__ = ["PhaseNoise"]


class PhaseNoise(Channel):
    r"""
    The Phase noise channel.

    This class represents the application of a random phase. The distributiuon of the phase
    is assumed to be a Gaussian with mean zero, and standard deviation `phase_stdev`.

    Args:
        mode: The mode the channel is applied to.
        phase_stdev: The standard deviation of the random phase noise.
        phase_stdev_trainable: Whether ``phase_stdev`` is trainable.
        phase_stdev_bounds: The bounds for ``phase_stdev``.

    .. code-block::

        >>> from mrmustard.lab import PhaseNoise, Coherent, DM
        >>> phase_noise = PhaseNoise(0, phase_stdev=0.5)
        >>> assert isinstance(Coherent(0, 1) >> phase_noise, DM)

    .. details::
        The Fock representation is connected to the Fourier coefficients of the distribution.
    """

    short_name = "P~"

    def __init__(
        self,
        mode: int | tuple[int],
        phase_stdev: float = 0.0,
        phase_stdev_trainable: bool = False,
        phase_stdev_bounds: tuple[float | None, float | None] = (0.0, None),
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="PhaseNoise")
        self.parameters.add_parameter(
            make_parameter(
                phase_stdev_trainable,
                phase_stdev,
                "phase_stdev",
                phase_stdev_bounds,
                dtype=math.float64,
            ),
        )
        self._ansatz = None
        self._wires = Wires(
            modes_in_bra=set(mode),
            modes_out_bra=set(mode),
            modes_in_ket=set(mode),
            modes_out_ket=set(mode),
        )

    def __custom_rrshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Since PhaseNoise admits a particularly nice form in the Fock basis, we have implemented its right-shift operation separately.

        Args:
            other: the component other than the PhaseNoise object that is present in the contraction

        Output:
            the result of the contraction.
        """
        if not other.wires.bra or not other.wires.ket:
            other = other.contract(other.adjoint, "zip")
        other = other.to_fock()
        array = other.fock_array()
        mode_indices = np.indices(other.ansatz.core_shape)
        for mode in self.modes:
            phase_factors = math.exp(
                -0.5
                * (mode_indices[mode] - mode_indices[other.n_modes + mode]) ** 2
                * self.parameters.phase_stdev.value**2,
            )
            array *= phase_factors
        return CircuitComponent._from_attributes(
            ArrayAnsatz(array, batch_dims=other.ansatz.batch_dims),
            other.wires,
            self.name,
        )

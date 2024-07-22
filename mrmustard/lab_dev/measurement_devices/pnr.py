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
This module contains the base classes for the available measurements.
"""

from __future__ import annotations

import numpy as np

from numbers import Number
from typing import Optional, Sequence

from ..states import Number, ConditionalState
from .base import MeasurementDevice
from ..circuit_components import CircuitComponent
from mrmustard import settings

__all__ = ["PNR"]


class PNR(MeasurementDevice):
    r"""
    The Photon Number Resolving (PNR) detector.
    For a given cutoff :math:`c`, a PNR detector is a ``Detector`` with measurement operators
    :math:`|0\rangle, |1\rangle, \ldots, |c\rangle`, where :math:`|n\rangle` corresponds to
    the state ``Number([mode], n, cutoff)``.
    """

    def __init__(
        self,
        modes: Sequence[int],
        cutoff: Optional[int] = None,
    ):
        self._cutoff = cutoff or settings.AUTOCUTOFF_MAX_CUTOFF
        super().__init__(
            modes=set(modes),
            name="PNR",
            sampling_technique=[Number(modes, n, self.cutoff) for n in range(self.cutoff + 1)],
        )

    @property
    def cutoff(self) -> int:
        r"""
        The cutoff of this PNR.
        """
        return self._cutoff
    
    def __custom_rrshift__(self, other: CircuitComponent | complex) -> ConditionalState | MeasurementDevice:
        r"""A custom ``>>`` operator for the ``PNR`` component.
        It allows ``PNR`` to carry the method that processes ``other >> PNR``.
        """
        # only if other is state right now
        # if other is not a state then return new MeasurementDevice

        # if isinstance(other, State):
        #     return ConditionalState()
        # else:
        #     return MeasurementDevice()

        # works if a mode is left over
        # if all modes measured should return an array of probabilities?
        modes = [mode for mode in other.modes if mode not in self.modes]
        states = [other >> meas_op.dual for meas_op in self.sampling_technique]
        ret = ConditionalState(modes, range(len(self.sampling_technique)), states)

        # this should be handled by self.sampling_technique.sample
        a = list(range(len(self.sampling_technique)))
        p = [ret.state_outcomes[i].probability for i in range(len(a))]
        rng = np.random.default_rng()
        meas_outcome = rng.choice(a=a, p=p)

        ret.set_state(meas_outcome)
        return ret
    

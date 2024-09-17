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
This module contains the PNR class.
"""

from __future__ import annotations

import numpy as np

from typing import Sequence

from .base import MeasurementDevice
from ..circuit_components import CircuitComponent, Wires
from ..sampler import PNRSampler
from mrmustard import settings, math

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
        cutoff: int | None = None,
    ):
        self._cutoff = cutoff or settings.AUTOCUTOFF_MAX_CUTOFF
        super().__init__(
            modes=modes,
            sampler=PNRSampler(self.cutoff),
            name="PNR",
        )

    @property
    def cutoff(self) -> int:
        r"""
        The cutoff of this PNR.
        """
        return self._cutoff

    def __custom_rrshift__(self, other: CircuitComponent | complex) -> CircuitComponent | float:
        r"""
        A custom ``>>`` operator for the ``PNR`` component.
        It allows ``PNR`` to carry the method that processes ``other >> PNR``.
        """
        #     sample = self.sampler.sample(other[self.modes], 1)[0]
        #     return sample if other.modes == self.modes else self.sampler._reduced_state(other, sample, self.modes)

        if isinstance(other, CircuitComponent):
            wires = Wires(
                modes_out_bra=set(np.setdiff1d(list(other.wires.args[0]), self.modes)),
                modes_in_bra=other.wires.args[1],
                modes_out_ket=set(np.setdiff1d(list(other.wires.args[2]), self.modes)),
                modes_in_ket=other.wires.args[3],
                classical_out=set(self.modes),
            )
            try:
                ret = other._getitem_builtin(other.modes)
                ret._wires = wires
            except TypeError:
                ret = other._from_attributes(other.representation, self.wires, self.name)
                ret._wires = wires
            return ret

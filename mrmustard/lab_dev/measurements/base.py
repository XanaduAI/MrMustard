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

from typing import Optional, Sequence

from mrmustard.physics.representations import Fock
from ..states import Ket
from ..circuit_components import CircuitComponent

__all__ = ["Measurement", "Detector"]


class Measurement(CircuitComponent):
    r"""
    Base class for all measurements.
    """


class Detector(Measurement):
    r"""
    Base class for all detectors.

    Arguments:
        name: The name of this detector.
        modes: The modes that this detector acts on.
        meas_op: A sequence of ket-like circuit components representing the set of operators
            for this measurement.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        modes: tuple[int, ...] = (),
        meas_op: Optional[Sequence[Ket]] = None,
    ):
        super().__init__(
            name or "D" + "".join(str(m) for m in modes), modes_in_bra=modes, modes_in_ket=modes
        )
        self._meas_op = meas_op
        self._representation = None

    @property
    def meas_op(self):
        return self._meas_op

    @property
    def representation(self):
        if not self._representation:
            array = [k.representation.array for k in self.meas_op]
            self._representation = Fock(array, True).outer(Fock(array, True))
        return self._representation

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Detector")

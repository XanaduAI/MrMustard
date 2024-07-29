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
This module contains the base classes for measurement devices.
"""

from __future__ import annotations

from typing import Optional, Any

from ..circuit_components import CircuitComponent
from ..sampler import Sampler

__all__ = ["MeasurementDevice"]


class MeasurementDevice(CircuitComponent):
    r"""
    Base class for all measurement devices.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        modes: tuple[int, ...] = (),
        sampler: Optional[Sampler] = None,
    ):
        super().__init__(
            name=name or "MD" + "".join(str(m) for m in modes),
            modes_in_bra=modes,
            modes_in_ket=modes,
        )

        # thinking this is either Sampler / ProbabilityDistribution / POVMs
        self._sampler = sampler

    @property
    def sampler(self):
        r""" """
        return self._sampler

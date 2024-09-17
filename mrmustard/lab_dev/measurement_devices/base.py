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

import numpy as np

from ..circuit_components import CircuitComponent, Wires
from ..conditional_circuit_components import ConditionalCircuitComponent

from ..sampler import Sampler

__all__ = ["MeasurementDevice"]


class MeasurementDevice(CircuitComponent):
    r"""
    Base class for all measurement devices.
    """

    def __init__(
        self,
        modes: tuple[int, ...] = (),
        sampler: Sampler | None = None,
        name: str | None = None,
    ):
        super().__init__(
            representation=None,
            wires=[(), modes, (), modes],
            name=name or "MD" + "".join(str(m) for m in modes),
        )

        self._sampler = sampler

    @property
    def sampler(self):
        r""" """
        return self._sampler

    def __custom_rrshift__(self, other: CircuitComponent | complex) -> CircuitComponent | float:
        r"""
        A custom ``>>`` operator for the ``PNR`` component.
        It allows ``PNR`` to carry the method that processes ``other >> PNR``.
        """
        if isinstance(other, ConditionalCircuitComponent):
            return other

        elif isinstance(other, CircuitComponent):
            wires = Wires(
                modes_out_bra=set(np.setdiff1d(list(other.wires.args[0]), self.modes)),
                modes_in_bra=other.wires.args[1],
                modes_out_ket=set(np.setdiff1d(list(other.wires.args[2]), self.modes)),
                modes_in_ket=other.wires.args[3],
                classical_out=set(self.modes),
            )
            ret = ConditionalCircuitComponent(other, wires, "")

            for mode in self.modes:
                ret._meas_devices[mode] = self.sampler
                ret._meas_outcomes[mode] = self.sampler.sample(other[mode], 1)[0]

            return ret

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
Simulators for quantum circuits.
"""

from __future__ import annotations

from abc import ABC, abstractclassmethod

from ..math.parameter_set import ParameterSet
from .circuits import Circuit
from .wires import Wire, Wires

__all__ = ["Simulator", "SimulatorBargmann"]


class Simulator(ABC):
    r"""
    A base class for simulators.
    """

    @abstractclassmethod
    def run(self, circuit: Circuit) -> any:
        r"""
        Simulates the given circuit.

        Arguments:
            circuit: The circuit to simulate.
        """
        ...


class SimulatorBargmann(Simulator):
    pass

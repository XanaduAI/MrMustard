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
A base class for the components of quantum circuits.
"""

# pylint: disable=super-init-not-called, protected-access, import-outside-toplevel
from __future__ import annotations
from typing import Sequence


from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.wires import Wires

from mrmustard.lab_dev.sampler import Sampler

__all__ = ["ConditionalCircuitComponent"]


class ConditionalCircuitComponent(CircuitComponent):
    r""" """

    def __init__(self, cc: CircuitComponent, wires: Wires, name: str | None = None) -> None:
        super().__init__(None, wires, name)
        self._base_cc = cc
        self._meas_devices = {}
        self._meas_outcomes = {}

    @property
    def representation(self):
        temp = self._base_cc
        for mode, meas_outcome in self._meas_outcomes.items():
            temp = self._meas_devices[mode]._reduce(temp, meas_outcome, mode)
        return temp.representation

    def post_select(self, meas_outcomes):
        r""" """
        pass

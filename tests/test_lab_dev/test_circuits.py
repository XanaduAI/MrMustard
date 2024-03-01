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

"""Tests for the ``Circuit`` class."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.lab_dev.circuits import Circuit
from mrmustard.lab_dev.states import Vacuum
from mrmustard.lab_dev.transformations import Attenuator, BSgate, Channel, Dgate, Sgate, Unitary


class TestCircuit:
    r"""
    Tests for the ``Circuit`` class.
    """

    def test_init(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ = Circuit([vac, s01, bs01, bs12])
        assert circ.components == [vac, s01, bs01, bs12]

    def test_eq(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])

        assert Circuit([vac, s01]) == Circuit([vac, s01])
        assert Circuit([vac, s01]) != Circuit([vac, s01, bs01])

    def test_len(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])

        assert len(Circuit([vac])) == 1
        assert len(Circuit([vac, s01])) == 2
        assert len(Circuit([vac, s01, s01])) == 3

    def test_get_item(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ = Circuit([vac, s01, bs01, bs12])
        assert circ.components[0] == vac
        assert circ.components[1] == s01
        assert circ.components[2] == bs01
        assert circ.components[3] == bs12

    def test_rshift(self):
        vac = Vacuum([0, 1, 2])
        s01 = Sgate([0, 1])
        bs01 = BSgate([0, 1])
        bs12 = BSgate([1, 2])

        circ1 = Circuit([vac]) >> s01
        circ2 = Circuit([bs01, bs12])

        assert circ1 >> circ2 == Circuit([vac, s01, bs01, bs12])

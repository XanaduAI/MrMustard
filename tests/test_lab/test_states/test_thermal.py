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

"""Tests for the ``Thermal`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import Thermal
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.triples import thermal_state_Abc


class TestThermal:
    r"""
    Tests for the ``Thermal`` class.
    """

    modes = [0, 1, 7]
    nbar = [3, 4, 5]

    @pytest.mark.parametrize("modes,nbar", zip(modes, nbar))
    def test_init(self, modes, nbar):
        state = Thermal(modes, nbar)

        assert state.name == "Thermal"
        assert state.modes == (modes,)

    @pytest.mark.parametrize("nbar", nbar)
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, nbar, batch_shape):
        nbar = math.broadcast_to(nbar, batch_shape)
        rep = Thermal(0, nbar).ansatz
        exp = PolyExpAnsatz(*thermal_state_Abc(nbar))
        assert rep == exp

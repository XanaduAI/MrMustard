# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ``SqueezedThermal`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import SqueezedThermal, SqueezedVacuum, Thermal
from mrmustard.parameters import Variable
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.triples import squeezed_thermal_state_Abc


class TestSqueezedThermal:
    r"""
    Tests for the ``SqueezedThermal`` class.
    """

    modes = [0, 1, 7]
    nbar = [3, 4, 5]
    r = [1, 2, 3]
    phi = [3, 4, 5]

    @pytest.mark.parametrize("modes,nbar,r,phi", zip(modes, nbar, r, phi))
    def test_init(self, modes, nbar, r, phi):
        state = SqueezedThermal(modes, nbar, r, phi)

        assert state.name == "SqueezedThermal"
        assert state.modes == (modes,)

    @pytest.mark.parametrize("modes,nbar,r,phi", zip(modes, nbar, r, phi))
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, modes, nbar, r, phi, batch_shape):
        nbar = math.broadcast_to(nbar, batch_shape)
        r = math.broadcast_to(r, batch_shape)
        phi = math.broadcast_to(phi, batch_shape)
        rep = SqueezedThermal(modes, nbar, r, phi).ansatz
        exp = PolyExpAnsatz(*squeezed_thermal_state_Abc(nbar, r, phi))
        assert rep == exp

    @pytest.mark.parametrize("modes,nbar", zip(modes, nbar))
    def test_reduce_to_thermal(self, modes, nbar):
        state = SqueezedThermal(modes, nbar, 0.0, 0.0)
        expected = Thermal(modes, nbar)
        assert state == expected

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_reduce_to_squeezed_vacuum(self, modes, r, phi):
        state = SqueezedThermal(modes, 0.0, r, phi)
        expected = SqueezedVacuum(modes, r, phi).dm()
        assert state == expected

    def test_trainable_parameters(self):
        state1 = SqueezedThermal(0, 1, 1, 1)
        nbar_var = Variable(1, "nbar", dtype=math.float64)
        r_var = Variable(1, "r", dtype=math.float64)
        phi_var = Variable(1, "phi", dtype=math.float64)
        state2 = SqueezedThermal(0, nbar=nbar_var, r=1, phi=1)
        state3 = SqueezedThermal(0, nbar=1, r=r_var, phi=1)
        state4 = SqueezedThermal(0, nbar=1, r=1, phi=phi_var)

        with pytest.raises(AttributeError):
            state1.parameters.nbar.value = 3

        state2.parameters.nbar.value = 2
        assert state2.parameters.nbar.value == 2

        state3.parameters.r.value = 2
        assert state3.parameters.r.value == 2

        state4.parameters.phi.value = 2
        assert state4.parameters.phi.value == 2

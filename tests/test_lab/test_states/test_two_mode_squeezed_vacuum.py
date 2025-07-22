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

"""Tests for the ``TwoModeSqueezedVacuum`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import TwoModeSqueezedVacuum, Vacuum
from mrmustard.lab.transformations import S2gate


class TestTwoModeSqueezedVacuum:
    r"""
    Tests for the ``TwoModeSqueezedVacuum`` class.
    """

    modes = [(0, 1), (1, 2), (1, 5)]
    r = [1, 2]
    phi = [3, 4, 1]

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_init(self, modes, r, phi):
        state = TwoModeSqueezedVacuum(modes, r, phi)

        assert state.name == "TwoModeSqueezedVacuum"
        assert state.modes == modes

    def test_trainable_parameters(self):
        state1 = TwoModeSqueezedVacuum((0, 1), 1, 1)
        state2 = TwoModeSqueezedVacuum((0, 1), 1, 1, r_trainable=True, r_bounds=(0, 2))
        state3 = TwoModeSqueezedVacuum((0, 1), 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.parameters.r.value = 3

        state2.parameters.r.value = 2
        assert state2.parameters.r.value == 2

        state3.parameters.phi.value = 2
        assert state3.parameters.phi.value == 2

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, modes, r, phi, batch_shape):
        r = math.broadcast_to(r, batch_shape, dtype=math.float64)
        phi = math.broadcast_to(phi, batch_shape, dtype=math.float64)
        rep = TwoModeSqueezedVacuum(modes, r, phi).ansatz
        exp = (Vacuum(modes) >> S2gate(modes, r, phi)).ansatz
        assert rep == exp

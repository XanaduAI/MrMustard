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

# pylint: disable=protected-access, unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import pytest

from mrmustard.lab_dev.states import Thermal
from mrmustard.physics.representations import Bargmann
from mrmustard.physics.triples import thermal_state_Abc


class TestThermal:
    r"""
    Tests for the ``Thermal`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    nbar = [[3], 4, [5, 6]]

    @pytest.mark.parametrize("modes,nbar", zip(modes, nbar))
    def test_init(self, modes, nbar):
        state = Thermal(modes, nbar)

        assert state.name == "Thermal"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_get_item(self):
        state = Thermal([0, 1], 3)
        assert state[0] == Thermal([0], 3)

    def test_init_error(self):
        with pytest.raises(ValueError, match="nbar"):
            Thermal(modes=[0, 1], nbar=[2, 3, 4])

    @pytest.mark.parametrize("nbar", [1, [2, 3], [4, 4]])
    def test_representation(self, nbar):
        rep = Thermal([0, 1], nbar).representation
        exp = Bargmann(*thermal_state_Abc([nbar, nbar] if isinstance(nbar, int) else nbar))
        assert rep == exp

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Thermal(modes=[0], nbar=[0.1, 0.2]).representation

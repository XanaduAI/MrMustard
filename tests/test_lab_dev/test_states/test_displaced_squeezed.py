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

"""Tests for the DisplacedSqueezed class."""

# pylint: disable=unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import pytest

from mrmustard.lab_dev.states import DisplacedSqueezed, Vacuum
from mrmustard.lab_dev.transformations import Dgate, Sgate


class TestDisplacedSqueezed:
    r"""
    Tests for the ``DisplacedSqueezed`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    x = [[1], 1, [1, 2]]
    y = [[3], [3, 4], [3, 4]]
    r = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,x,y,r,phi", zip(modes, x, y, r, phi))
    def test_init(self, modes, x, y, r, phi):
        state = DisplacedSqueezed(modes, x, y, r, phi)

        assert state.name == "DisplacedSqueezed"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="x"):
            DisplacedSqueezed(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="y"):
            DisplacedSqueezed(modes=[0, 1], x=1, y=[2, 3, 4])

    def test_trainable_parameters(self):
        state1 = DisplacedSqueezed([0], 1, 1)
        state2 = DisplacedSqueezed([0], 1, 1, x_trainable=True, x_bounds=(-2, 2))
        state3 = DisplacedSqueezed([0], 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.parameters.x.value = 3

        state2.parameters.x.value = 2
        assert state2.parameters.x.value == 2

        state3.parameters.y.value = 2
        assert state3.parameters.y.value == 2

    @pytest.mark.parametrize("modes,x,y,r,phi", zip(modes, x, y, r, phi))
    def test_representation(self, modes, x, y, r, phi):
        rep = DisplacedSqueezed(modes, x, y, r, phi).ansatz
        exp = (Vacuum(modes) >> Sgate(modes, r, phi) >> Dgate(modes, x, y)).ansatz
        assert rep == exp

    def test_representation_error(self):
        with pytest.raises(ValueError):
            DisplacedSqueezed(modes=[0], x=[0.1, 0.2]).ansatz

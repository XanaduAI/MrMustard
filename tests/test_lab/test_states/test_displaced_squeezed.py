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

import pytest

from mrmustard import math
from mrmustard.lab.states import DisplacedSqueezed, Vacuum
from mrmustard.lab.transformations import Dgate, Sgate


class TestDisplacedSqueezed:
    r"""
    Tests for the ``DisplacedSqueezed`` class.
    """

    modes = [0, 1, 7]
    x = [1, 2, 3]
    y = [3, 4, 5]
    r = [1, 2, 3]
    phi = [3, 4, 5]

    @pytest.mark.parametrize("modes,x,y,r,phi", zip(modes, x, y, r, phi))
    def test_init(self, modes, x, y, r, phi):
        state = DisplacedSqueezed(modes, x, y, r, phi)

        assert state.name == "DisplacedSqueezed"
        assert state.modes == (modes,)

    def test_trainable_parameters(self):
        state1 = DisplacedSqueezed(0, 1, 1)
        state2 = DisplacedSqueezed(0, 1, 1, x_trainable=True, x_bounds=(-2, 2))
        state3 = DisplacedSqueezed(0, 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.parameters.x.value = 3

        state2.parameters.x.value = 2
        assert state2.parameters.x.value == 2

        state3.parameters.y.value = 2
        assert state3.parameters.y.value == 2

    @pytest.mark.parametrize("modes,x,y,r,phi", zip(modes, x, y, r, phi))
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, modes, x, y, r, phi, batch_shape):
        x = math.broadcast_to(x, batch_shape)
        x, y, r, phi = math.broadcast_arrays(x, y, r, phi)
        rep = DisplacedSqueezed(modes, x, y, r, phi).ansatz
        exp = (
            Vacuum(modes) >> Sgate(modes, r, phi).contract(Dgate(modes, x, y), "zip")
        ).ansatz  # TODO: revisit rshift
        assert rep == exp

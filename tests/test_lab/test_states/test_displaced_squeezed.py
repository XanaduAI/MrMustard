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
    alpha = [1 + 3j, 2 + 4j, 3 + 5j]
    r = [1, 2, 3]
    phi = [3, 4, 5]

    @pytest.mark.parametrize("modes,alpha,r,phi", zip(modes, alpha, r, phi))
    def test_init(self, modes, alpha, r, phi):
        state = DisplacedSqueezed(modes, alpha, r, phi)

        assert state.name == "DisplacedSqueezed"
        assert state.modes == (modes,)

    def test_trainable_parameters(self):
        state1 = DisplacedSqueezed(0, 1 + 1j)
        state2 = DisplacedSqueezed(0, 1 + 1j, alpha_trainable=True, alpha_bounds=(0, 2))
        state3 = DisplacedSqueezed(0, 1 + 1j, r_trainable=True, r_bounds=(0, 2))

        with pytest.raises(AttributeError):
            state1.parameters.alpha.value = 3

        state2.parameters.alpha.value = 2
        assert state2.parameters.alpha.value == 2

        state3.parameters.r.value = 2
        assert state3.parameters.r.value == 2

    @pytest.mark.parametrize("modes,alpha,r,phi", zip(modes, alpha, r, phi))
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, modes, alpha, r, phi, batch_shape):
        alpha = math.broadcast_to(alpha, batch_shape)
        alpha, r, phi = math.broadcast_arrays(alpha, r, phi)
        rep = DisplacedSqueezed(modes, alpha, r, phi).ansatz
        exp = (
            Vacuum(modes) >> Sgate(modes, r, phi).contract(Dgate(modes, alpha), "zip")
        ).ansatz  # TODO: revisit rshift
        assert rep == exp

    @pytest.mark.parametrize("alpha", [10 + 10j, 18 + 18j, 25 + 25j])
    @pytest.mark.parametrize("r", [0, 1, 2, 3])
    def test_probability(self, alpha, r):
        """Tests that highly displaced squeezed states are properly normalized."""
        state = DisplacedSqueezed(mode=0, alpha=alpha, r=r)
        assert math.allclose(state.probability, 1)

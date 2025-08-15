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

    def test_stateless_construction(self):
        # Test that DisplacedSqueezed is now stateless - parameters only used for construction
        state = DisplacedSqueezed(0, 1 + 1j, 0.5, 0.3)

        # Should not have any parameter storage
        assert not hasattr(state, "alpha")
        assert not hasattr(state, "r")
        assert not hasattr(state, "phi")

        # But ansatz should be correctly constructed
        assert state.name == "DisplacedSqueezed"
        assert state.modes == (0,)

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

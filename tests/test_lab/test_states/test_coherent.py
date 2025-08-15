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

"""Tests for the Coherent class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import Coherent


class TestCoherent:
    r"""
    Tests for the ``Coherent`` class.
    """

    @pytest.mark.parametrize("modes", [0, 1, 7])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_init(self, modes, batch_shape):
        x = math.broadcast_to(1, batch_shape)
        y = math.broadcast_to(2, batch_shape)
        state = Coherent(modes, x + 1j * y)

        assert state.name == "Coherent"
        assert state.modes == (modes,)
        assert state.ansatz.batch_shape == batch_shape

    def test_stateless_construction(self):
        # Test that Coherent is now stateless - parameters only used for construction
        state = Coherent(0, 1 + 1j)

        # Should not have any parameter storage
        assert not hasattr(state, "alpha")

        # But ansatz should be correctly constructed
        assert state.name == "Coherent"
        assert state.modes == (0,)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, batch_shape):
        x = math.broadcast_to(0.1, batch_shape)
        y = math.broadcast_to(0.2, batch_shape)
        rep1 = Coherent(mode=0, alpha=x + 1j * y).ansatz
        assert math.allclose(rep1.A, math.zeros((1, 1)))
        assert math.allclose(rep1.b, [0.1 + 0.2j])
        assert math.allclose(rep1.c, 0.97530991)

        rep3 = Coherent(mode=1, alpha=x).ansatz
        assert math.allclose(rep3.A, math.zeros((1, 1)))
        assert math.allclose(rep3.b, [0.1])
        assert math.allclose(rep3.c, 0.9950124791926823)

    def test_linear_combinations(self):
        state1 = Coherent(0, 1 + 2j)
        state2 = Coherent(0, 2 + 3j)
        state3 = Coherent(0, 3 + 4j)

        lc = state1 + state2 - state3
        assert lc.ansatz.batch_size == 3

        assert (lc.contract(lc.dual)).ansatz.batch_size == 9
        assert (lc.contract(lc.dual, mode="zip")).ansatz.batch_size == 9

    def test_vacuum_shape(self):
        assert Coherent(0, 0.0).auto_shape() == (1,)

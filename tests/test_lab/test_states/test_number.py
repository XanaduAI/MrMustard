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

"""Tests for the ``Number`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import Number
from mrmustard.physics.fock_utils import fock_state
from mrmustard.physics.wires import ReprEnum


class TestNumber:
    r"""
    Tests for the ``Number`` class.
    """

    modes = [0, 1, 7]
    n = [3, 4, 5]
    cutoffs = [None, 5, 7]

    @pytest.mark.parametrize("modes,n,cutoffs", zip(modes, n, cutoffs))
    def test_init(self, modes, n, cutoffs):
        state = Number(modes, n, cutoffs)

        assert state.name == "N"
        assert state.modes == (modes,)
        assert all(isinstance(x, int) for x in state.manual_shape)

        batched_number = Number(modes, [n] * 3, cutoffs)
        assert batched_number.ansatz.batch_shape == (3,)

    @pytest.mark.requires_backend("numpy")
    def test_init_with_np_int(self):
        state = Number(math.int64(0), n=1)
        assert state.name == "N"
        assert state.modes == (0,)
        assert all(isinstance(x, int) for x in state.manual_shape)

    def test_auto_shape(self):
        n = 2
        state = Number(0, n=n).to_bargmann().dm()
        assert state.auto_shape() == (n + 1, n + 1)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 4)])
    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("cutoff", [None, 7])
    def test_representation(self, batch_shape, n, cutoff):
        n = math.broadcast_to(n, batch_shape)
        rep1 = Number(0, n, cutoff).ansatz.array
        exp1 = fock_state(n, cutoff)
        assert math.allclose(rep1, exp1)

        rep2 = Number(0, n, cutoff).to_fock().ansatz.array
        assert math.allclose(rep2, rep1)

    def test_scalar_bargmann(self):
        # meant to cover the case where we have derived variables and
        # no CV variables
        state = Number(0, 2).to_bargmann()
        contracted = state.contract(state.dual)
        assert contracted.ansatz.num_derived_vars == 2
        assert contracted.ansatz.num_CV_vars == 0
        assert math.allclose(contracted.ansatz.scalar, 1)

    def test_wires(self):
        """Test that the wires are correct."""
        state = Number(0, n=1)
        for w in state.wires.quantum:
            assert w.repr == ReprEnum.FOCK
            assert w.fock_cutoff == state.ansatz.core_shape[w.index]

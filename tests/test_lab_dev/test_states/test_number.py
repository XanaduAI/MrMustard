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

# pylint: disable=unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import pytest

from mrmustard import math, settings
from mrmustard.lab_dev.states import Number
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

    def test_auto_shape(self):
        # meant to cover the case where we have derived variables
        state = Number(0, 2).to_bargmann().dm()
        assert state.auto_shape() == (settings.AUTOSHAPE_MAX, settings.AUTOSHAPE_MAX)

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("cutoffs", [None, 4, 5])
    def test_representation(self, n, cutoffs):
        rep1 = Number(0, n, cutoffs).ansatz.array
        exp1 = fock_state(n, cutoffs)
        assert math.allclose(rep1, exp1)

        rep2 = Number(0, n, cutoffs).to_fock().ansatz.array
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
        for w in state.representation.wires:
            assert w.repr == ReprEnum.FOCK

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

# pylint: disable=protected-access, unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import pytest

from mrmustard import math
from mrmustard.physics.fock import fock_state
from mrmustard.lab_dev.states import Coherent, Number


class TestNumber:
    r"""
    Tests for the ``Number`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    n = [[3], 4, [5, 6]]
    cutoffs = [None, 5, [6, 7]]

    @pytest.mark.parametrize("modes,n,cutoffs", zip(modes, n, cutoffs))
    def test_init(self, modes, n, cutoffs):
        state = Number(modes, n, cutoffs)

        assert state.name == "N"
        assert state.modes == [modes] if not isinstance(modes, list) else sorted(modes)
        assert all(isinstance(x, int) for x in state.manual_shape)

    def test_init_error(self):
        with pytest.raises(ValueError, match="n"):
            Number(modes=[0, 1], n=[2, 3, 4])

        with pytest.raises(ValueError, match="cutoffs"):
            Number(modes=[0, 1], n=[2, 3], cutoffs=[4, 5, 6])

    @pytest.mark.parametrize("n", [2, [2, 3], [4, 4]])
    @pytest.mark.parametrize("cutoffs", [None, [4, 5], [5, 5]])
    def test_representation(self, n, cutoffs):
        rep1 = Number([0, 1], n, cutoffs).representation.array
        exp1 = fock_state((n,) * 2 if isinstance(n, int) else n, cutoffs)
        assert math.allclose(rep1, math.asnumpy(exp1).reshape(1, *exp1.shape))

        rep2 = Number([0, 1], n, cutoffs).to_fock().representation.array
        assert math.allclose(rep2, rep1)

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Coherent(modes=[0], x=[0.1, 0.2]).representation

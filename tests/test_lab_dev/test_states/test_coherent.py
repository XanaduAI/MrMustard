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

# pylint: disable=unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab_dev.states import Coherent


class TestCoherent:
    r"""
    Tests for the ``Coherent`` class.
    """

    modes = [0, 1, 7]
    x = [1, 2, 3]
    y = [3, 4, 5]

    @pytest.mark.parametrize("modes,x,y", zip(modes, x, y))
    def test_init(self, modes, x, y):
        state = Coherent(modes, x, y)

        assert state.name == "Coherent"
        assert state.modes == (modes,)

    def test_trainable_parameters(self):
        state1 = Coherent(0, 1, 1)
        state2 = Coherent(0, 1, 1, x_trainable=True, x_bounds=(-2, 2))
        state3 = Coherent(0, 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.parameters.x.value = 3

        state2.parameters.x.value = 2
        assert state2.parameters.x.value == 2

        state3.parameters.y.value = 2
        assert state3.parameters.y.value == 2

    def test_representation(self):
        rep1 = Coherent(mode=0, x=0.1, y=0.2).ansatz
        assert math.allclose(rep1.A, np.zeros((1, 1, 1)))
        assert math.allclose(rep1.b, [[0.1 + 0.2j]])
        assert math.allclose(rep1.c, [0.97530991])

        rep3 = Coherent(mode=1, x=0.1).ansatz
        assert math.allclose(rep3.A, np.zeros((1, 1, 1)))
        assert math.allclose(rep3.b, [[0.1]])
        assert math.allclose(rep3.c, [0.9950124791926823])

    def test_linear_combinations(self):
        state1 = Coherent(0, x=1, y=2)
        state2 = Coherent(0, x=2, y=3)
        state3 = Coherent(0, x=3, y=4)

        lc = state1 + state2 - state3
        assert lc.ansatz.batch_size == 3

        assert (lc @ lc.dual).ansatz.batch_size == 9
        settings.UNSAFE_ZIP_BATCH = True
        assert (lc @ lc.dual).ansatz.batch_size == 3  # not 9
        settings.UNSAFE_ZIP_BATCH = False

    def test_vacuum_shape(self):
        assert Coherent(0, x=0.0, y=0.0).auto_shape() == (1,)

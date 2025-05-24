# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ``CZgate`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import pytest

from mrmustard import math
from mrmustard.lab.states import Coherent
from mrmustard.lab.transformations import CZgate


class TestCZgate:
    r"""
    Tests for the ``CZgate`` class.
    """

    def test_init(self):
        "Tests the CZgate initialization."
        cz = CZgate((0, 1), 0.3)
        assert cz.modes == (0, 1)
        assert cz.name == "CZgate"
        assert cz.parameters.s.value == 0.3

    @pytest.mark.parametrize("s", [0.1, 0.2, 1.5])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_application(self, s, batch_shape):
        "Tests the application of CZgate"
        s_batch = math.broadcast_to(s, batch_shape)
        psi = Coherent(0, 1j) >> Coherent(1, 1) >> CZgate((0, 1), s_batch)
        _, d, _ = psi.phase_space(s=0)
        d_by_hand = math.astensor([0, math.sqrt(complex(2)), (1 + s) * math.sqrt(complex(2)), 0])
        assert d.shape[:-1] == batch_shape
        assert math.allclose(d, d_by_hand)

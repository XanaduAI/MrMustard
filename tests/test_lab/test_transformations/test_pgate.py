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

"""Tests for the ``Pgate`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import Vacuum
from mrmustard.lab.transformations import Pgate


class TestPgate:
    r"""
    Tests for the ``Pgate`` class.
    """

    def test_init(self):
        "Tests the Pgate initialization."
        up = Pgate(0, 0.3)
        assert up.modes == (0,)
        assert up.name == "Pgate"
        assert up.parameters.shearing.value == 0.3

    @pytest.mark.parametrize("s", [0.1, 0.5, 1])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_application(self, s, batch_shape):
        "Tests if Pgate is being applied correctly."
        s_batch = math.broadcast_to(s, batch_shape)
        up = Pgate(0, s_batch)
        rho = Vacuum(0) >> up
        cov, _, _ = rho.phase_space(s=0)
        temp = math.astensor([[1, 0], [s, 1]], dtype="complex128")
        assert math.allclose(cov, temp @ math.eye(2) @ temp.T / 2)

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

import re

import pytest

from mrmustard import math
from mrmustard.lab_dev.states import Coherent, Vacuum
from mrmustard.lab_dev.transformations import CZgate


class TestCZgate:
    r"""
    Tests for the ``CZgate`` class.
    """

    def test_init(self):
        "Tests the CZgate initialization."
        cz = CZgate([0, 1], 0.3)
        assert cz.modes == [0, 1]
        assert cz.name == "CZgate"
        assert cz.s.value == 0.3

        with pytest.raises(
            ValueError,
            match=re.escape(
                "The number of modes for a CZgate must be 2 (your input has 3 many modes)."
            ),
        ):
            CZgate([0, 1, 2], 0.2)

    @pytest.mark.parametrize("s", [0.1, 0.2, 1.5])
    def test_application(self, s):
        "Tests the application of CZgate"
        psi = Coherent([0], 0, 1) >> Coherent([1], 1, 0) >> CZgate([0, 1], 1)
        _, d, _ = psi.phase_space(s=0)
        psi = Coherent([0], 0, 1) >> Coherent([1], 1, 0) >> CZgate([0, 1], 1)
        d_by_hand = math.astensor([0, math.sqrt(complex(2)), (1 + s) * math.sqrt(complex(2)), 0])
        assert math.allclose(d[0], d_by_hand)

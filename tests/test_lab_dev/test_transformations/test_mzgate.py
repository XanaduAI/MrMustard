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

"""Tests for the ``MZgate`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard.lab_dev.states import Vacuum, Coherent
from mrmustard.lab_dev.transformations import MZgate


class TestMZgate:
    r"""
    Tests the Mach-Zehnder gate (MZgate)
    """

    def test_init(self):
        "Tests the initialization of an MZgate object"
        mz = MZgate([0, 1], 0.1, 0.2, internal=True)
        assert mz.modes == [0, 1]
        assert mz.phi_a.value == 0.1
        assert mz.phi_b.value == 0.2

        mz = MZgate([1, 2])
        assert mz.phi_a.value == 0
        assert mz.phi_b.value == 0

    @pytest.mark.parametrize("phi_a", [0, np.random.random(), np.pi / 2])
    def test_application(self, phi_a):
        "Tests the correctness of the application of an MZgate."
        rho = Vacuum([0]) >> Coherent([1], 1) >> MZgate([0, 1], phi_a, 0, internal=0)
        assert rho[0].ansatz == Coherent([1], x=0, y=1).dm().ansatz

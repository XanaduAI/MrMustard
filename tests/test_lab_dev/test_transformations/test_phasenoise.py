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

"""Tests for the ``PhaseNoise`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np

from mrmustard import math
from mrmustard.lab_dev.states import Ket, DM
from mrmustard.lab_dev.transformations import Dgate
from mrmustard.lab_dev.transformations import PhaseNoise


class TestGRN:
    r"""
    Tests for the ``PhaseNoise`` class.
    """

    def test_init(self):
        "Tests the GaussRandNoise initialization."
        ch = PhaseNoise([0, 1], 0.2)
        assert ch.name == "PhN"
        assert ch.phase_stdev.value == 0.2
        assert ch.modes == [0, 1]
        assert ch.ansatz == None

    def test_application(self):
        "Tests application of PhaseNoise on Ket and DM"
        psi = Ket.random([0, 1]) >> Dgate([0], 0.5, 0.5) >> PhaseNoise([0], 0.2)
        assert isinstance(psi, DM)
        assert psi.purity < 1

        rho = DM.random([0, 1]) >> Dgate([0], 0.5, 0.5) >> PhaseNoise([0], 0.2)
        assert isinstance(rho, DM)
        assert rho.purity < 1

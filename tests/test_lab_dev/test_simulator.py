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

"""Tests for the ``Simulator`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import settings
from mrmustard.lab_dev.circuits import Circuit
from mrmustard.lab_dev.simulator import Simulator
from mrmustard.lab_dev.states import Vacuum, Number
from mrmustard.lab_dev.transformations import Dgate, Attenuator

# original settings
autocutoff_max0 = settings.AUTOSHAPE_MAX


class TestSimulator:
    r"""
    Tests for the ``Circuit`` class.
    """

    @pytest.mark.parametrize(
        "path",
        [
            [],
            [(0, 1), (2, 3), (0, 2), (0, 4), (0, 5)],
            [(4, 5), (3, 4), (2, 3), (1, 2), (0, 1)],
        ],
    )
    def test_run(self, path):
        settings.AUTOSHAPE_MAX = 10

        vac12 = Vacuum([1, 2])
        d1 = Dgate([1], x=0.1, y=0.1)
        d2 = Dgate([2], x=0.1, y=0.2)
        d12 = Dgate([1, 2], x=0.1, y=[0.1, 0.2])
        a1 = Attenuator([1], transmissivity=0.8)
        n12 = Number([1, 2], n=1).dual

        circuit = Circuit([vac12, d1, d2, d12, a1, n12])
        circuit.path = path

        res = Simulator().run(circuit)
        exp = vac12 >> d1 >> d2 >> d12 >> a1 >> n12

        assert np.isclose(res, exp)

        settings.AUTOSHAPE_MAX = autocutoff_max0

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

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab.states import Coherent, Vacuum
from mrmustard.lab.transformations import MZgate


class TestMZgate:
    r"""
    Tests the Mach-Zehnder gate (MZgate)
    """

    def test_init(self):
        "Tests the initialization of an MZgate object"
        mz = MZgate((0, 1), 0.1, 0.2, internal=True)
        assert mz.modes == (0, 1)
        assert mz.parameters.phi_a.value == 0.1
        assert mz.parameters.phi_b.value == 0.2
        assert mz.name == "MZgate"

        mz = MZgate((1, 2))
        assert mz.parameters.phi_a.value == 0
        assert mz.parameters.phi_b.value == 0

    @pytest.mark.parametrize("phi_a", [0, settings.rng.random(), np.pi / 2])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_application(self, phi_a, batch_shape):
        "Tests the correctness of the application of an MZgate."
        phi_a_batch = math.broadcast_to(phi_a, batch_shape)
        rho = Vacuum(0) >> Coherent(1, 1) >> MZgate((0, 1), phi_a_batch, 0)

        rho0 = rho[0]
        assert rho0.ansatz.batch_shape == batch_shape
        assert rho0 == Coherent(0, x=0, y=1).dm()

        rho = Coherent(0, 1) >> Vacuum(1) >> MZgate((0, 1), phi_a_batch, phi_a_batch, internal=True)
        rho1 = rho[1]
        assert rho1.ansatz.batch_shape == batch_shape
        assert (
            rho1.ansatz
            == Coherent(1, x=-math.sin(complex(phi_a)), y=math.cos(complex(phi_a))).dm().ansatz
        )

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

import pytest

from mrmustard import math
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.lab.states import DM, Coherent, Ket, Number
from mrmustard.lab.transformations import Dgate, FockDamping, PhaseNoise


class TestPhaseNoise:
    r"""
    Tests for the ``PhaseNoise`` class.
    """

    def test_init(self):
        "Tests the PhaseNoise initialization."
        ch = PhaseNoise(0, 0.2)
        assert ch.name == "PhaseNoise"
        assert ch.parameters.phase_stdev.value == 0.2
        assert ch.modes == (0,)
        assert ch.ansatz is None

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_application(self, batch_shape):
        "Tests application of PhaseNoise on Ket and DM"
        x = math.broadcast_to(0.5, batch_shape)
        psi_1 = Ket.random((0, 1)) >> Dgate(0, x, 0.5) >> PhaseNoise(0, 0.2)
        assert isinstance(psi_1, DM)
        assert math.all(psi_1.purity < 1)

        rho = DM.random((0, 1)) >> Dgate(0, 0.5, 0.5) >> PhaseNoise(0, 0.2)
        assert isinstance(rho, DM)
        assert math.all(rho.purity < 1)

        psi_2 = Coherent(0, 2)
        after_noise_array = (psi_2 >> PhaseNoise(0, 10)).fock_array(10)
        assert math.allclose(
            math.diag_part(after_noise_array),
            math.diag_part(psi_2.dm().fock_array(10)),
        )  # the diagonal entries must remain unchanged
        mask = ~math.eye(after_noise_array.shape[0], dtype=bool)
        assert math.allclose(
            after_noise_array[mask],
            math.zeros_like(after_noise_array[mask]),
        )  # the off-diagonal entries must vanish

    @pytest.mark.parametrize("sigma", [0.2, 0.5, 0.7])
    def test_numeric(self, sigma):
        r"""
        A numeric example
        """
        psi = Number(0, 0) + Number(0, 1)
        phi = psi >> PhaseNoise(0, sigma)
        assert math.allclose(phi.fock_array(2)[0, 1], math.exp(-(complex(sigma) ** 2) / 2))

        # test with different order
        psi = Number(0, 1) + Number(0, 0)
        phi = psi >> PhaseNoise(0, sigma)
        assert math.allclose(phi.fock_array(2)[0, 1], math.exp(-(complex(sigma) ** 2) / 2))

    def test_check_adding_adjoint(self):
        r"""
        Tests if the PhaseNoise custum rrshift correcly adds the adjoint.
        """
        assert isinstance(FockDamping(0, 0.5) >> PhaseNoise(0, 0.2), CircuitComponent)

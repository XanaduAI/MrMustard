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

"""Tests for the ``CFT`` class."""

import numpy as np
from mrmustard import math, settings
from mrmustard.lab_dev import BtoPS, DisplacedSqueezed
from mrmustard.lab_dev.transformations.cft import CFT
from mrmustard.physics.wigner import wigner_discretized


class TestCFT:
    r"""
    Tests the CFT gate
    """

    def test_init(self):
        "tests the initialization of the CFT gate"
        cft = CFT([0])
        assert cft.name == "CFT"
        assert cft.modes == [0]

    def test_wigner_function(self):
        r"""
        Tests that the characteristic function is converted to the Wigner function
        for a single-mode squeezed state.
        """

        state = DisplacedSqueezed([0], r=0.5, phi=0.0, x=1.0, y=0.0)

        state2 = DisplacedSqueezed([0], r=0.5, phi=np.pi, x=0.0, y=1.0)
        dm = math.sum(state2.to_fock(100).dm().representation.array, axes=[0])
        vec = np.linspace(-5, 5, 100)
        wigner, _, _ = wigner_discretized(dm, vec, vec)

        Wigner = (state >> CFT([0]) >> BtoPS([0], s=0)).representation.ansatz
        X, Y = np.meshgrid(vec * np.sqrt(2), vec * np.sqrt(2))  # scaling to take care of HBAR
        Z = np.array([X - 1j * Y, X + 1j * Y]).transpose((1, 2, 0))
        assert math.allclose(
            2 * (np.real(Wigner(Z))), (np.real(wigner)), atol=1e-8
        )  # scaling to take care of HBAR

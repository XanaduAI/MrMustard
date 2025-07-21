# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BtoPS."""

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab import BtoPS, Dgate, Ket
from mrmustard.physics.wigner import wigner_discretized


class TestBtoPS:
    r"""
    Tests for the ``BtoPS`` class.
    """

    modes = [(0,), (1, 2), (7, 9)]
    s = [0, -0.9, 1]

    @pytest.mark.parametrize("hbar", [1.0, 2.0, 3.0])
    def test_application(self, hbar):
        state = Ket.random((0,), max_r=0.8) >> Dgate(0, x=2, y=0.1)

        dm = state.to_fock(100).dm().ansatz.array
        vec = np.linspace(-4.5, 4.5, 100)
        wigner, _, _ = wigner_discretized(dm, vec, vec)

        with settings(HBAR=hbar):
            Wigner = (state >> BtoPS(0, s=0)).ansatz

        X, Y = np.meshgrid(vec / np.sqrt(2 * settings.HBAR), vec / np.sqrt(2 * settings.HBAR))
        assert math.allclose(
            np.real(Wigner(X - 1j * Y, X + 1j * Y)) / (2 * settings.HBAR),
            np.real(wigner.T),
            atol=1e-6,
        )

    @pytest.mark.parametrize("modes,s", zip(modes, s))
    def test_init(self, modes, s):
        bw = BtoPS(modes, s)
        assert bw.name == "BtoPS"
        assert bw.modes == modes
        assert bw.parameters.s.value == s

    def test_fock_array(self):
        btops = BtoPS((0,), 0.5)
        with pytest.raises(NotImplementedError):
            btops.fock_array()

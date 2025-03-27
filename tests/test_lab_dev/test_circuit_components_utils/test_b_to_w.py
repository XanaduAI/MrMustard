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

"""Tests for BtoW."""

# pylint: disable=fixme, missing-function-docstring, pointless-statement

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab_dev import BtoW, Dgate, Ket
from mrmustard.physics.wigner import wigner_discretized


class TestBtoW:
    r"""
    Tests for the ``BtoW`` class.
    """

    modes = [(0,), (1, 2), (7, 9)]
    s = [0, -0.9, 1]

    @pytest.mark.parametrize("modes,s", zip(modes, s))
    def test_init(self, modes, s):
        bw = BtoW(modes, s)
        assert bw.name == "BtoW"
        assert bw.modes == modes
        assert bw.parameters.s.value == s

    @pytest.mark.parametrize("hbar", [1.0, 2.0, 3.0])
    def test_application(self, hbar):
        settings.HBAR = hbar
        state = Ket.random((0,)) >> Dgate(0, x=2, y=0.1)

        dm = math.sum(state.to_fock(100).dm().ansatz.array, axis=0)
        vec = np.linspace(-5, 5, 100)
        wigner, _, _ = wigner_discretized(dm, vec, vec)

        Wigner = (state >> BtoW([0], s=0)).ansatz
        X, Y = np.meshgrid(vec / np.sqrt(2 * settings.HBAR), vec / np.sqrt(2 * settings.HBAR))
        Z = np.array([X - 1j * Y, X + 1j * Y]).transpose((1, 2, 0))
        assert math.allclose(
            np.real(Wigner(Z)) / (2 * settings.HBAR),
            np.real(wigner.T),
            atol=1e-6,
        )

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

"""Tests for the ``S2gate`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.states import TwoModeSqueezedVacuum, Vacuum
from mrmustard.lab_dev.transformations import S2gate


class TestS2gate:
    r"""
    Tests for the ``S2gate`` class.
    """

    modes = [[0, 8], [1, 2], [7, 9]]
    r = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    def test_init(self):
        gate = S2gate([0, 1], 2, 1)

        assert gate.name == "S2gate"
        assert gate.modes == [0, 1]
        assert gate.r.value == 2
        assert gate.phi.value == 1

    def test_init_error(self):
        with pytest.raises(ValueError, match="Expected a pair"):
            S2gate([1, 2, 3])

    def test_representation(self):
        rep1 = S2gate([0, 1], 0.1, 0.2).ansatz
        tanhr = np.exp(1j * 0.2) * np.sinh(0.1) / np.cosh(0.1)
        sechr = 1 / np.cosh(0.1)

        A_exp = [
            [
                [0, -tanhr, sechr, 0],
                [-tanhr, 0, 0, sechr],
                [sechr, 0, 0, np.conj(tanhr)],
                [0, sechr, np.conj(tanhr), 0],
            ]
        ]
        assert math.allclose(rep1.A, A_exp)
        assert math.allclose(rep1.b, np.zeros((1, 4)))
        assert math.allclose(rep1.c, [1 / np.cosh(0.1)])

    def test_trainable_parameters(self):
        gate1 = S2gate([0, 1], 1, 1)
        gate2 = S2gate([0, 1], 1, 1, r_trainable=True, r_bounds=(0, 2))
        gate3 = S2gate([0, 1], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.r.value = 3

        gate2.r.value = 2
        assert gate2.r.value == 2

        gate3.phi.value = 2
        assert gate3.phi.value == 2

    def test_operation(self):
        rep1 = (Vacuum([0]) >> Vacuum([1]) >> S2gate(modes=[0, 1], r=1, phi=0.5)).ansatz
        rep2 = (TwoModeSqueezedVacuum(modes=[0, 1], r=1, phi=0.5)).ansatz

        assert math.allclose(rep1.A, rep2.A)
        assert math.allclose(rep1.b, rep2.b)
        assert math.allclose(rep1.c, rep2.c)

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

import pytest

from mrmustard import math
from mrmustard.lab.states import TwoModeSqueezedVacuum, Vacuum
from mrmustard.lab.transformations import S2gate


class TestS2gate:
    r"""
    Tests for the ``S2gate`` class.
    """

    def test_init(self):
        gate = S2gate((0, 1), 2, 1)

        assert gate.name == "S2gate"
        assert gate.modes == (0, 1)
        assert gate.parameters.r.value == 2
        assert gate.parameters.phi.value == 1

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, batch_shape):
        r = math.broadcast_to(0.1, batch_shape)
        phi = math.broadcast_to(0.2, batch_shape)

        rep1 = S2gate((0, 1), r, phi).ansatz
        tanhr = math.exp(1j * 0.2) * math.sinh(0.1) / math.cosh(0.1)
        sechr = math.astensor(1 / math.cosh(0.1), dtype=math.complex128)

        A_exp = [
            [0, tanhr, sechr, 0],
            [tanhr, 0, 0, sechr],
            [sechr, 0, 0, -math.conj(tanhr)],
            [0, sechr, -math.conj(tanhr), 0],
        ]
        assert math.allclose(rep1.A, A_exp)
        assert math.allclose(rep1.b, math.zeros((4,)))
        assert math.allclose(rep1.c, 1 / math.cosh(0.1))

    def test_trainable_parameters(self):
        gate1 = S2gate((0, 1), 1, 1)
        gate2 = S2gate((0, 1), 1, 1, r_trainable=True, r_bounds=(0, 2))
        gate3 = S2gate((0, 1), 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.parameters.r.value = 3

        gate2.parameters.r.value = 2
        assert gate2.parameters.r.value == 2

        gate3.parameters.phi.value = 2
        assert gate3.parameters.phi.value == 2

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_operation(self, batch_shape):
        r = math.broadcast_to(1.0, batch_shape)
        phi = math.broadcast_to(0.5, batch_shape)
        rep1 = (Vacuum((0, 1)) >> S2gate(modes=(0, 1), r=r, phi=phi)).ansatz
        rep2 = (TwoModeSqueezedVacuum(modes=(0, 1), r=r, phi=phi)).ansatz

        assert math.allclose(rep1.A, rep2.A)
        assert math.allclose(rep1.b, rep2.b)
        assert math.allclose(rep1.c, rep2.c)

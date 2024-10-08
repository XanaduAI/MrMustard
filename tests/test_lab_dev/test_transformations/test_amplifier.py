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

"""Tests for the ``Amplifier`` class."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.transformations import Attenuator, Amplifier
from mrmustard.lab_dev.states import Coherent


class TestAmplifier:
    r"""
    Tests for the ``Amplifier`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    gain = [[1.1], 1.1, [1.1, 1.2]]

    @pytest.mark.parametrize("modes,gain", zip(modes, gain))
    def test_init(self, modes, gain):
        gate = Amplifier(modes, gain)

        assert gate.name == "Amp"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="gain"):
            Amplifier(modes=[0, 1], gain=[1.2, 1.3, 1.4])

    def test_representation(self):
        rep1 = Amplifier(modes=[0], gain=1.1).ansatz
        g1 = 0.95346258
        g2 = 0.09090909
        assert math.allclose(
            rep1.A, [[[0, g1, g2, 0], [g1, 0, 0, 0], [g2, 0, 0, g1], [0, 0, g1, 0]]]
        )
        assert math.allclose(rep1.b, np.zeros((1, 4)))
        assert math.allclose(rep1.c, [0.90909090])

    def test_trainable_parameters(self):
        gate1 = Amplifier([0], 1.2)
        gate2 = Amplifier([0], 1.1, gain_trainable=True, gain_bounds=(1.0, 1.5))

        with pytest.raises(AttributeError):
            gate1.gain.value = 1.7

        gate2.gain.value = 1.5
        assert gate2.gain.value == 1.5

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Amplifier(modes=[0], gain=[1.1, 1.2]).ansatz

    def test_operation(self):
        amp_channel = Amplifier(modes=[0], gain=1.5)
        att_channel = Attenuator(modes=[0], transmissivity=0.7)
        operation = amp_channel >> att_channel

        assert math.allclose(
            operation.ansatz.A,
            [
                [
                    [0.0 + 0.0j, 0.75903339 + 0.0j, 0.25925926 + 0.0j, 0.0 + 0.0j],
                    [0.75903339 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.22222222 + 0.0j],
                    [0.25925926 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.75903339 + 0.0j],
                    [0.0 + 0.0j, 0.22222222 + 0.0j, 0.75903339 + 0.0j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(operation.ansatz.b, np.zeros((1, 4)))
        assert math.allclose(operation.ansatz.c, [0.74074074 + 0.0j])

    def test_circuit_identity(self):
        amp_channel = Amplifier(modes=[0], gain=2)
        att_channel = Attenuator(modes=[0], transmissivity=0.5)
        input_state = Coherent(modes=[0], x=0.5, y=0.7)

        assert math.allclose(
            (input_state >> amp_channel).ansatz.A,
            (input_state >> att_channel.dual).ansatz.A,
        )
        assert math.allclose(
            (input_state >> amp_channel).ansatz.b,
            (input_state >> att_channel.dual).ansatz.b,
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_swap_with_attenuator(self, n):
        def Amp(gain):
            return Amplifier([0], gain)

        def Att(transmissivity):
            return Attenuator([0], transmissivity)

        assert Amp((n + 1) / n) >> Att(n / (n + 1)) == Att((n + 1) / (n + 2)) >> Amp(
            (n + 2) / (n + 1)
        )

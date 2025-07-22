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

import pytest

from mrmustard import math
from mrmustard.lab.states import Coherent
from mrmustard.lab.transformations import Amplifier, Attenuator


class TestAmplifier:
    r"""
    Tests for the ``Amplifier`` class.
    """

    modes = [0, 1, 7]
    gain = [1.1, 1.2, 1.3]

    @pytest.mark.parametrize("modes,gain", zip(modes, gain))
    def test_init(self, modes, gain):
        gate = Amplifier(modes, gain)

        assert gate.name == "Amp~"
        assert gate.modes == (modes,)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, batch_shape):
        gain = math.broadcast_to(1.1, batch_shape)
        rep1 = Amplifier(mode=0, gain=gain).ansatz
        g1 = 0.95346258
        g2 = 0.09090909
        assert math.allclose(
            rep1.A,
            [[[0, g1, g2, 0], [g1, 0, 0, 0], [g2, 0, 0, g1], [0, 0, g1, 0]]],
        )
        assert math.allclose(rep1.b, math.zeros((1, 4)))
        assert math.allclose(rep1.c, 0.90909090)

    def test_trainable_parameters(self):
        gate1 = Amplifier(0, 1.2)
        gate2 = Amplifier(0, 1.1, gain_trainable=True, gain_bounds=(1.0, 1.5))

        with pytest.raises(AttributeError):
            gate1.parameters.gain.value = 1.7

        gate2.parameters.gain.value = 1.5
        assert gate2.parameters.gain.value == 1.5

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_operation(self, batch_shape):
        gain = math.broadcast_to(1.5, batch_shape)
        amp_channel = Amplifier(mode=0, gain=gain)
        att_channel = Attenuator(mode=0, transmissivity=0.7)
        operation = amp_channel >> att_channel

        assert math.allclose(
            operation.ansatz.A,
            [
                [0.0 + 0.0j, 0.75903339 + 0.0j, 0.25925926 + 0.0j, 0.0 + 0.0j],
                [0.75903339 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.22222222 + 0.0j],
                [0.25925926 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.75903339 + 0.0j],
                [0.0 + 0.0j, 0.22222222 + 0.0j, 0.75903339 + 0.0j, 0.0 + 0.0j],
            ],
        )
        assert math.allclose(operation.ansatz.b, math.zeros((4,)))
        assert math.allclose(operation.ansatz.c, 0.74074074 + 0.0j)

    def test_circuit_identity(self):
        amp_channel = Amplifier(mode=0, gain=2)
        att_channel = Attenuator(mode=0, transmissivity=0.5)
        input_state = Coherent(mode=0, x=0.5, y=0.7)

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
            return Amplifier(mode=0, gain=gain)

        def Att(transmissivity):
            return Attenuator(mode=0, transmissivity=transmissivity)

        assert Amp((n + 1) / n) >> Att(n / (n + 1)) == Att((n + 1) / (n + 2)) >> Amp(
            (n + 2) / (n + 1),
        )

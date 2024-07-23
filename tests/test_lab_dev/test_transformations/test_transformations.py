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

"""Tests for the transformation subpackage."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.transformations import (
    Attenuator,
    Amplifier,
    BSgate,
    Dgate,
    Rgate,
    Sgate,
    S2gate,
    Identity,
)
from mrmustard.lab_dev.states import Coherent, Vacuum, TwoModeSqueezedVacuum


class TestBSgate:
    r"""
    Tests for the ``BSgate`` class.
    """

    modes = [[0, 8], [1, 2], [9, 7]]
    theta = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    def test_init(self):
        gate = BSgate([0, 1], 2, 3)

        assert gate.name == "BSgate"
        assert gate.modes == [0, 1]
        assert gate.theta.value == 2
        assert gate.phi.value == 3

    def test_init_error(self):
        with pytest.raises(ValueError, match="Expected a pair"):
            BSgate([1, 2, 3])

    def test_representation(self):
        rep1 = BSgate([0, 1], 0.1, 0.2).representation
        A_exp = [
            [
                [0, 0, 0.99500417, -0.0978434 + 0.01983384j],
                [0.0, 0, 0.0978434 + 0.01983384j, 0.99500417],
                [0.99500417, 0.0978434 + 0.01983384j, 0, 0],
                [-0.0978434 + 0.01983384j, 0.99500417, 0, 0],
            ]
        ]
        assert math.allclose(rep1.A, A_exp)
        assert math.allclose(rep1.b, np.zeros((1, 4)))
        assert math.allclose(rep1.c, [1])

        rep2 = BSgate([0, 1], 0.1).representation
        A_exp = [
            [
                [0, 0, 9.95004165e-01, -9.98334166e-02],
                [0.0, 0, 9.98334166e-02, 9.95004165e-01],
                [9.95004165e-01, 9.98334166e-02, 0, 0],
                [-9.98334166e-02, 9.95004165e-01, 0, 0],
            ]
        ]
        assert math.allclose(rep2.A, A_exp)
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [1])

    def test_trainable_parameters(self):
        gate1 = BSgate([0, 1], 1, 1)
        gate2 = BSgate([0, 1], 1, 1, theta_trainable=True, theta_bounds=(-2, 2))
        gate3 = BSgate([0, 1], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.theta.value = 3

        gate2.theta.value = 2
        assert gate2.theta.value == 2

        gate3.phi.value = 2
        assert gate3.phi.value == 2


class TestDgate:
    r"""
    Tests for the ``Dgate`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    x = [[1], 1, [1, 2]]
    y = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,x,y", zip(modes, x, y))
    def test_init(self, modes, x, y):
        gate = Dgate(modes, x, y)

        assert gate.name == "Dgate"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="x"):
            Dgate(modes=[0, 1], x=[2, 3, 4])

        with pytest.raises(ValueError, match="y"):
            Dgate(modes=[0, 1], x=1, y=[2, 3, 4])

    def test_representation(self):
        rep1 = Dgate(modes=[0], x=0.1, y=0.1).representation
        assert math.allclose(rep1.A, [[[0, 1], [1, 0]]])
        assert math.allclose(rep1.b, [[0.1 + 0.1j, -0.1 + 0.1j]])
        assert math.allclose(rep1.c, [0.990049833749168])

        rep2 = Dgate(modes=[0, 1], x=[0.1, 0.2], y=0.1).representation
        assert math.allclose(
            rep2.A, [[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]]
        )
        assert math.allclose(
            rep2.b, [[0.1 + 0.1j, 0.2 + 0.1j, -0.1 + 0.1j, -0.2 + 0.1j]]
        )
        assert math.allclose(rep2.c, [0.9656054162575665])

        rep3 = Dgate(modes=[1, 8], x=[0.1, 0.2]).representation
        assert math.allclose(
            rep3.A, [[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]]
        )
        assert math.allclose(rep3.b, [[0.1, 0.2, -0.1, -0.2]])
        assert math.allclose(rep3.c, [0.9753099120283327])

    def test_trainable_parameters(self):
        gate1 = Dgate([0], 1, 1)
        gate2 = Dgate([0], 1, 1, x_trainable=True, x_bounds=(-2, 2))
        gate3 = Dgate([0], 1, 1, y_trainable=True, y_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.x.value = 3

        gate2.x.value = 2
        assert gate2.x.value == 2

        gate3.y.value = 2
        assert gate3.y.value == 2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Dgate(modes=[0], x=[0.1, 0.2]).representation


class TestRgate:
    r"""
    Tests for the ``Rgate`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    phis = [[1], 1, [1, 2]]

    @pytest.mark.parametrize("modes,phi", zip(modes, phis))
    def test_init(self, modes, phi):
        gate = Rgate(modes, phi)

        assert gate.name == "Rgate"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="phi"):
            Rgate(modes=[0, 1], phi=[2, 3, 4])

    def test_representation(self):
        rep1 = Rgate(modes=[0], phi=0.1).representation
        assert math.allclose(
            rep1.A,
            [
                [
                    [0.0 + 0.0j, 0.99500417 + 0.09983342j],
                    [0.99500417 + 0.09983342j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep1.b, np.zeros((1, 2)))
        assert math.allclose(rep1.c, [1.0 + 0.0j])

        rep2 = Rgate(modes=[0, 1], phi=[0.1, 0.3]).representation
        assert math.allclose(
            rep2.A,
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.95533649 + 0.29552021j],
                    [0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.95533649 + 0.29552021j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [1.0 + 0.0j])

        rep3 = Rgate(modes=[1], phi=0.1).representation
        assert math.allclose(
            rep3.A,
            [
                [
                    [0.0 + 0.0j, 0.99500417 + 0.09983342j],
                    [0.99500417 + 0.09983342j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep3.b, np.zeros((1, 2)))
        assert math.allclose(rep3.c, [1.0 + 0.0j])

    def test_trainable_parameters(self):
        gate1 = Rgate([0], 1)
        gate2 = Rgate([0], 1, True, (-2, 2))

        with pytest.raises(AttributeError):
            gate1.phi.value = 3

        gate2.phi.value = 2
        assert gate2.phi.value == 2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Rgate(modes=[0], phi=[0.1, 0.2]).representation


class TestSgate:
    r"""
    Tests for the ``Sgate`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    r = [[1], 1, [1, 2]]
    phi = [[3], [3, 4], [3, 4]]

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_init(self, modes, r, phi):
        gate = Sgate(modes, r, phi)

        assert gate.name == "Sgate"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="r"):
            Sgate(modes=[0, 1], r=[2, 3, 4])

        with pytest.raises(ValueError, match="phi"):
            Sgate(modes=[0, 1], r=1, phi=[2, 3, 4])

    def test_representation(self):
        rep1 = Sgate(modes=[0], r=0.1, phi=0.2).representation
        assert math.allclose(
            rep1.A,
            [
                [
                    [-0.09768127 - 1.98009738e-02j, 0.99502075],
                    [0.99502075, 0.09768127 - 0.01980097j],
                ]
            ],
        )
        assert math.allclose(rep1.b, np.zeros((1, 2)))
        assert math.allclose(rep1.c, [0.9975072676192522])

        rep2 = Sgate(modes=[0, 1], r=[0.1, 0.3], phi=0.2).representation
        assert math.allclose(
            rep2.A,
            [
                [
                    [-0.09768127 - 1.98009738e-02j, 0, 0.99502075, 0],
                    [0, -0.28550576 - 5.78748818e-02j, 0, 0.95662791],
                    [0.99502075, 0, 0.09768127 - 1.98009738e-02j, 0],
                    [0, 0.95662791, 0, 0.28550576 - 5.78748818e-02j],
                ]
            ],
        )
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [0.9756354961606032])

        rep3 = Sgate(modes=[1], r=0.1).representation
        assert math.allclose(
            rep3.A,
            [
                [
                    [-0.09966799 + 0.0j, 0.99502075 + 0.0j],
                    [0.99502075 + 0.0j, 0.09966799 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep3.b, np.zeros((1, 2)))
        assert math.allclose(rep3.c, [0.9975072676192522])

    def test_trainable_parameters(self):
        gate1 = Sgate([0], 1, 1)
        gate2 = Sgate([0], 1, 1, r_trainable=True, r_bounds=(-2, 2))
        gate3 = Sgate([0], 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            gate1.r.value = 3

        gate2.r.value = 2
        assert gate2.r.value == 2

        gate3.phi.value = 2
        assert gate3.phi.value == 2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Sgate(modes=[0], r=[0.1, 0.2]).representation


class TestIdentity:
    r"""
    Tests for the ``Identity`` class.
    """

    modes = [[0], [1, 2], [7, 9]]

    @pytest.mark.parametrize("modes", modes)
    def test_init(
        self,
        modes,
    ):
        gate = Identity(modes)

        assert gate.name == "Identity"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            Identity()

    def test_representation(self):
        rep1 = Identity(modes=[0]).representation
        assert math.allclose(
            rep1.A,
            [
                [
                    [0.0 + 0.0j, 1 + 0j],
                    [1 + 0j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep1.b, np.zeros((1, 2)))
        assert math.allclose(rep1.c, [1.0 + 0.0j])

        rep2 = Identity(modes=[0, 1]).representation
        assert math.allclose(
            rep2.A,
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [1.0 + 0.0j])


class TestS2gate:
    r"""
    Tests for the ``S2gate`` class.
    """

    modes = [[0, 8], [1, 2], [9, 7]]
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
        rep1 = S2gate([0, 1], 0.1, 0.2).representation
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
        rep1 = (
            Vacuum([0]) >> Vacuum([1]) >> S2gate(modes=[0, 1], r=1, phi=0.5)
        ).representation
        rep2 = (TwoModeSqueezedVacuum(modes=[0, 1], r=1, phi=0.5)).representation

        assert math.allclose(rep1.A, rep2.A)
        assert math.allclose(rep1.b, rep2.b)
        assert math.allclose(rep1.c, rep2.c)


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
        rep1 = Amplifier(modes=[0], gain=1.1).representation
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
            Amplifier(modes=[0], gain=[1.1, 1.2]).representation

    def test_operation(self):
        amp_channel = Amplifier(modes=[0], gain=1.5)
        att_channel = Attenuator(modes=[0], transmissivity=0.7)
        operation = amp_channel >> att_channel

        assert math.allclose(
            operation.representation.A,
            [
                [
                    [0.0 + 0.0j, 0.75903339 + 0.0j, 0.25925926 + 0.0j, 0.0 + 0.0j],
                    [0.75903339 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.22222222 + 0.0j],
                    [0.25925926 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.75903339 + 0.0j],
                    [0.0 + 0.0j, 0.22222222 + 0.0j, 0.75903339 + 0.0j, 0.0 + 0.0j],
                ]
            ],
        )
        assert math.allclose(operation.representation.b, np.zeros((1, 4)))
        assert math.allclose(operation.representation.c, [0.74074074 + 0.0j])

    def test_circuit_identity(self):
        amp_channel = Amplifier(modes=[0], gain=2)
        att_channel = Attenuator(modes=[0], transmissivity=0.5)
        input_state = Coherent(modes=[0], x=0.5, y=0.7)

        assert math.allclose(
            (input_state >> amp_channel).representation.A,
            (input_state >> att_channel.dual).representation.A,
        )
        assert math.allclose(
            (input_state >> amp_channel).representation.b,
            (input_state >> att_channel.dual).representation.b,
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


class TestAttenuator:
    r"""
    Tests for the ``Attenuator`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    transmissivity = [[0.1], 0.1, [0.1, 0.2]]

    @pytest.mark.parametrize("modes,transmissivity", zip(modes, transmissivity))
    def test_init(self, modes, transmissivity):
        gate = Attenuator(modes, transmissivity)

        assert gate.name == "Att"
        assert gate.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_init_error(self):
        with pytest.raises(ValueError, match="transmissivity"):
            Attenuator(modes=[0, 1], transmissivity=[0.2, 0.3, 0.4])

    def test_representation(self):
        rep1 = Attenuator(modes=[0], transmissivity=0.1).representation
        e = 0.31622777
        assert math.allclose(
            rep1.A, [[[0, e, 0, 0], [e, 0, 0, 0.9], [0, 0, 0, e], [0, 0.9, e, 0]]]
        )
        assert math.allclose(rep1.b, np.zeros((1, 4)))
        assert math.allclose(rep1.c, [1.0])

    def test_trainable_parameters(self):
        gate1 = Attenuator([0], 0.1)
        gate2 = Attenuator(
            [0], 0.1, transmissivity_trainable=True, transmissivity_bounds=(-0.2, 0.2)
        )

        with pytest.raises(AttributeError):
            gate1.transmissivity.value = 0.3

        gate2.transmissivity.value = 0.2
        assert gate2.transmissivity.value == 0.2

    def test_representation_error(self):
        with pytest.raises(ValueError):
            Attenuator(modes=[0], transmissivity=[0.1, 0.2]).representation

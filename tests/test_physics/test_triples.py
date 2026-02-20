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

"""Tests the Bargmann triples."""

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz

try:
    from equinox import EquinoxRuntimeError

    _validation_error = (ValueError, EquinoxRuntimeError)
except ImportError:
    _validation_error = (ValueError,)


class TestTriples:
    r"""
    Tests the Bargmann triples.
    """

    def test_amplifier_Abc(self):
        A1, b1, c1 = triples.amplifier_Abc(2)
        assert math.allclose(
            A1,
            [
                [0, 0.70710678, 0.5, 0],
                [0.70710678, 0, 0, 0],
                [0.5, 0, 0, 0.70710678],
                [0.0, 0, 0.70710678, 0],
            ],
        )
        assert math.allclose(b1, math.zeros(4))
        assert math.allclose(c1, 0.5)

        A2, b2, c2 = triples.amplifier_Abc([2, 1])
        assert math.allclose(
            A2,
            [
                [
                    [0, 0.70710678, 0.5, 0],
                    [0.70710678, 0, 0, 0],
                    [0.5, 0, 0, 0.70710678],
                    [0, 0, 0.70710678, 0],
                ],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            ],
        )
        assert math.allclose(b2, math.zeros((2, 4)))
        assert math.allclose(c2, [0.5, 1])

    def test_amplifier_Abc_error(self):
        with pytest.raises(_validation_error, match="smaller than"):
            triples.amplifier_Abc(0.1)

    def test_attenuator_Abc(self):
        A1, b1, c1 = triples.attenuator_Abc(0.1)
        e = 0.31622777
        assert math.allclose(A1, [[0, e, 0, 0], [e, 0, 0, 0.9], [0, 0, 0, e], [0, 0.9, e, 0]])
        assert math.allclose(b1, math.zeros(4))
        assert math.allclose(c1, 1.0)

        A2, b2, c2 = triples.attenuator_Abc([0.1, 1])
        e = 0.31622777
        assert math.allclose(
            A2,
            [
                [[0.0, e, 0.0, 0.0], [e, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, e], [0.0, 0.9, e, 0.0]],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            ],
        )
        assert math.allclose(b2, math.zeros((2, 4)))
        assert math.allclose(c2, math.ones(2))

    def test_attenuator_Abc_error(self):
        with pytest.raises(_validation_error, match="greater than `1`"):
            triples.attenuator_Abc(2)

        with pytest.raises(_validation_error, match="less than `0`"):
            triples.attenuator_Abc(-2)

    @pytest.mark.parametrize("eta", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_attenuator_kraus_Abc(self, eta):
        B = PolyExpAnsatz(*triples.attenuator_kraus_Abc(eta))
        Att = PolyExpAnsatz(*triples.attenuator_Abc(eta))
        assert B.contract(B, [[2], [2]]) == Att

    def test_beamsplitter_gate_Abc(self):
        A1, b1, c1 = triples.beamsplitter_gate_Abc(0.1, 0.2)
        A_exp = [
            [0, 0, 0.99500417, -0.0978434 + 0.01983384j],
            [0.0, 0, 0.0978434 + 0.01983384j, 0.99500417],
            [0.99500417, 0.0978434 + 0.01983384j, 0, 0],
            [-0.0978434 + 0.01983384j, 0.99500417, 0, 0],
        ]
        assert math.allclose(A1, A_exp)
        assert math.allclose(b1, math.zeros(4))
        assert math.allclose(c1, 1)

        A2, b2, c2 = triples.beamsplitter_gate_Abc(0.1, [0.2, 0.2])
        A = [
            [
                [0, 0, 0.99500417, -0.0978434 + 0.01983384j],
                [0, 0, 0.0978434 + 0.01983384j, 0.99500417],
                [0.99500417, 0.0978434 + 0.01983384j, 0, 0],
                [-0.0978434 + 0.01983384j, 0.99500417, 0, 0],
            ],
            [
                [0, 0, 0.99500417, -0.0978434 + 0.01983384j],
                [0, 0, 0.0978434 + 0.01983384j, 0.99500417],
                [0.99500417, 0.0978434 + 0.01983384j, 0, 0],
                [-0.0978434 + 0.01983384j, 0.99500417, 0, 0],
            ],
        ]

        assert math.allclose(A2, A)
        assert math.allclose(b2, math.zeros((2, 4)))
        assert math.allclose(c2, math.ones(2))

        A3, b3, c3 = triples.beamsplitter_gate_Abc(0.1)
        A_exp = [
            [0, 0, 9.95004165e-01, -9.98334166e-02],
            [0.0, 0, 9.98334166e-02, 9.95004165e-01],
            [9.95004165e-01, 9.98334166e-02, 0, 0],
            [-9.98334166e-02, 9.95004165e-01, 0, 0],
        ]
        assert math.allclose(A3, A_exp)
        assert math.allclose(b3, math.zeros(4))
        assert math.allclose(c3, 1)

    def test_coherent_state_Abc(self):
        A1, b1, c1 = triples.coherent_state_Abc(0.1 + 0.2j)
        assert math.allclose(A1, math.zeros((1, 1)))
        assert math.allclose(b1, [0.1 + 0.2j])
        assert math.allclose(c1, 0.97530991)

        A2, b2, c2 = triples.coherent_state_Abc([0.1 + 0.2j, 0.1 + 0.3j])
        assert math.allclose(A2, math.zeros((2, 1, 1)))
        assert math.allclose(b2, [[0.1 + 0.2j], [0.1 + 0.3j]])
        assert math.allclose(c2, [0.97530991 + 0.0j, 0.95122942 + 0.0j])

        A3, b3, c3 = triples.coherent_state_Abc(0.1)
        assert math.allclose(A3, math.zeros((1, 1)))
        assert math.allclose(b3, [0.1])
        assert math.allclose(c3, 0.9950124791926823)

    def test_displacement_gate_Abc(self):
        A1, b1, c1 = triples.displacement_gate_Abc(0.1 + 0.1j)
        assert math.allclose(A1, [[0, 1], [1, 0]])
        assert math.allclose(b1, [0.1 + 0.1j, -0.1 + 0.1j])
        assert math.allclose(c1, 0.990049833749168)

        A2, b2, c2 = triples.displacement_gate_Abc([0.1 + 0.1j, 0.2 + 0.1j])
        assert math.allclose(
            A2,
            [
                [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
            ],
        )
        assert math.allclose(b2, [[0.1 + 0.1j, -0.1 + 0.1j], [0.2 + 0.1j, -0.2 + 0.1j]])
        assert math.allclose(c2, [0.99004983 + 0.0j, 0.97530991 + 0.0j])

        A3, b3, c3 = triples.displacement_gate_Abc(alpha=[0.1, 0.2])
        assert math.allclose(
            A3,
            [
                [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]],
            ],
        )
        assert math.allclose(b3, [[0.1 + 0.0j, -0.1 + 0.0j], [0.2 + 0.0j, -0.2 + 0.0j]])
        assert math.allclose(c3, [0.99501248 + 0.0j, 0.98019867 + 0.0j])

    def test_displacement_gate_s_parametrized_Abc(self):
        A1, b1, c1 = triples.displacement_map_s_parametrized_Abc(s=0, n_modes=1)
        A1_correct = math.astensor([[0, -0.5, -1, 0], [-0.5, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 1, 0]])
        assert math.allclose(A1, A1_correct[[0, 3, 1, 2], :][:, [0, 3, 1, 2]])
        assert math.allclose(b1, math.zeros(4))
        assert math.allclose(c1, 1.0)

        A2, b2, c2 = triples.displacement_map_s_parametrized_Abc(s=1, n_modes=1)
        A2_correct = math.astensor([[0, 0, -1, 0], [0, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 1, 0]])
        assert math.allclose(A2, A2_correct[[0, 3, 1, 2], :][:, [0, 3, 1, 2]])
        assert math.allclose(b2, math.zeros(4))
        assert math.allclose(c2, 1.0)

        A3, b3, c3 = triples.displacement_map_s_parametrized_Abc(s=-1, n_modes=1)
        A3_correct = math.astensor([[0, -1, -1, 0], [-1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 1, 0]])
        assert math.allclose(A3, A3_correct[[0, 3, 1, 2], :][:, [0, 3, 1, 2]])
        assert math.allclose(b3, math.zeros(4))
        assert math.allclose(c3, 1.0)

    def test_displaced_squeezed_vacuum_state_Abc(self):
        A1, b1, c1 = triples.displaced_squeezed_vacuum_state_Abc(0.1 + 0.2j, 0.3, 0.4)
        assert math.allclose(A1, [[-0.26831668 - 0.11344247j]])
        assert math.allclose(b1, [0.14952016 + 0.15768091j])
        assert math.allclose(c1, 0.95557745 + 0.00675411j)

        A2, b2, c2 = triples.displaced_squeezed_vacuum_state_Abc(0.1 + 0.2j, 0.3, [0.4, 0.5])
        assert math.allclose(A2, [[[-0.26831668 - 0.11344247j]], [[-0.25565087 - 0.13966271j]]])
        assert math.allclose(b2, [[0.14952016 + 0.15768091j], [0.15349763 + 0.1628361j]])
        assert math.allclose(c2, [0.95557745 + 0.00675411j, 0.95489408 + 0.00688296j])

        A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc([0.1, 0.2])
        A3_correct, b3_correct, c3_correct = triples.coherent_state_Abc([0.1, 0.2])
        assert math.allclose(A3, A3_correct)
        assert math.allclose(b3, b3_correct)
        assert math.allclose(c3, c3_correct)

    def test_fock_damping_Abc(self):
        A1, b1, c1 = triples.fock_damping_Abc(0.5)
        assert math.allclose(
            A1,
            [
                [0, 0.60653065],
                [0.60653065, 0],
            ],
        )
        assert math.allclose(b1, math.zeros(2))
        assert math.allclose(c1, 1)

    def test_gaussian_random_noise_Abc(self):
        A, b, c = triples.gaussian_random_noise_Abc(np.eye(2))
        A_by_hand = math.astensor(
            [
                [0.0, 0.5, 0.5, 0.0],
                [0.5, 0.0, 0.0, 0.5],
                [0.5, 0.0, 0.0, 0.5],
                [0.0, 0.5, 0.5, 0.0],
            ],
        )
        b_by_hand = math.zeros(4)
        c_by_hand = 0.5

        assert math.allclose(A, A_by_hand)
        assert math.allclose(b, b_by_hand)
        assert math.allclose(c, c_by_hand)

    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    def test_homodyne_projector_Abc(self, batch_shape):
        bs_theta = math.broadcast_to(0.1, batch_shape)
        bs_phi = math.broadcast_to(0.2, batch_shape)
        homodyne_angle = math.broadcast_to(0.3, batch_shape)
        homodyne_outcome = math.broadcast_to(0.4, batch_shape)

        A_expected = math.astensor(
            [
                [0, 0.99500417 + 0.0j, -0.0978434 + 0.01983384j],
                [0.99500417 + 0.0j, -0.00976804 + 0.00198008j, -0.09149329 + 0.03868274j],
                [-0.0978434 + 0.01983384j, -0.09149329 + 0.03868274j, -0.81710973 + 0.55901484j],
            ]
        )
        b_expected = math.astensor([0, 0.05619217 - 0.00563802j, 0.53772008 - 0.16633631j])
        c_expected = math.astensor(0.6933762682841502 + 0j)

        A, b, c = triples.homodyne_projector_Abc(bs_theta, bs_phi, homodyne_angle, homodyne_outcome)
        assert math.allclose(A, math.broadcast_to(A_expected, (*batch_shape, 3, 3)))
        assert math.allclose(b, math.broadcast_to(b_expected, (*batch_shape, 3)))
        assert math.allclose(c, math.broadcast_to(c_expected, batch_shape))

    def test_identity_Abc(self):
        A1, b1, c1 = triples.identity_Abc(1)
        assert math.allclose(A1, [[0, 1], [1, 0]])
        assert math.allclose(b1, [0, 0])
        assert math.allclose(c1, 1)

        A2, b2, c2 = triples.identity_Abc(2)
        assert math.allclose(A2, [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        assert math.allclose(b2, [0, 0, 0, 0])
        assert math.allclose(c2, 1)

    def test_incompatible_shapes(self):
        if math.backend_name == "jax":
            match = "Incompatible shapes for broadcasting"
        else:
            match = "shape mismatch"

        with pytest.raises(ValueError, match=match):
            triples.squeezed_vacuum_state_Abc([1, 2], [3, 4, 5])

        with pytest.raises(ValueError, match=match):
            triples.displaced_squeezed_vacuum_state_Abc([1, 2], [3, 4, 5], 6)

    def test_rotation_gate_Abc(self):
        A1, b1, c1 = triples.rotation_gate_Abc(0.1)
        assert math.allclose(A1, [[0, 0.99500417 + 0.09983342j], [0.99500417 + 0.09983342j, 0]])
        assert math.allclose(b1, math.zeros(2))
        assert math.allclose(c1, 1.0)

        A2, b2, c2 = triples.rotation_gate_Abc([0.1, 0.2])
        g1 = 0.99500417 + 0.09983342j
        g2 = 0.98006658 + 0.19866933j
        assert math.allclose(A2, [[[0, g1], [g1, 0]], [[0, g2], [g2, 0]]])
        assert math.allclose(b2, math.zeros((2, 2)))
        assert math.allclose(c2, math.ones(2))

    def test_squeezed_vacuum_state_Abc(self):
        A1, b1, c1 = triples.squeezed_vacuum_state_Abc(0.1, 0.2)
        assert math.allclose(A1, [[-0.09768127 - 0.01980097j]])
        assert math.allclose(b1, math.zeros(1))
        assert math.allclose(c1, 0.9975072676192522)

        A2, b2, c2 = triples.squeezed_vacuum_state_Abc(0.1, [0.2, 0.3])
        assert math.allclose(A2, [[[-0.09768127 - 0.01980097j]], [[-0.09521647 - 0.02945391j]]])
        assert math.allclose(b2, math.zeros((2, 1)))
        assert math.allclose(c2, [0.99750727, 0.99750727])

        A3, b3, c3 = triples.squeezed_vacuum_state_Abc(0.1)
        assert math.allclose(A3, [[-0.09966799]])
        assert math.allclose(b3, math.zeros(1))
        assert math.allclose(c3, 0.9975072676192522)

    def test_squeezing_gate_Abc(self):
        A1, b1, c1 = triples.squeezing_gate_Abc(0.1, 0.2)
        assert math.allclose(
            A1,
            [
                [-0.09768127 - 1.98009738e-02j, 0.99502075],
                [0.99502075, 0.09768127 - 0.01980097j],
            ],
        )
        assert math.allclose(b1, math.zeros(2))
        assert math.allclose(c1, 0.9975072676192522)

        A2, b2, c2 = triples.squeezing_gate_Abc([0.1, 0.3], 0.2)
        assert math.allclose(
            A2,
            [
                [
                    [-0.09768127 - 0.01980097j, 0.99502075 + 0.0j],
                    [0.99502075 + 0.0j, 0.09768127 - 0.01980097j],
                ],
                [
                    [-0.28550576 - 0.05787488j, 0.95662791 + 0.0j],
                    [0.95662791 + 0.0j, 0.28550576 - 0.05787488j],
                ],
            ],
        )
        assert math.allclose(b2, math.zeros((2, 2)))
        assert math.allclose(c2, [0.99750727 + 0.0j, 0.97807357 + 0.0j])

        A3, b3, c3 = triples.squeezing_gate_Abc(0.1)
        assert math.allclose(
            A3,
            [
                [-0.09966799 + 0.0j, 0.99502075 + 0.0j],
                [0.99502075 + 0.0j, 0.09966799 + 0.0j],
            ],
        )
        assert math.allclose(b3, math.zeros(2))
        assert math.allclose(c3, 0.9975072676192522)

    def test_thermal_state_Abc(self):
        A1, b1, c1 = triples.thermal_state_Abc(0.1)
        assert math.allclose(A1, [[0, 0.09090909], [0.09090909, 0]])
        assert math.allclose(b1, math.zeros(2))
        assert math.allclose(c1, 1 / (0.1 + 1))

        A2, b2, c2 = triples.thermal_state_Abc([0.1, 0.2])
        assert math.allclose(
            A2,
            [
                [[0.0 + 0.0j, 0.09090909 + 0.0j], [0.09090909 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 0.16666667 + 0.0j], [0.16666667 + 0.0j, 0.0 + 0.0j]],
            ],
        )
        assert math.allclose(b2, math.zeros((2, 2)))
        assert math.allclose(c2, [1 / (0.1 + 1), 1 / (0.2 + 1)])

    @pytest.mark.parametrize("n_modes", [1, 3])
    def test_vacuum_state_Abc(self, n_modes):
        A, b, c = triples.vacuum_state_Abc(n_modes)

        assert math.allclose(A, math.zeros((n_modes, n_modes)))
        assert math.allclose(b, math.zeros(n_modes))
        assert math.allclose(c, 1.0)

    def test_XY_to_channel_Abc(self):
        # Creating an attenuator object and testing its Abc triple
        eta = settings.get_rng().random()
        X = math.sqrt(eta) * math.eye(2)
        Y = settings.HBAR / 2 * (1 - eta) * math.eye(2)

        A, b, c = triples.XY_to_channel_Abc(X, Y)

        A_by_hand = math.block(
            [
                math.astensor([[0, math.sqrt(eta), 0, 0]]),
                math.astensor([[math.sqrt(eta), 0, 0, 1 - eta]]),
                math.astensor([[0, 0, 0, math.sqrt(eta)]]),
                math.astensor([[0, 1 - eta, math.sqrt(eta), 0]]),
            ],
        )
        assert math.allclose(A, A_by_hand)
        assert math.allclose(b, math.zeros((4,)))
        assert b.shape == (4,)
        assert math.allclose(c, 1.0)

    def test_XY_to_channel_Abc_batched(self):
        eta = settings.get_rng().random(2)[:, None, None]
        X = math.sqrt(eta) * math.eye(2)[None, :, :]
        # Now X has shape (2, 2, 2)
        Y = settings.HBAR / 2 * (1 - eta) * math.eye(2)[None, :, :]

        A, b, c = triples.XY_to_channel_Abc(X, Y)

        A_by_hand = (
            math.sqrt(eta)
            * math.astensor(
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=math.complex128,
            )[None, :, :]
            + (1 - eta)
            * math.astensor(
                [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]],
                dtype=math.complex128,
            )[None, :, :]
        )

        assert math.allclose(A, A_by_hand)
        assert math.allclose(b, math.zeros((2, 4)))
        assert math.allclose(c, math.astensor([1.0, 1.0], dtype=math.complex128))
        assert c.shape == (2,)

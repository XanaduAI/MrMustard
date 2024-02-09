import numpy as np
import pytest

from mrmustard import math
from mrmustard.physics.triples import *

r"""
Tests the Bargmann triples.
"""


class TestTriples:
    r"""
    Tests the Bargmann triples.
    """

    def test_incompatible_shapes(self):
        match = "incompatible shape"

        with pytest.raises(ValueError, match=match):
            coherent_state_Abc([1, 2], [3, 4, 5])

        with pytest.raises(ValueError, match=match):
            coherent_state_Abc([1, 2], [3, 4, 5])

        with pytest.raises(ValueError, match=match):
            squeezed_vacuum_state_Abc([1, 2], [3, 4, 5])

        with pytest.raises(ValueError, match=match):
            displaced_squeezed_vacuum_state_Abc([1, 2], [3, 4, 5], 6, 7)

    @pytest.mark.parametrize("n_modes", [1, 3])
    def test_vacuum_state_Abc(self, n_modes):
        A, b, c = vacuum_state_Abc(n_modes)

        assert np.allclose(A, np.zeros((n_modes, n_modes)))
        assert np.allclose(b, np.zeros((n_modes)))
        assert np.allclose(c, 1.0)

    def test_coherent_state_Abc(self):
        A1, b1, c1 = coherent_state_Abc(0.1, 0.2)
        assert np.allclose(A1, 0)
        assert np.allclose(b1, 0.1 + 0.2j)
        assert np.allclose(c1, 0.97530991)

        A2, b2, c2 = coherent_state_Abc(0.1, [0.2, 0.3])
        assert np.allclose(A2, 0)
        assert np.allclose(b2, [0.1 + 0.2j, 0.1 + 0.3j])
        assert np.allclose(c2, 0.9277434863)

    def test_squeezed_vacuum_state_Abc(self):
        A1, b1, c1 = squeezed_vacuum_state_Abc(0.1, 0.2)
        assert np.allclose(A1, -0.09768127 - 0.01980097j)
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 0.9975072676192522)

        A2, b2, c2 = squeezed_vacuum_state_Abc(0.1, [0.2, 0.3])
        assert np.allclose(A2, [[-0.09768127 - 0.01980097j, 0], [0, -0.09521647 - 0.02945391j]])
        assert np.allclose(b2, 0)
        assert np.allclose(c2, 0.9950207489532265)

    def test_displaced_squeezed_vacuum_state_Abc(self):
        A1, b1, c1 = displaced_squeezed_vacuum_state_Abc(0.1, 0.2, 0.3, 0.4)
        assert np.allclose(A1, -0.26831668 - 0.11344247j)
        assert np.allclose(b1, [0.14952016 + 0.15768091j])
        assert np.allclose(c1, 0.95557745 + 0.00675411j)

        A2, b2, c2 = displaced_squeezed_vacuum_state_Abc(0.1, 0.2, 0.3, [0.4, 0.5])
        assert np.allclose(A2, [[-0.26831668 - 0.11344247j, 0], [0, -0.25565087 - 0.13966271j]])
        assert np.allclose(b2, [0.14952016 + 0.15768091j, 0.15349763 + 0.1628361j])
        assert np.allclose(c2, 0.912428762764038 + 0.013026652993991094j)

    def displaced_squeezed_vacuum_state_Abc(self):
        pass

    def test_thermal_state_Abc(self):
        A1, b1, c1 = thermal_state_Abc(0.1)
        assert np.allclose(A1, [[0, 0.09090909], [0.09090909, 0]])
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 1 / (0.1 + 1))

    def test_rotation_gate_Abc(self):
        A1, b1, c1 = rotation_gate_Abc(0.1)
        assert np.allclose(A1, [[0, 0.99500417 + 0.09983342j], [0.99500417 + 0.09983342j, 0]])
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 1)

    def test_rotation_gate_Abc(self):
        A1, b1, c1 = rotation_gate_Abc(0.1)
        assert np.allclose(A1, [[0, 0.99500417 + 0.09983342j], [0.99500417 + 0.09983342j, 0]])
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 1.0)

        A2, b2, c2 = rotation_gate_Abc([0.1, 0.2])
        g1 = 0.99500417 + 0.09983342j
        g2 = 0.98006658 + 0.19866933j
        assert np.allclose(A2, [[0, 0, g1, 0], [0, 0, 0, g2], [g1, 0, 0, 0], [0, g2, 0, 0]])
        assert np.allclose(b2, 0)
        assert np.allclose(c2, 1.0)

    def test_displacement_gate_Abc(self):
        A1, b1, c1 = displacement_gate_Abc([0.1, 0.2], 0.1)
        assert np.allclose(A1, [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        assert np.allclose(b1, [0.1 + 0.1j, 0.2 + 0.1j, -0.1 + 0.1j, -0.2 + 0.1j])
        assert np.allclose(c1, 0.9656054162575665)

    def test_squeezing_gate_Abc(self):
        A1, b1, c1 = squeezing_gate_Abc(0.1, 0.2)
        assert np.allclose(
            A1, [[0.09768127 + 0.01980097j, 0.99502075], [0.99502075, -0.09768127 + 0.01980097j]]
        )
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 0.9975072676192522)

    def test_beamsplitter_gate_Abc(self):
        A1, b1, c1 = beamsplitter_gate_Abc(0.1, 0.2)
        A_exp = [
            [0, 0, 0.99500417, -0.0978434 + 0.01983384j],
            [0.0, 0, 0.0978434 + 0.01983384j, 0.99500417],
            [0.99500417, 0.0978434 + 0.01983384j, 0, 0],
            [-0.0978434 + 0.01983384j, 0.99500417, 0, 0],
        ]
        assert np.allclose(A1, A_exp)
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 1)

    def test_attenuator_Abc(self):
        A1, b1, c1 = attenuator_Abc(0.1)
        e = 0.31622777
        assert np.allclose(A1, [[0, e, 0, 0], [e, 0, 0, 0.9], [0, 0, 0, e], [0, 0.9, e, 0]])
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 0.1)

    def test_attenuator_Abc_error(self):
        with pytest.raises(ValueError, match="in the interval"):
            attenuator_Abc(2)

        with pytest.raises(ValueError, match="in the interval"):
            attenuator_Abc(-2)

    def test_amplifier_Abc(self):
        pass

    def test_amplifier_Abc_error(self):
        with pytest.raises(ValueError, match="smaller than"):
            amplifier_Abc(0.1)

    @pytest.mark.parametrize("n_modes", [1, 2, 3])
    def test_fock_damping_Abc(self, n_modes):
        A1, b1, c1 = fock_damping_Abc(n_modes)
        assert np.allclose(A1, np.kron(math.astensor([[0, 1], [1, 0]]), math.eye(2 * n_modes)))
        assert np.allclose(b1, 0)
        assert np.allclose(c1, 1)

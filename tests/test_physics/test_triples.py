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

    @pytest.mark.parametrize("n_modes", [1, 3])
    def test_vacuum_state_ABC_triples(self, n_modes):
        A, b, c = vacuum_state_ABC_triples(n_modes)
        
        assert np.allclose(A, np.zeros((n_modes, n_modes)))
        assert np.allclose(b, np.zeros((n_modes)))
        assert np.allclose(c, 1.0)

    @pytest.mark.parametrize("x", [0, [1], [2, 3]])
    @pytest.mark.parametrize("y", [4, [5], [6, 7]])
    def test_coherent_state_ABC_triples(self, x, y):
        A, b, c = coherent_state_ABC_triples(x, y)

        x = math.atleast_1d(x, math.complex128)
        y = math.atleast_1d(y, math.complex128)
        n_modes = max(len(x), len(y))
        if len(x) == 1:
            x = math.tile(x, (n_modes,))
        if len(y) == 1:
            y = math.tile(y, (n_modes,))
        
        assert np.allclose(A, np.zeros((n_modes, n_modes)))
        assert np.allclose(b, x + 1j*y)
        assert np.allclose(c, np.exp(-0.5 * (x**2 + y**2)))
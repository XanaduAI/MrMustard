import numpy as np

from mrmustard import math
from mrmustard.lab import Attenuator, Dgate, Gaussian, Ggate
from mrmustard.lab_dev import Unitary, Vacuum
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
    norm_ket,
    trace_dm,
)


def test_wigner_to_bargmann_psi():
    """Test that the Bargmann representation of a ket is correct"""
    G = Gaussian(2) >> Dgate(0.1, 0.2)

    for x, y in zip(G.bargmann(), wigner_to_bargmann_psi(G.cov, G.means)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_rho():
    """Test that the Bargmann representation of a dm is correct"""
    G = Gaussian(2) >> Dgate(0.1, 0.2) >> Attenuator(0.9)

    for x, y in zip(G.bargmann(), wigner_to_bargmann_rho(G.cov, G.means)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_U():
    """Test that the Bargmann representation of a unitary is correct"""
    G = Ggate(2) >> Dgate(0.1, 0.2)
    X, _, d = G.XYd(allow_none=False)
    for x, y in zip(G.bargmann(), wigner_to_bargmann_U(X, d)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_choi():
    """Test that the Bargmann representation of a Choi matrix is correct"""
    G = Ggate(2) >> Dgate(0.1, 0.2) >> Attenuator(0.9)
    X, Y, d = G.XYd(allow_none=False)
    for x, y in zip(G.bargmann(), wigner_to_bargmann_Choi(X, Y, d)):
        assert np.allclose(x, y)


def test_bargmann_numpy_state():
    """Tests that the numpy option of the bargmann method of State works correctly"""
    state = Gaussian(1)
    assert all(isinstance(thing, np.ndarray) for thing in state.bargmann(numpy=True))


def test_bargmann_numpy_transformation():
    """Tests that the numpy option of the bargmann method of State works correctly"""
    transformation = Ggate(1)
    assert all(isinstance(thing, np.ndarray) for thing in transformation.bargmann(numpy=True))


def test_norm_ket():
    """Test that the norm of a ket is calculated correctly"""
    ket = Vacuum([0, 1]) >> Unitary.from_symplectic([0, 1], [0, 1], math.random_symplectic(2))
    A, b, c = [x[0] for x in ket.bargmann]
    assert np.isclose(norm_ket(A, b, c), ket.probability)


def test_trace_dm():
    """Test that the trace of a density matrix is calculated correctly"""
    ket = Vacuum([0, 1, 2, 3]) >> Unitary.from_symplectic(
        [0, 1, 2, 3], [0, 1, 2, 3], math.random_symplectic(4)
    )
    dm = ket[0, 1]
    A, b, c = [x[0] for x in dm.bargmann]
    assert np.allclose(trace_dm(A, b, c), dm.probability)

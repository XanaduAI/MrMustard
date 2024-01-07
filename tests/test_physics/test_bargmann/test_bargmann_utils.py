import numpy as np

from mrmustard.lab import Attenuator, Dgate, Gaussian, Ggate
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
    reorder_abc,
)


def test_reorder_abc():
    """Test that the reorder_abc function works correctly"""
    A = np.array([[1, 2], [2, 3]])
    b = np.array([4, 5])
    c = np.array(6)
    same = reorder_abc((A, b, c), (0, 1))
    assert all(np.allclose(x, y) for x, y in zip(same, (A, b, c)))
    flipped = reorder_abc((A, b, c), (1, 0))
    assert all(np.allclose(x, y) for x, y in zip(flipped, (A[[1, 0], :][:, [1, 0]], b[[1, 0]], c)))
    c = np.array([[6, 7], [8, 9]])
    flipped = reorder_abc((A, b, c), (1, 0))  #  test transposition of c
    assert all(
        np.allclose(x, y) for x, y in zip(flipped, (A[[1, 0], :][:, [1, 0]], b[[1, 0]], c.T))
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

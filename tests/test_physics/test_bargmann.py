import numpy as np

from mrmustard.lab import Attenuator, Dgate, Gaussian, Ggate
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
    Au2Symplectic,
    Symplectic2Au,
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


def test_Au2Symplectic():
    """Tests the Au -> symplectic code; we check two simple examples"""
    # Beam splitter example
    V = 1 / np.sqrt(2) * np.array([[1, 1], [-1, 1]])

    Au = np.block([[np.zeros_like(V), V], [V.T, np.zeros_like(V)]])
    S = Au2Symplectic(Au)
    S_by_hand = np.block([[V, np.zeros_like(V)], [np.zeros_like(V), np.conjugate(V)]])
    Transformation = (
        1 / np.sqrt(2) * np.block([[np.eye(2), np.eye(2)], [-1j * np.eye(2), 1j * np.eye(2)]])
    )
    S_by_hand = Transformation @ S_by_hand @ np.conjugate(Transformation.T)
    assert np.allclose(S, S_by_hand)

    # squeezing example
    r = 2
    Au = np.array([[-np.tanh(r), 1 / np.cosh(r)], [1 / np.cosh(r), np.tanh(r)]])
    S = Au2Symplectic(Au)
    S_by_hand = np.array([[np.cosh(r), -np.sinh(r)], [-np.sinh(r), np.cosh(r)]])
    Transformation = 1 / np.sqrt(2) * np.array([[1, 1], [-1j, 1j]])
    S_by_hand = Transformation @ S_by_hand @ np.conjugate(Transformation.T)
    assert np.allclose(S, S_by_hand)


def test_Symplectic2Au():
    """Tests the Symplectic -> Au code"""

    # here we consider the circuit of two-mode squeezing

    r = 2  # some squeezing parameter

    S_bs = np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, 1], [0, 0, -1, 1]]) / np.sqrt(2)

    S_sq = np.array(
        [
            [np.cosh(r), 0, -np.sinh(r), 0],
            [0, np.cosh(r), 0, np.sinh(r)],
            [-np.sinh(r), 0, np.cosh(r), 0],
            [0, np.sinh(r), 0, np.cosh(r)],
        ]
    )

    S = S_bs @ S_sq

    m = S.shape[-1]
    m = m // 2
    # the following lines of code transform the quadrature symplectic matrix to
    # the annihilation one
    Transformation = (
        1 / np.sqrt(2) * np.block([[np.eye(m), np.eye(m)], [-1j * np.eye(m), 1j * np.eye(m)]])
    )
    S = Transformation @ S @ np.conjugate(Transformation.T)

    A = Symplectic2Au(S)

    W = S_bs[:2, :2]
    T = np.diag([np.tanh(r), -np.tanh(r)])
    C = np.diag([np.cosh(r), np.cosh(r)])
    Sec = np.linalg.pinv(C)
    A_by_hand = np.block([[-W @ T @ W.T, W @ Sec], [Sec @ W.T, np.conjugate(T)]])

    assert np.allclose(A, A_by_hand)

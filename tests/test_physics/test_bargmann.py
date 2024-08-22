import numpy as np

from mrmustard import math
from mrmustard.lab import Attenuator, Dgate, Gaussian, Ggate
from mrmustard.lab_dev import Unitary, Vacuum, Channel
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
    norm_ket,
    trace_dm,
    au2Symplectic,
    symplectic2Au,
    XY_of_channel,
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

    ket = Vacuum([0, 1]) >> Unitary.from_symplectic([0, 1], math.random_symplectic(2))
    A, b, c = ket.bargmann_triple()
    assert np.isclose(norm_ket(A, b, c), ket.probability)


def test_trace_dm():
    """Test that the trace of a density matrix is calculated correctly"""
    ket = Vacuum([0, 1, 2, 3]) >> Unitary.from_symplectic([0, 1, 2, 3], math.random_symplectic(4))
    dm = ket[0, 1]
    A, b, c = dm.bargmann_triple()
    assert np.allclose(trace_dm(A, b, c), dm.probability)


def test_au2Symplectic():
    """Tests the Au -> symplectic code; we check two simple examples"""
    # Beam splitter example
    V = 1 / np.sqrt(2) * np.array([[1, 1], [-1, 1]])

    Au = np.block([[np.zeros_like(V), V], [np.transpose(V), np.zeros_like(V)]])
    S = au2Symplectic(Au)
    S_by_hand = np.block([[V, np.zeros_like(V)], [np.zeros_like(V), np.conjugate(V)]])
    transformation = (
        1
        / np.sqrt(2)
        * np.block(
            [
                [np.eye(2), np.eye(2)],
                [-1j * np.eye(2), 1j * np.eye(2)],
            ]
        )
    )
    S_by_hand = transformation @ S_by_hand @ np.conjugate(np.transpose(transformation))
    assert np.allclose(S, S_by_hand)

    # squeezing example
    r = 2
    Au = np.array([[-np.tanh(r), 1 / np.cosh(r)], [1 / np.cosh(r), np.tanh(r)]])
    S = au2Symplectic(Au)
    S_by_hand = np.array([[np.cosh(r), -np.sinh(r)], [-np.sinh(r), np.cosh(r)]])
    transformation = 1 / np.sqrt(2) * np.array([[1, 1], [-1j, 1j]])
    S_by_hand = transformation @ S_by_hand @ np.conjugate(np.transpose(transformation))
    assert np.allclose(S, S_by_hand)


def test_symplectic2Au():
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
    transformation = np.block(
        [[np.eye(m), np.eye(m)], [-1j * np.eye(m), 1j * np.eye(m)]]
    ) / np.sqrt(2)
    S = transformation @ S @ np.conjugate(np.transpose(transformation))
    A = symplectic2Au(S)

    W = S_bs[:2, :2]
    T = np.diag([np.tanh(r), -np.tanh(r)])
    C = np.diag([np.cosh(r), np.cosh(r)])
    Sec = np.linalg.pinv(C)
    A_by_hand = np.block(
        [[-W @ T @ np.transpose(W), W @ Sec], [Sec @ np.transpose(W), np.conjugate(T)]]
    )

    assert np.allclose(A, A_by_hand)


def test_XY_of_channel():
    r"""
    Tests the function X_of_channel.
    """

    X, Y = XY_of_channel(Channel.random([0]).representation.A[0])
    omega = np.array([[0, 1j], [-1j, 0]])
    channel_check = X @ omega @ X.T / 2 - omega / 2 + Y
    assert np.all([mu > 0 for mu in np.linalg.eigvals(channel_check)])

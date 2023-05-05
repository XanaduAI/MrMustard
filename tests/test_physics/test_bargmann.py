import numpy as np

from mrmustard.lab import Attenuator, Dgate, Gaussian, Ggate
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
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
    X, _, d = G.XYd(allow_none=True)
    for x, y in zip(G.bargmann(), wigner_to_bargmann_U(X, d)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_choi():
    """Test that the Bargmann representation of a Choi matrix is correct"""
    G = Ggate(2) >> Dgate(0.1, 0.2) >> Attenuator([0.9, 1.0])
    X, Y, d = G.XYd()
    for x, y in zip(G.bargmann(), wigner_to_bargmann_Choi(X, Y, d)):
        assert np.allclose(x, y)

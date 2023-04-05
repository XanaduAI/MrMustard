import numpy as np

from mrmustard.lab import Attenuator, Dgate, Gaussian
from mrmustard.physics.bargmann import wigner_to_bargmann_psi, wigner_to_bargmann_rho


def test_wigner_to_bargmann_psi():
    G = Gaussian(2) >> Dgate(0.1, 0.2)

    # Test that the Bargmann representation of a state is correct
    for x, y in zip(G.bargmann(), wigner_to_bargmann_psi(G.cov, G.means)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_rho():
    G = Gaussian(2) >> Dgate(0.1, 0.2) >> Attenuator(0.9)

    # Test that the Bargmann representation of a state is correct
    for x, y in zip(G.bargmann(), wigner_to_bargmann_rho(G.cov, G.means)):
        assert np.allclose(x, y)

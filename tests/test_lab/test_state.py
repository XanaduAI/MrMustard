import numpy as np

from mrmustard import math
from mrmustard.lab import Attenuator, Coherent, Gaussian, Vacuum, Dgate
from mrmustard.lab.abstract.state import mikkel_plot


def test_addition():
    """Test that addition of Gaussians is correct"""
    G0 = Gaussian(1, cutoffs=[10])
    G1 = Gaussian(1, cutoffs=[10])

    mixed = G0 + G1

    assert np.allclose(mixed.dm([10]), G0.dm([10]) + G1.dm([10]))


def test_multiplication_ket():
    """Test that multiplication of Gaussians is correct"""
    G = Gaussian(1, cutoffs=[10])

    scaled = 42.0 * G

    assert np.allclose(scaled.ket(G.shape), 42.0 * G.ket())


def test_multiplication_dm():
    """Test that multiplication of Gaussians is correct"""
    G = Gaussian(1, cutoffs=[10]) >> Attenuator(0.9)

    scaled = 42.0 * G

    assert np.allclose(scaled.dm(G.cutoffs), 42.0 * G.dm())


def test_division_ket():
    """Test that division of Gaussians is correct"""
    G = Gaussian(1, cutoffs=[10])

    scaled = G / 42.0

    assert np.allclose(scaled.ket([10]), G.ket([10]) / 42.0)


def test_division_dm():
    """Test that division of Gaussians is correct"""
    G = Gaussian(1, cutoffs=[10]) >> Attenuator(0.9)

    scaled = G / 42.0

    assert np.allclose(scaled.dm(G.cutoffs), G.dm() / 42.0)


def test_mikkel_plot():
    """Tests that mikkel plot returns figure and axes."""
    dm = Coherent().dm(cutoffs=[10])
    fig, axs = mikkel_plot(math.asnumpy(dm))

    assert fig is not None
    assert axs is not None


def test_rshit():
    """Tests that right shifr of the state will not change the state object. This was a bug (PR349)."""
    vac0 = Vacuum(1)
    vac0_cov_original = vac0.cov
    vac0_means_original = vac0.means
    d1 = Dgate(0.1, 0.1)
    _ = vac0 >> d1
    assert np.all(vac0_cov_original == vac0.cov)
    assert np.all(vac0_means_original == vac0.means)

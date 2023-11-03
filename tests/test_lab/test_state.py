import numpy as np

import mrmustard as mm
from mrmustard.lab.abstract.state import mikkel_plot


def test_addition():
    """Test that addition of Gaussians is correct"""
    G0 = mm.Gaussian(1, cutoffs=[10])
    G1 = mm.Gaussian(1, cutoffs=[10])

    mixed = G0 + G1

    assert np.allclose(mixed.dm([10]), G0.dm([10]) + G1.dm([10]))


def test_multiplication_ket():
    """Test that multiplication of Gaussians is correct"""
    G = mm.Gaussian(1, cutoffs=[10])

    scaled = 42.0 * G

    assert np.allclose(scaled.ket(G.shape), 42.0 * G.ket())


def test_multiplication_dm():
    """Test that multiplication of Gaussians is correct"""
    G = mm.Gaussian(1) >> mm.Attenuator(0.9)

    scaled = 42.0 * G

    assert np.allclose(scaled.dm(), 42.0 * G.dm())


def test_division_ket():
    """Test that division of Gaussians is correct"""
    G = mm.Gaussian(1, cutoffs=[10])

    scaled = G / 42.0

    assert np.allclose(scaled.ket([10]), G.ket([10]) / 42.0)


def test_division_dm():
    """Test that division of Gaussians is correct"""
    G = mm.Gaussian(1) >> mm.Attenuator(0.9)

    scaled = G / 42.0

    assert np.allclose(scaled.dm(G.cutoffs), G.dm() / 42.0)


def test_mikkel_plot():
    """Tests that mikkel plot returns figure and axes."""
    dm = mm.Coherent().dm(cutoffs=[10])
    fig, axs = mikkel_plot(mm.math.asnumpy(dm))

    assert fig is not None
    assert axs is not None

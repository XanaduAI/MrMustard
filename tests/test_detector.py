import pytest

import numpy as np
from scipy.stats import poisson

from mrmustard.tf import Dgate, Sgate, S2gate, Circuit, PNR


np.random.seed(137)


@pytest.mark.parametrize("alpha", np.random.rand(3) + 1j * np.random.rand(3))
@pytest.mark.parametrize("eta", [0, 0.3, 1.0])
@pytest.mark.parametrize("dc", [0, 0.2])
def test_detector_coherent_state(alpha, eta, dc):
    """Tests the correct Poisson statistics are generated when a coherent state hits an imperfect detector"""
    circ = Circuit(num_modes=1)
    cutoff = 20
    circ.add_gate(Dgate(modes=[0], x=alpha.real, y=alpha.imag))
    circ.add_detector(PNR(mode=0, quantum_efficiency=eta, dark_count_prob=dc))
    ps = circ.detection_probabilities(cutoffs=[cutoff])
    expected = poisson.pmf(k=np.arange(cutoff), mu=eta * np.abs(alpha) ** 2 + dc)
    assert np.allclose(ps, expected)


@pytest.mark.parametrize("r", np.random.rand(3))
@pytest.mark.parametrize("phi", 2 * np.pi * np.random.rand(3))
@pytest.mark.parametrize("eta", [0, 0.3, 1.0])
@pytest.mark.parametrize("dc", [0, 0.2])
def test_detector_squeezed_state(r, phi, eta, dc):
    """Tests the correct mean and variance are generated when a squeezed state hits an imperfect detector"""
    circ = Circuit(num_modes=1)
    circ.add_gate(Sgate(modes=[0], r=r, phi=phi))
    circ.add_detector(PNR(mode=0, quantum_efficiency=eta, dark_count_prob=dc))
    cutoff = 40
    ps = circ.detection_probabilities(cutoffs=[cutoff])
    assert np.allclose(np.sum(ps), 1.0, atol=1e-3)
    mean = np.arange(cutoff) @ ps.numpy()
    expected_mean = eta * np.sinh(r) ** 2 + dc
    assert np.allclose(mean, expected_mean, atol=1e-3)
    variance = np.arange(cutoff) ** 2 @ ps.numpy() - mean ** 2
    expected_variance = eta * np.sinh(r) ** 2 * (1 + eta * (1 + 2 * np.sinh(r) ** 2)) + dc
    assert np.allclose(variance, expected_variance, atol=1e-3)

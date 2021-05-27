import pytest

import numpy as np
from scipy.stats import poisson

from mrmustard.tf import Dgate, Sgate, S2gate, Circuit, PNRDetector, Vacuum


np.random.seed(137)


@pytest.mark.parametrize("alpha", np.random.rand(3) + 1j * np.random.rand(3))
@pytest.mark.parametrize("eta", [0, 0.3, 1.0])
@pytest.mark.parametrize("dc", [0, 0.2])
def test_detector_coherent_state(alpha, eta, dc):
    """Tests the correct Poisson statistics are generated when a coherent state hits an imperfect detector"""
    circ = Circuit()
    cutoff = 20
    circ.append(Dgate(modes=[0], x=alpha.real, y=alpha.imag))
    detector = PNRDetector(modes=[0], efficiency=eta, dark_counts=dc)
    ps = detector(circ(Vacuum(num_modes=1)), cutoffs=[cutoff])
    expected = poisson.pmf(k=np.arange(cutoff), mu=eta * np.abs(alpha) ** 2 + dc)
    assert np.allclose(ps, expected)


@pytest.mark.parametrize("r", np.random.rand(3))
@pytest.mark.parametrize("phi", 2 * np.pi * np.random.rand(3))
@pytest.mark.parametrize("eta", [0, 0.3, 1.0])
@pytest.mark.parametrize("dc", [0, 0.2])
def test_detector_squeezed_state(r, phi, eta, dc):
    """Tests the correct mean and variance are generated when a squeezed state hits an imperfect detector"""
    circ = Circuit()
    circ.append(Sgate(modes=[0], r=r, phi=phi))
    detector = PNRDetector(modes=[0], efficiency=eta, dark_counts=dc)
    cutoff = 40
    ps = detector(circ(Vacuum(num_modes=1)), cutoffs=[cutoff])

    assert np.allclose(np.sum(ps), 1.0, atol=1e-3)
    mean = np.arange(cutoff) @ ps.numpy()
    expected_mean = eta * np.sinh(r) ** 2 + dc
    assert np.allclose(mean, expected_mean, atol=1e-3)
    variance = np.arange(cutoff) ** 2 @ ps.numpy() - mean ** 2
    expected_variance = eta * np.sinh(r) ** 2 * (1 + eta * (1 + 2 * np.sinh(r) ** 2)) + dc
    assert np.allclose(variance, expected_variance, atol=1e-3)


@pytest.mark.parametrize("r", 0.5 * np.random.rand(3))
@pytest.mark.parametrize("phi", 2 * np.pi * np.random.rand(3))
@pytest.mark.parametrize("eta_s", [0, 0.3, 1.0])
@pytest.mark.parametrize("eta_i", [0, 0.3, 1.0])
@pytest.mark.parametrize("dc_s", [0, 0.2])
@pytest.mark.parametrize("dc_i", [0, 0.2])
def test_detector_two_mode_squeezed_state(r, phi, eta_s, eta_i, dc_s, dc_i):
    """Tests the correct mean and variance are generated when a two mode squeezed state hits an imperfect detector"""
    circ = Circuit()
    circ.append(S2gate(modes=[0, 1], r=r, phi=phi))
    detector = PNRDetector(modes=[0, 1], efficiency=[eta_s, eta_i], dark_counts=[dc_s, dc_i])
    cutoff = 30
    ps = detector(circ(Vacuum(num_modes=2)), cutoffs=[cutoff, cutoff])

    n = np.arange(cutoff)
    mean_s = np.sum(ps, axis=1) @ n
    n_s = eta_s * np.sinh(r) ** 2
    expected_mean_s = n_s + dc_s
    mean_i = np.sum(ps, axis=0) @ n
    n_i = eta_i * np.sinh(r) ** 2
    expected_mean_i = n_i + dc_i
    expected_mean_s = n_s + dc_s
    var_s = np.sum(ps, axis=1) @ n ** 2 - mean_s ** 2
    var_i = np.sum(ps, axis=0) @ n ** 2 - mean_i ** 2
    expected_var_s = n_s * (n_s + 1) + dc_s
    expected_var_i = n_i * (n_i + 1) + dc_i
    covar = n @ ps.numpy() @ n - mean_s * mean_i
    expected_covar = eta_s * eta_i * (np.sinh(r) * np.cosh(r)) ** 2
    assert np.allclose(mean_s, expected_mean_s, atol=1e-3)
    assert np.allclose(mean_i, expected_mean_i, atol=1e-3)
    assert np.allclose(var_s, expected_var_s, atol=1e-3)
    assert np.allclose(var_i, expected_var_i, atol=1e-3)
    assert np.allclose(covar, expected_covar, atol=1e-3)

from thewalrus.symplectic import two_mode_squeezing, squeezing, rotation, beam_splitter, expand
import numpy as np
from scipy.special import factorial
from mrmustard.tf import (
    Dgate,
    Sgate,
    LossChannel,
    BSgate,
    Ggate,
    Optimizer,
    Circuit,
    S2gate,
    Rgate,
    Vacuum,
)

import pytest


@pytest.mark.parametrize("n_mean", [0, 1, 2, 3])
@pytest.mark.parametrize("phi", 2 * np.pi * np.random.rand(4))
def test_two_mode_squeezing_fock(n_mean, phi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state
    Note that we use an extra - sign with respect to TMSV in https://en.wikipedia.org/wiki/Squeezed_coherent_state"""
    cutoff = 4
    circ = Circuit()
    r = np.arcsinh(np.sqrt(n_mean))
    circ.append(S2gate(modes=[0, 1], r=-r, phi=phi))
    amps = circ(Vacuum(num_modes=2)).ket(cutoffs=[cutoff, cutoff])
    diag = (1 / np.cosh(r)) * (-np.exp(1j * phi) * np.tanh(r)) ** np.arange(cutoff)
    expected = np.diag(diag)
    assert np.allclose(amps, expected)


@pytest.mark.parametrize("n_mean", [0, 1, 2, 3])
@pytest.mark.parametrize("phi", 2 * np.pi * np.random.rand(4))
@pytest.mark.parametrize("varphi", 2 * np.pi * np.random.rand(4))
def test_hong_ou_mandel(n_mean, phi, varphi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state"""
    cutoff = 2
    circ = Circuit()
    r = np.arcsinh(np.sqrt(n_mean))
    circ.append(S2gate(modes=[0, 1], r=-r, phi=phi))
    circ.append(S2gate(modes=[2, 3], r=-r, phi=phi))
    circ.append(BSgate(modes=[1, 2], theta=np.pi / 4, phi=varphi))
    amps = circ(Vacuum(num_modes=4)).ket(cutoffs=[cutoff, cutoff, cutoff, cutoff])
    assert np.allclose(amps[1, 1, 1, 1], 0.0)


@pytest.mark.parametrize("realpha", np.random.rand(4) - 0.5)
@pytest.mark.parametrize("imalpha", np.random.rand(4) - 0.5)
def test_coherent_state(realpha, imalpha):
    """Test that coherent states have the correct photon number statistics"""
    cutoff = 10
    circ = Circuit()
    circ.append(Dgate(modes=[0], x=realpha, y=imalpha))
    amps = circ(Vacuum(num_modes=1)).ket(cutoffs=[cutoff])
    alpha = realpha + 1j * imalpha
    expected = np.exp(-0.5 * np.abs(alpha) ** 2) * np.array(
        [alpha ** n / np.sqrt(factorial(n)) for n in range(cutoff)]
    )
    assert np.allclose(amps, expected)


@pytest.mark.parametrize("r", 2 * np.random.rand(4))
@pytest.mark.parametrize("phi", 2 * np.pi * np.random.rand(4))
def test_squeezed_state(r, phi):
    """Test that squeezed states have the correct photon number statistics
    Note that we use the same sign with respect to SMSV in https://en.wikipedia.org/wiki/Squeezed_coherent_state"""
    cutoff = 10
    circ = Circuit()
    circ.append(Sgate(modes=[0], r=r, phi=phi))
    amps = circ(Vacuum(num_modes=1)).ket(cutoffs=[cutoff])
    assert np.allclose(amps[1::2], 0.0)
    non_zero_amps = amps[0::2]
    len_non_zero = len(non_zero_amps)
    amp_pairs = (
        1
        / np.sqrt(np.cosh(r))
        * np.array(
            [
                (-np.exp(1j * phi) * np.tanh(r)) ** n
                * np.sqrt(factorial(2 * n))
                / (2 ** n * factorial(n))
                for n in range(len_non_zero)
            ]
        )
    )
    assert np.allclose(non_zero_amps, amp_pairs)


####
# The following tests currently fail
####
@pytest.mark.parametrize("n_mean", [0, 1, 2, 3])
@pytest.mark.parametrize("phi", 2 * np.pi * np.random.rand(4))
def test_two_mode_squeezing_fock_mean_and_covar(n_mean, phi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state"""
    circ = Circuit()
    r = np.arcsinh(np.sqrt(n_mean))
    circ.append(S2gate(modes=[0, 1], r=-r, phi=phi))
    state = circ(Vacuum(num_modes=2))
    meanN = state.number_means
    covN = state.number_cov
    expectedN = np.array([n_mean, n_mean])
    expectedCov = n_mean * (n_mean + 1) * np.ones([2, 2])
    assert np.allclose(meanN, expectedN)
    assert np.allclose(covN, expectedCov)

# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hypothesis import settings, given, strategies as st
import pytest

import numpy as np
from scipy.special import factorial
from thewalrus.quantum import total_photon_number_distribution
from mrmustard.lab import *
from mrmustard.physics.fock import dm_to_ket, ket_to_dm


# helper strategies
st_angle = st.floats(min_value=0, max_value=2 * np.pi)


@given(n_mean=st.floats(0, 3), phi=st_angle)
def test_two_mode_squeezing_fock(n_mean, phi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state
    Note that this is consistent with the Strawberryfields convention"""
    cutoff = 4
    r = np.arcsinh(np.sqrt(n_mean))
    circ = Circuit(ops=[S2gate(r=r, phi=phi)])
    amps = (Vacuum(num_modes=2) >> circ).ket(cutoffs=[cutoff, cutoff])
    diag = (1 / np.cosh(r)) * (np.exp(1j * phi) * np.tanh(r)) ** np.arange(cutoff)
    expected = np.diag(diag)
    assert np.allclose(amps, expected)


@given(n_mean=st.floats(0, 3), phi=st_angle, varphi=st_angle)
def test_hong_ou_mandel(n_mean, phi, varphi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state"""
    cutoff = 2
    r = np.arcsinh(np.sqrt(n_mean))
    ops = [
        S2gate(r=r, phi=phi)[0, 1],
        S2gate(r=r, phi=phi)[2, 3],
        BSgate(theta=np.pi / 4, phi=varphi)[1, 2],
    ]
    circ = Circuit(ops)
    amps = (Vacuum(4) >> circ).ket(cutoffs=[cutoff, cutoff, cutoff, cutoff])
    assert np.allclose(amps[1, 1, 1, 1], 0.0, atol=1e-6)


@given(alpha=st.complex_numbers(min_magnitude=0, max_magnitude=2))
def test_coherent_state(alpha):
    """Test that coherent states have the correct photon number statistics"""
    cutoff = 10
    amps = Coherent(x=alpha.real, y=alpha.imag).ket(cutoffs=[cutoff])
    expected = np.exp(-0.5 * np.abs(alpha) ** 2) * np.array(
        [alpha**n / np.sqrt(factorial(n)) for n in range(cutoff)]
    )
    assert np.allclose(amps, expected, atol=1e-6)


@given(r=st.floats(0, 2), phi=st_angle)
def test_squeezed_state(r, phi):
    """Test that squeezed states have the correct photon number statistics
    Note that we use the same sign with respect to SMSV in https://en.wikipedia.org/wiki/Squeezed_coherent_state"""
    cutoff = 10
    amps = SqueezedVacuum(r=r, phi=phi).ket(cutoffs=[cutoff])
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
                / (2**n * factorial(n))
                for n in range(len_non_zero)
            ]
        )
    )
    assert np.allclose(non_zero_amps, amp_pairs)


@given(n_mean=st.floats(0, 3), phi=st_angle)
def test_two_mode_squeezing_fock_mean_and_covar(n_mean, phi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state"""
    r = np.arcsinh(np.sqrt(n_mean))
    state = Vacuum(num_modes=2) >> S2gate(r=r, phi=phi)
    meanN = state.number_means
    covN = state.number_cov
    expectedN = np.array([n_mean, n_mean])
    expectedCov = n_mean * (n_mean + 1) * np.ones([2, 2])
    assert np.allclose(meanN, expectedN)
    assert np.allclose(covN, expectedCov)


@given(n_mean=st.floats(0, 2), phi=st_angle, eta=st.floats(min_value=0, max_value=1))
def test_lossy_squeezing(n_mean, phi, eta):
    """Tests the total photon number distribution of a lossy squeezed state"""
    r = np.arcsinh(np.sqrt(n_mean))
    cutoff = 40
    ps = (SqueezedVacuum(r=r, phi=phi) >> Attenuator(transmissivity=eta)).fock_probabilities(
        [cutoff]
    )
    expected = np.array([total_photon_number_distribution(n, 1, r, eta) for n in range(cutoff)])
    assert np.allclose(ps, expected, atol=1e-6)


@given(n_mean=st.floats(0, 2), phi=st_angle, eta_0=st.floats(0, 1), eta_1=st.floats(0, 1))
def test_lossy_two_mode_squeezing(n_mean, phi, eta_0, eta_1):
    """Tests the photon number distribution of a lossy two-mode squeezed state"""
    cutoff = 40
    n = np.arange(cutoff)
    L = Attenuator(transmissivity=[eta_0, eta_1])
    state = TMSV(r=np.arcsinh(np.sqrt(n_mean)), phi=phi) >> L
    ps0 = state.get_modes(0).fock_probabilities([cutoff])
    ps1 = state.get_modes(1).fock_probabilities([cutoff])
    mean_0 = np.sum(n * ps0)
    mean_1 = np.sum(n * ps1)
    assert np.allclose(mean_0, n_mean * eta_0, atol=1e-5)
    assert np.allclose(mean_1, n_mean * eta_1, atol=1e-5)


@given(num_modes=st.integers(1, 3))
def test_density_matrix(num_modes):
    """Tests the density matrix of a pure state is equal to |psi><psi|"""
    modes = list(range(num_modes))
    cutoffs = [num_modes + 1] * num_modes
    G = Ggate(num_modes=num_modes)
    L = Attenuator(transmissivity=1.0)
    rho_legit = (Vacuum(num_modes) >> G >> L[modes]).dm(cutoffs=cutoffs)
    rho_made = (Vacuum(num_modes) >> G).dm(cutoffs=cutoffs)
    # rho_legit = L[modes](G(Vacuum(num_modes))).dm(cutoffs=cutoffs)
    # rho_built = G(Vacuum(num_modes=num_modes)).dm(cutoffs=cutoffs)
    assert np.allclose(rho_legit, rho_made)


@pytest.mark.parametrize(
    "state",
    [
        Vacuum(num_modes=2),
        Fock(4),
        Coherent(x=0.1, y=-0.4, cutoffs=[15]),
        Gaussian(num_modes=2, cutoffs=[15]),
    ],
)
def test_dm_to_ket(state):
    """Tests pure state density matrix conversion to ket"""
    dm = state.dm()

    ket = dm_to_ket(dm)
    # check if ket is normalized
    assert np.allclose(np.linalg.norm(ket), 1)
    # check kets are equivalent
    assert np.allclose(ket, state.ket())

    dm_reconstructed = ket_to_dm(ket)
    # check ket leads to same dm
    assert np.allclose(dm, dm_reconstructed)


def test_dm_to_ket_error():
    """Test dm_to_ket raises an error when state is mixed"""
    state = Coherent(x=0.1, y=-0.4, cutoffs=[15]) >> Attenuator(0.5)

    with pytest.raises(ValueError):
        dm_to_ket(state)

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

"""Tests for the fock_utils.py file."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import factorial
from thewalrus.quantum import total_photon_number_distribution

from mrmustard import math, settings
from mrmustard.lab import (
    Attenuator,
    BSgate,
    Coherent,
    S2gate,
    SqueezedVacuum,
    TwoModeSqueezedVacuum,
    Vacuum,
)
from mrmustard.physics import fock_utils

# helper strategies
st_angle = st.floats(min_value=0, max_value=2 * np.pi)


@pytest.mark.parametrize("batch_shape", [(), (2,), (3, 4)])
def test_fock_state(batch_shape):
    r"""
    Tests that the `fock_state` method gives expected values.
    """
    batch_indices = np.indices(batch_shape)
    n = settings.rng.integers(0, 10, size=batch_shape)

    array1 = fock_utils.fock_state(n)
    assert array1.shape == (*batch_shape, math.max(n) + 1)
    assert math.all(array1[(*batch_indices, n)] == 1)

    array2 = fock_utils.fock_state(n, cutoff=math.max(n) + 1)
    assert math.allclose(array1, array2)

    array3 = fock_utils.fock_state(n, cutoff=15)
    assert array3.shape == (*batch_shape, 15)
    assert math.all(array3[(*batch_indices, n)] == 1)


def test_fock_state_error():
    r"""
    Tests that the `fock_state` method handles errors as expected.
    """
    n = [4, 5]

    with pytest.raises(ValueError, match="cannot be larger than"):
        fock_utils.fock_state(n, cutoff=4)


@given(n_mean=st.floats(0, 3), phi=st_angle)
def test_two_mode_squeezing_fock(n_mean, phi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state
    Note that this is consistent with the Strawberryfields convention"""
    cutoff = 4
    r = np.arcsinh(np.sqrt(n_mean))
    amps = (Vacuum((0, 1)) >> S2gate((0, 1), r=r, phi=phi)).fock_array([cutoff, cutoff])
    diag = (1 / np.cosh(r)) * (np.exp(1j * phi) * np.tanh(r)) ** np.arange(cutoff)
    expected = np.diag(diag)
    assert np.allclose(amps, expected)


@given(n_mean=st.floats(0, 3), phi=st_angle, varphi=st_angle)
def test_hong_ou_mandel(n_mean, phi, varphi):
    """Tests that perfect number correlations are obtained for a two-mode squeezed vacuum state"""
    cutoff = 2
    r = np.arcsinh(np.sqrt(n_mean))
    circ = (
        S2gate((0, 1), r=r, phi=phi)
        >> S2gate((2, 3), r=r, phi=phi)
        >> BSgate((1, 2), theta=np.pi / 4, phi=varphi)
    )
    amps = (Vacuum((0, 1, 2, 3)) >> circ).fock_array([cutoff, cutoff, cutoff, cutoff])
    assert np.allclose(amps[1, 1, 1, 1], 0.0, atol=1e-6)


@given(alpha=st.complex_numbers(min_magnitude=0, max_magnitude=2))
def test_coherent_state(alpha):
    """Test that coherent states have the correct photon number statistics"""
    cutoff = 10
    amps = Coherent(0, alpha.real, alpha.imag).fock_array([cutoff])
    expected = np.exp(-0.5 * np.abs(alpha) ** 2) * np.array(
        [alpha**n / np.sqrt(factorial(n)) for n in range(cutoff)],
    )
    assert np.allclose(amps, expected, atol=1e-6)


@given(r=st.floats(0, 2), phi=st_angle)
def test_squeezed_state(r, phi):
    """Test that squeezed states have the correct photon number statistics
    Note that we use the same sign with respect to SMSV in https://en.wikipedia.org/wiki/Squeezed_coherent_state
    """
    cutoff = 10
    amps = SqueezedVacuum(0, r=r, phi=phi).fock_array([cutoff])
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
            ],
        )
    )
    assert np.allclose(non_zero_amps, amp_pairs)


@given(n_mean=st.floats(0, 2), phi=st_angle, eta=st.floats(min_value=0, max_value=1))
def test_lossy_squeezing(n_mean, phi, eta):
    """Tests the total photon number distribution of a lossy squeezed state"""
    r = np.arcsinh(np.sqrt(n_mean))
    cutoff = 40
    ps = np.diag(
        (SqueezedVacuum(0, r=r, phi=phi) >> Attenuator(0, transmissivity=eta)).fock_array(cutoff),
    )
    expected = np.array([total_photon_number_distribution(n, 1, r, eta) for n in range(cutoff)])
    assert np.allclose(ps, expected, atol=1e-5)


@given(n_mean=st.floats(0, 2), phi=st_angle, eta_0=st.floats(0, 1), eta_1=st.floats(0, 1))
def test_lossy_two_mode_squeezing(n_mean, phi, eta_0, eta_1):
    """Tests the photon number distribution of a lossy two-mode squeezed state"""
    cutoff = 40
    n = np.arange(cutoff)
    L = Attenuator(0, transmissivity=eta_0) >> Attenuator(1, transmissivity=eta_1)
    state = TwoModeSqueezedVacuum((0, 1), r=np.arcsinh(np.sqrt(n_mean)), phi=phi) >> L
    ps0 = np.diag(state[0].fock_array(cutoff))
    ps1 = np.diag(state[1].fock_array(cutoff))
    mean_0 = np.sum(n * ps0)
    mean_1 = np.sum(n * ps1)
    assert np.allclose(mean_0, n_mean * eta_0, atol=1e-5)
    assert np.allclose(mean_1, n_mean * eta_1, atol=1e-5)

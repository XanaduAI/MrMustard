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

"""Tests for the fock.py file."""

# pylint: disable=pointless-statement

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import factorial
from thewalrus.quantum import total_photon_number_distribution

from mrmustard import math
from mrmustard.lab import (
    TMSV,
    Attenuator,
    BSgate,
    Circuit,
    Coherent,
    Fock,
    Gaussian,
    Ggate,
    S2gate,
    Sgate,
    SqueezedVacuum,
    State,
    Vacuum,
)
from mrmustard.math.lattice.strategies import displacement, grad_displacement
from mrmustard.physics import fock

# helper strategies
st_angle = st.floats(min_value=0, max_value=2 * np.pi)


def test_fock_state():
    n = [4, 5, 6]

    array1 = fock.fock_state(n)
    assert array1.shape == (5, 6, 7)
    assert array1[4, 5, 6] == 1

    array2 = fock.fock_state(n, cutoffs=10)
    assert array2.shape == (11, 11, 11)
    assert array2[4, 5, 6] == 1

    array3 = fock.fock_state(n, cutoffs=[5, 6, 7])
    assert array3.shape == (6, 7, 8)
    assert array3[4, 5, 6] == 1


def test_fock_state_error():
    n = [4, 5]

    with pytest.raises(ValueError):
        fock.fock_state(n, cutoffs=[5, 6, 7])

    with pytest.raises(ValueError):
        fock.fock_state(n, cutoffs=2)


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
    Note that we use the same sign with respect to SMSV in https://en.wikipedia.org/wiki/Squeezed_coherent_state
    """
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
    ps = (
        SqueezedVacuum(r=r, phi=phi) >> Attenuator(transmissivity=eta)
    ).fock_probabilities([cutoff])
    expected = np.array(
        [total_photon_number_distribution(n, 1, r, eta) for n in range(cutoff)]
    )
    assert np.allclose(ps, expected, atol=1e-5)


@given(
    n_mean=st.floats(0, 2), phi=st_angle, eta_0=st.floats(0, 1), eta_1=st.floats(0, 1)
)
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
    "state, kwargs",
    [
        (Vacuum, {"num_modes": 2}),
        (Fock, {"n": [4, 3], "modes": [0, 1]}),
        (Coherent, {"x": [0.1, 0.2], "y": [-0.4, 0.4], "cutoffs": [10, 10]}),
        (Gaussian, {"num_modes": 2, "cutoffs": [35, 35]}),
    ],
)
def test_dm_to_ket(state, kwargs):
    """Tests pure state density matrix conversion to ket"""
    state = state(**kwargs)
    dm = state.dm()
    ket = fock.dm_to_ket(dm)
    # check if ket is normalized
    assert np.allclose(np.linalg.norm(ket), 1, atol=1e-4)
    # check kets are equivalent
    assert np.allclose(ket, state.ket(), atol=1e-4)

    dm_reconstructed = fock.ket_to_dm(ket)
    # check ket leads to same dm
    assert np.allclose(dm, dm_reconstructed, atol=1e-15)


def test_dm_to_ket_error():
    """Test fock.dm_to_ket raises an error when state is mixed"""
    state = Coherent(x=0.1, y=-0.4, cutoffs=[15]) >> Attenuator(0.5)

    e = ValueError if math.backend_name == "tensorflow" else TypeError
    with pytest.raises(e):
        fock.dm_to_ket(state)


def test_fock_trace_mode1_dm():
    """tests that the Fock state is correctly traced out from mode 1 for mixed states"""
    state = Vacuum(2) >> Ggate(2) >> Attenuator([0.1, 0.1])
    from_gaussian = state.get_modes(0).dm([3])
    from_fock = State(dm=state.dm([3, 30])).get_modes(0).dm([3])
    assert np.allclose(from_gaussian, from_fock, atol=1e-5)


def test_fock_trace_mode0_dm():
    """tests that the Fock state is correctly traced out from mode 0 for mixed states"""
    state = Vacuum(2) >> Ggate(2) >> Attenuator([0.1, 0.1])
    from_gaussian = state.get_modes(1).dm([3])
    from_fock = State(dm=state.dm([30, 3])).get_modes(1).dm([3])
    assert np.allclose(from_gaussian, from_fock, atol=1e-5)


def test_fock_trace_mode1_ket():
    """tests that the Fock state is correctly traced out from mode 1 for pure states"""
    state = Vacuum(2) >> Sgate(r=[0.1, 0.2], phi=[0.3, 0.4])
    from_gaussian = state.get_modes(0).dm([3])
    from_fock = State(dm=state.dm([3, 30])).get_modes(0).dm([3])
    assert np.allclose(from_gaussian, from_fock, atol=1e-5)


def test_fock_trace_mode0_ket():
    """tests that the Fock state is correctly traced out from mode 0 for pure states"""
    state = Vacuum(2) >> Sgate(r=[0.1, 0.2], phi=[0.3, 0.4])
    from_gaussian = state.get_modes(1).dm([3])
    from_fock = State(dm=state.dm([30, 3])).get_modes(1).dm([3])
    assert np.allclose(from_gaussian, from_fock, atol=1e-5)


def test_fock_trace_function():
    """tests that the Fock state is correctly traced"""
    state = Vacuum(2) >> Ggate(2) >> Attenuator([0.1, 0.1])
    dm = state.dm([3, 20])
    dm_traced = fock.trace(dm, keep=[0])
    assert np.allclose(dm_traced, State(dm=dm).get_modes(0).dm(), atol=1e-5)


def test_dm_choi():
    """tests that choi op is correctly applied to a dm"""
    circ = Ggate(1) >> Attenuator([0.1])
    dm_out = fock.apply_choi_to_dm(circ.choi([10]), Vacuum(1).dm([10]), [0], [0])
    dm_expected = (Vacuum(1) >> circ).dm([10])
    assert np.allclose(dm_out, dm_expected, atol=1e-5)


def test_single_mode_choi_application_order():
    """Test dual operations output the correct mode ordering"""
    s = Attenuator(1.0) << State(
        dm=SqueezedVacuum(1.0, np.pi / 2).dm([40])
    )  # apply identity gate
    assert np.allclose(s.dm([10]), SqueezedVacuum(1.0, np.pi / 2).dm([10]))


def test_apply_kraus_to_ket_1mode():
    """Test that Kraus operators are applied to a ket on the correct indices"""
    ket = np.random.normal(size=(2, 3, 4))
    kraus = np.random.normal(size=(5, 3))
    ket_out = fock.apply_kraus_to_ket(kraus, ket, [1], [1])
    assert ket_out.shape == (2, 5, 4)


def test_apply_kraus_to_ket_1mode_with_argument_names():
    """Test that Kraus operators are applied to a ket on the correct indices with argument names"""
    ket = np.random.normal(size=(2, 3, 4))
    kraus = np.random.normal(size=(5, 3))
    ket_out = fock.apply_kraus_to_ket(
        kraus=kraus, ket=ket, kraus_in_modes=[1], kraus_out_modes=[1]
    )
    assert ket_out.shape == (2, 5, 4)


def test_apply_kraus_to_ket_2mode():
    """Test that Kraus operators are applied to a ket on the correct indices"""
    ket = np.random.normal(size=(2, 3, 4))
    kraus = np.random.normal(size=(5, 3, 4))
    ket_out = fock.apply_kraus_to_ket(kraus, ket, [1, 2], [1])
    assert ket_out.shape == (2, 5)


def test_apply_kraus_to_ket_2mode_2():
    """Test that Kraus operators are applied to a ket on the correct indices"""
    ket = np.random.normal(size=(2, 3))
    kraus = np.random.normal(size=(5, 4, 3))
    ket_out = fock.apply_kraus_to_ket(kraus, ket, [1], [1, 2])
    assert ket_out.shape == (2, 5, 4)


def test_apply_kraus_to_dm_1mode():
    """Test that Kraus operators are applied to a dm on the correct indices"""
    dm = np.random.normal(size=(2, 3, 2, 3))
    kraus = np.random.normal(size=(5, 3))
    dm_out = fock.apply_kraus_to_dm(kraus, dm, [1], [1])
    assert dm_out.shape == (2, 5, 2, 5)


def test_apply_kraus_to_dm_1mode_with_argument_names():
    """Test that Kraus operators are applied to a dm on the correct indices with argument names"""
    dm = np.random.normal(size=(2, 3, 2, 3))
    kraus = np.random.normal(size=(5, 3))
    dm_out = fock.apply_kraus_to_dm(
        kraus=kraus, dm=dm, kraus_in_modes=[1], kraus_out_modes=[1]
    )
    assert dm_out.shape == (2, 5, 2, 5)


def test_apply_kraus_to_dm_2mode():
    """Test that Kraus operators are applied to a dm on the correct indices"""
    dm = np.random.normal(size=(2, 3, 4, 2, 3, 4))
    kraus = np.random.normal(size=(5, 3, 4))
    dm_out = fock.apply_kraus_to_dm(kraus, dm, [1, 2], [1])
    assert dm_out.shape == (2, 5, 2, 5)


def test_apply_kraus_to_dm_2mode_2():
    """Test that Kraus operators are applied to a dm on the correct indices"""
    dm = np.random.normal(size=(2, 3, 4, 2, 3, 4))
    kraus = np.random.normal(size=(5, 6, 3))
    dm_out = fock.apply_kraus_to_dm(kraus, dm, [1], [3, 1])
    assert dm_out.shape == (2, 6, 4, 5, 2, 6, 4, 5)


def test_apply_choi_to_ket_1mode():
    """Test that choi operators are applied to a ket on the correct indices"""
    ket = np.random.normal(size=(3, 5))
    choi = np.random.normal(size=(4, 3, 4, 3))  # [out_l, in_l, out_r, in_r]
    ket_out = fock.apply_choi_to_ket(choi, ket, [0], [0])
    assert ket_out.shape == (4, 5, 4, 5)


def test_apply_choi_to_ket_1mode_with_argument_names():
    """Test that choi operators are applied to a ket on the correct indices with argument names"""
    ket = np.random.normal(size=(3, 5))
    choi = np.random.normal(size=(4, 3, 4, 3))  # [out_l, in_l, out_r, in_r]
    ket_out = fock.apply_choi_to_ket(
        choi=choi, ket=ket, choi_in_modes=[0], choi_out_modes=[0]
    )
    assert ket_out.shape == (4, 5, 4, 5)


def test_apply_choi_to_ket_2mode():
    """Test that choi operators are applied to a ket on the correct indices"""
    ket = np.random.normal(size=(3, 5))
    choi = np.random.normal(size=(2, 3, 5, 2, 3, 5))  # [out_l, in_l, out_r, in_r]
    ket_out = fock.apply_choi_to_ket(choi, ket, [0, 1], [0])
    assert ket_out.shape == (2, 2)


def test_apply_choi_to_dm_1mode():
    """Test that choi operators are applied to a dm on the correct indices"""
    dm = np.random.normal(size=(3, 5, 3, 5))
    choi = np.random.normal(size=(4, 3, 4, 3))  # [out_l, in_l, out_r, in_r]
    dm_out = fock.apply_choi_to_dm(choi, dm, [0], [0])
    assert dm_out.shape == (4, 5, 4, 5)


def test_apply_choi_to_dm_1mode_with_argument_names():
    """Test that choi operators are applied to a dm on the correct indices with argument names"""
    dm = np.random.normal(size=(3, 5, 3, 5))
    choi = np.random.normal(size=(4, 3, 4, 3))  # [out_l, in_l, out_r, in_r]
    dm_out = fock.apply_choi_to_dm(
        choi=choi, dm=dm, choi_in_modes=[0], choi_out_modes=[0]
    )
    assert dm_out.shape == (4, 5, 4, 5)


def test_apply_choi_to_dm_2mode():
    """Test that choi operators are applied to a dm on the correct indices"""
    dm = np.random.normal(size=(4, 5, 4, 5))
    choi = np.random.normal(
        size=(2, 3, 5, 2, 3, 5)
    )  # [out_l_1,2, in_l_1, out_r_1,2, in_r_1]
    dm_out = fock.apply_choi_to_dm(choi, dm, [1], [1, 2])
    assert dm_out.shape == (4, 2, 3, 4, 2, 3)


def test_displacement_grad():
    """Tests the value of the analytic gradient for the Dgate against finite differences"""
    cutoff = 4
    r = 2.0
    theta = np.pi / 8
    T = displacement((cutoff, cutoff), r * np.exp(1j * theta))
    Dr, Dtheta = grad_displacement(T, r, theta)

    dr = 0.001
    dtheta = 0.001
    Drp = displacement((cutoff, cutoff), (r + dr) * np.exp(1j * theta))
    Drm = displacement((cutoff, cutoff), (r - dr) * np.exp(1j * theta))
    Dthetap = displacement((cutoff, cutoff), r * np.exp(1j * (theta + dtheta)))
    Dthetam = displacement((cutoff, cutoff), r * np.exp(1j * (theta - dtheta)))
    Drapprox = (Drp - Drm) / (2 * dr)
    Dthetaapprox = (Dthetap - Dthetam) / (2 * dtheta)
    assert np.allclose(Dr, Drapprox, atol=1e-5, rtol=0)
    assert np.allclose(Dtheta, Dthetaapprox, atol=1e-5, rtol=0)


def test_displacement_values():
    """Tests the correct construction of the single mode displacement operation"""
    cutoff = 5
    alpha = 0.3 + 0.5 * 1j
    # This data is obtained by using qutip
    # np.array(displace(40,alpha).data.todense())[0:5,0:5]
    expected = np.array(
        [
            [
                0.84366482 + 0.00000000e00j,
                -0.25309944 + 4.21832408e-01j,
                -0.09544978 - 1.78968334e-01j,
                0.06819609 + 3.44424719e-03j,
                -0.01109048 + 1.65323865e-02j,
            ],
            [
                0.25309944 + 4.21832408e-01j,
                0.55681878 + 0.00000000e00j,
                -0.29708743 + 4.95145724e-01j,
                -0.14658716 - 2.74850926e-01j,
                0.12479885 + 6.30297236e-03j,
            ],
            [
                -0.09544978 + 1.78968334e-01j,
                0.29708743 + 4.95145724e-01j,
                0.31873657 + 0.00000000e00j,
                -0.29777767 + 4.96296112e-01j,
                -0.18306015 - 3.43237787e-01j,
            ],
            [
                -0.06819609 + 3.44424719e-03j,
                -0.14658716 + 2.74850926e-01j,
                0.29777767 + 4.96296112e-01j,
                0.12389162 + 1.10385981e-17j,
                -0.27646677 + 4.60777945e-01j,
            ],
            [
                -0.01109048 - 1.65323865e-02j,
                -0.12479885 + 6.30297236e-03j,
                -0.18306015 + 3.43237787e-01j,
                0.27646677 + 4.60777945e-01j,
                -0.03277289 + 1.88440656e-17j,
            ],
        ]
    )
    D = displacement((cutoff, cutoff), alpha)
    assert np.allclose(D, expected, atol=1e-5, rtol=0)


@given(x=st.floats(-1, 1), y=st.floats(-1, 1))
def test_number_means(x, y):
    assert np.allclose(State(ket=Coherent(x, y).ket([80])).number_means, x * x + y * y)
    assert np.allclose(State(dm=Coherent(x, y).dm([80])).number_means, x * x + y * y)


@given(x=st.floats(-1, 1), y=st.floats(-1, 1))
def test_number_variances_coh(x, y):
    assert np.allclose(
        fock.number_variances(Coherent(x, y).ket([80]), False)[0], x * x + y * y
    )
    assert np.allclose(
        fock.number_variances(Coherent(x, y).dm([80]), True)[0], x * x + y * y
    )


def test_number_variances_fock():
    assert np.allclose(fock.number_variances(Fock(n=1).ket(), False), 0)
    assert np.allclose(fock.number_variances(Fock(n=1).dm(), True), 0)


def test_normalize_dm():
    dm = np.array([[0.2, 0], [0, 0.2]])
    assert np.allclose(fock.normalize(dm, True), np.array([[0.5, 0], [0, 0.5]]))

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
from hypothesis.extra.numpy import arrays
from mrmustard.math import Math

math = Math()
import numpy as np
import tensorflow as tf
from scipy.stats import poisson

from mrmustard.lab import *
from mrmustard.utils.training import Optimizer
from mrmustard.physics import gaussian
from mrmustard import settings

np.random.seed(137)


@given(
    alpha=st.complex_numbers(min_magnitude=0, max_magnitude=1),
    eta=st.floats(0, 1),
    dc=st.floats(0, 0.2),
)
def test_detector_coherent_state(alpha, eta, dc):
    """Tests the correct Poisson statistics are generated when a coherent state hits an imperfect detector"""
    detector = PNRDetector(efficiency=eta, dark_counts=dc, modes=[0])
    ps = Coherent(x=alpha.real, y=alpha.imag) << detector
    expected = poisson.pmf(k=np.arange(len(ps)), mu=eta * np.abs(alpha) ** 2 + dc)
    assert np.allclose(ps, expected)


@given(r=st.floats(0, 0.5), phi=st.floats(0, 2 * np.pi), eta=st.floats(0, 1), dc=st.floats(0, 0.2))
def test_detector_squeezed_state(r, phi, eta, dc):
    """Tests the correct mean and variance are generated when a squeezed state hits an imperfect detector"""
    S = Sgate(r=r, phi=phi)
    ps = Vacuum(1) >> S >> PNRDetector(efficiency=eta, dark_counts=dc)
    assert np.allclose(np.sum(ps), 1.0)
    mean = np.arange(len(ps)) @ ps.numpy()
    expected_mean = eta * np.sinh(r) ** 2 + dc
    assert np.allclose(mean, expected_mean)
    variance = np.arange(len(ps)) ** 2 @ ps.numpy() - mean ** 2
    expected_variance = eta * np.sinh(r) ** 2 * (1 + eta * (1 + 2 * np.sinh(r) ** 2)) + dc
    assert np.allclose(variance, expected_variance)


@given(
    r=st.floats(0, 0.5),
    phi=st.floats(0, 2 * np.pi),
    eta_s=st.floats(0, 1),
    eta_i=st.floats(0, 1),
    dc_s=st.floats(0, 0.2),
    dc_i=st.floats(0, 0.2),
)
def test_detector_two_mode_squeezed_state(r, phi, eta_s, eta_i, dc_s, dc_i):
    """Tests the correct mean and variance are generated when a two mode squeezed state hits an imperfect detector"""
    pnr = PNRDetector(efficiency=[eta_s, eta_i], dark_counts=[dc_s, dc_i])
    ps = Vacuum(2) >> S2gate(r=r, phi=phi) >> pnr
    n = np.arange(len(ps))
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
    assert np.allclose(mean_s, expected_mean_s)
    assert np.allclose(mean_i, expected_mean_i)
    assert np.allclose(var_s, expected_var_s)
    assert np.allclose(var_i, expected_var_i)
    assert np.allclose(covar, expected_covar)


def test_postselection():
    """Check the correct state is heralded for a two-mode squeezed vacuum with perfect detector"""
    n_mean = 1.0
    n_measured = 1
    cutoff = 3
    detector = PNRDetector(efficiency=1.0, dark_counts=0.0, cutoffs=[cutoff])
    S2 = S2gate(r=np.arcsinh(np.sqrt(n_mean)), phi=0.0)
    proj_state = (Vacuum(2) >> S2 >> detector)[n_measured]
    success_prob = math.real(math.trace(proj_state))
    proj_state = proj_state / math.trace(proj_state)
    # outputs the ket/dm in the third mode by projecting the first and second in 1,2 photons
    expected_prob = 1 / (1 + n_mean) * (n_mean / (1 + n_mean)) ** n_measured
    assert np.allclose(success_prob, expected_prob)
    expected_state = np.zeros_like(proj_state)
    expected_state[n_measured, n_measured] = 1.0
    assert np.allclose(proj_state, expected_state)


@given(eta=st.floats(0, 1))
def test_loss_probs(eta):
    "Checks that a lossy channel is equivalent to quantum efficiency on detection probs"
    ideal_detector = PNRDetector(efficiency=1.0, dark_counts=0.0)
    lossy_detector = PNRDetector(efficiency=eta, dark_counts=0.0)
    S = Sgate(r=0.2, phi=[0.0, 0.7])
    BS = BSgate(theta=1.4, phi=0.0)
    L = Attenuator(transmissivity=eta)
    dms_lossy = Vacuum(2) >> S[0, 1] >> BS[0, 1] >> lossy_detector[0]
    dms_ideal = Vacuum(2) >> S[0, 1] >> BS[0, 1] >> L[0] >> ideal_detector[0]
    assert np.allclose(dms_lossy, dms_ideal)


@given(s=st.floats(min_value=0.0, max_value=10.0), X=st.floats(-10.0, 10.0))
def test_homodyne_on_2mode_squeezed_vacuum(s, X):
    homodyne = Homodyne(quadrature_angle=0.0, result=X)
    r = homodyne.r
    remaining_state = TMSV(r=np.arcsinh(np.sqrt(abs(s)))) << homodyne[0]
    cov = (
        np.diag(
            [1 - 2 * s / (1 / np.tanh(r) * (1 + s) + s), 1 + 2 * s / (1 / np.tanh(r) * (1 + s) - s)]
        )
        * settings.HBAR
        / 2.0
    )
    assert np.allclose(remaining_state.cov, cov)
    means = np.array([2 * np.sqrt(s * (1 + s)) * X / (np.exp(-2 * r) + 1 + 2 * s), 0.0]) * np.sqrt(
        2 * settings.HBAR
    )
    assert np.allclose(remaining_state.means, means)


@given(s=st.floats(1.0, 10.0), X=st.floats(-5.0, 5.0), angle=st.floats(0, np.pi * 2))
def test_homodyne_on_2mode_squeezed_vacuum_with_angle(s, X, angle):
    pass  # TODO: reimplement this test
    # homodyne = Homodyne(quadrature_angle=angle, result=X)
    # r = homodyne.r
    # remaining_state = TMSV(r=np.arcsinh(np.sqrt(abs(s)))) << homodyne[0]
    # denom = 1 + 2 * s * (s + 1) + (2 * s + 1) * np.cosh(2 * r)
    # cov = (
    #     settings.HBAR / 2
    #     * np.array(
    #         [
    #             [
    #                 1 + 2 * s - 2 * s * (s + 1)
    #                 * (1 + 2 * s + np.cosh(2 * r) + np.cos(2*angle) * np.sinh(2 * r)) / denom,
    #                 2 * s * (1 + s) * np.sin(2*angle) * np.sinh(2 * r) / denom,
    #             ],
    #             [
    #                 2 * s * (1 + s) * np.sin(2*angle) * np.sinh(2 * r) / denom,
    #                 (
    #                     1 + 2 * s + (1 + 2 * s * (1 + s)) * np.cosh(2 * r)
    #                     + 2 * s * (s + 1) * np.cos(2*angle) * np.sinh(2 * r)
    #                 )
    #                 / denom,
    #             ],
    #         ]
    #     )
    # )
    # assert np.allclose(remaining_state.cov, cov)
    # denom = 1 + 2 * s * (1 + s) + (1 + 2 * s) * np.cosh(2 * r)
    # means = (
    #     np.array(
    #         [
    #             np.sqrt(s * (1 + s))
    #             * X
    #             * (np.cos(2*angle) * (1 + 2 * s + np.cosh(2 * r)) + np.sinh(2 * r))
    #             / denom,
    #             -np.sqrt(s * (1 + s)) * X * (np.sin(2*angle) * (1 + 2 * s + np.cosh(2 * r))) / denom,
    #         ]
    #     )
    #     * np.sqrt(2 * settings.HBAR)
    # )
    # assert np.allclose(remaining_state.means, means)


@given(
    s=st.floats(min_value=0.0, max_value=10.0),
    X=st.floats(-10.0, 10.0),
    d=arrays(np.float64, 4, elements=st.floats(-10.0, 10.0)),
)
def test_homodyne_on_2mode_squeezed_vacuum_with_displacement(s, X, d):
    tmsv = TMSV(r=np.arcsinh(np.sqrt(s))) >> Dgate(x=d[:2], y=d[2:])
    homodyne = Homodyne(modes=[0], quadrature_angle=0.0, result=X)
    r = homodyne.r
    remaining_state = tmsv << homodyne[0]
    xb, xa, pb, pa = d
    means = (
        np.array(
            [
                xa
                + (2 * np.sqrt(s * (s + 1)) * (X - xb))
                / (1 + 2 * s + np.cosh(2 * r) - np.sinh(2 * r)),
                pa
                + (2 * np.sqrt(s * (s + 1)) * pb) / (1 + 2 * s + np.cosh(2 * r) + np.sinh(2 * r)),
            ]
        )
        * np.sqrt(2 * settings.HBAR)
    )
    assert np.allclose(remaining_state.means, means)


@given(
    s=st.floats(min_value=0.0, max_value=10.0),
    x=st.floats(-10.0, 10.0),
    y=st.floats(-10.0, 10.0),
    d=arrays(np.float64, 4, elements=st.floats(-10.0, 10.0)),
)
def test_heterodyne_on_2mode_squeezed_vacuum_with_displacement(
    s, x, y, d
):  # TODO: check if this is correct
    tmsv = TMSV(r=np.arcsinh(np.sqrt(s))) >> Dgate(x=d[:2], y=d[2:])
    heterodyne = Heterodyne(modes=[0], x=x, y=y)
    remaining_state = tmsv << heterodyne[0]
    cov = settings.HBAR / 2 * np.array([[1, 0], [0, 1]])
    assert np.allclose(remaining_state.cov, cov)
    xb, xa, pb, pa = d
    means = (
        np.array(
            [
                xa * (1 + s) + np.sqrt(s * (1 + s)) * (x - xb),
                pa * (1 + s) + np.sqrt(s * (1 + s)) * (pb - y),
            ]
        )
        * np.sqrt(2 * settings.HBAR)
        / (1 + s)
    )
    assert np.allclose(remaining_state.means, means, atol=1e-5)

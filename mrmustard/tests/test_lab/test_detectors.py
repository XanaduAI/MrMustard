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

import numpy as np
import tensorflow as tf
from scipy.stats import poisson

from mrmustard.lab import *
from mrmustard.utils import Optimizer
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
    cutoff = 20
    D = Dgate(x=alpha.real, y=alpha.imag)
    detector = PNRDetector(modes=[0], efficiency=eta, dark_counts=dc)
    ps = detector(Vacuum(1) >> D[0], cutoffs=[cutoff])
    expected = poisson.pmf(k=np.arange(cutoff), mu=eta * np.abs(alpha) ** 2 + dc)
    assert np.allclose(ps, expected)


@given(r=st.floats(0, 1), phi=st.floats(0, 2 * np.pi), eta=st.floats(0, 1), dc=st.floats(0, 0.2))
def test_detector_squeezed_state(r, phi, eta, dc):
    """Tests the correct mean and variance are generated when a squeezed state hits an imperfect detector"""
    S = Sgate(r=r, phi=phi)
    detector = PNRDetector(modes=[0], efficiency=eta, dark_counts=dc)
    cutoff = 50
    ps = detector(Vacuum(1) >> S[0], cutoffs=[cutoff])
    assert np.allclose(np.sum(ps), 1.0, atol=1e-3)
    mean = np.arange(cutoff) @ ps.numpy()
    expected_mean = eta * np.sinh(r) ** 2 + dc
    assert np.allclose(mean, expected_mean, atol=1e-3)
    variance = np.arange(cutoff) ** 2 @ ps.numpy() - mean ** 2
    expected_variance = eta * np.sinh(r) ** 2 * (1 + eta * (1 + 2 * np.sinh(r) ** 2)) + dc
    assert np.allclose(variance, expected_variance, atol=1e-3)


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
    S2 = S2gate(r=r, phi=phi)
    detector = PNRDetector(modes=[0, 1], efficiency=[eta_s, eta_i], dark_counts=[dc_s, dc_i])
    cutoff = 30
    ps = detector(Vacuum(2) >> S2, cutoffs=[cutoff, cutoff])
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


def test_detector_two_temporal_modes_two_mode_squeezed_vacuum():
    """Adds a basic test for convolutions with two mode squeezed vacuum"""
    tf.random.set_seed(20)
    guess = {
        "eta_s": 0.9,
        "eta_i": 0.8,
        "sq_0": np.sinh(1.0) ** 2,
        "sq_1": np.sinh(0.5) ** 2,
        "noise_s": 0.05,
        "noise_i": 0.025,
        "n_modes": 2,
    }
    cutoff = 20
    tfbe = gaussian.math
    circc = Circuit()
    circd = Circuit()
    r1 = np.arcsinh(np.sqrt(guess["sq_0"]))
    r2 = np.arcsinh(np.sqrt(guess["sq_1"]))
    S2c = S2gate(r=r1, phi=0.0, r_trainable=True, phi_trainable=True)
    S2d = S2gate(r=r2, phi=0.0, r_trainable=True, phi_trainable=True)
    circc.append(S2c)
    circd.append(S2d)
    tetas = [guess["eta_s"], guess["eta_i"]]
    tdcs = [guess["noise_s"], guess["noise_i"]]
    tdetector = PNRDetector(
        modes=[0, 1],
        efficiency=tetas,
        efficiency_trainable=True,
        efficiency_bounds=(0.7, 1.0),
        dark_counts=tdcs,
        dark_counts_trainable=True,
        dark_counts_bounds=(0.0, 0.2),
        max_cutoffs=20,
    )
    outc = circc(Vacuum(2))
    outd = circd(Vacuum(2))
    tdetector.recompute_stochastic_channel()
    psc = tdetector(outc, cutoffs=[cutoff, cutoff])
    psd = tdetector(outd, cutoffs=[cutoff, cutoff])
    fake_data = tfbe.convolve_probs(psc, psd)

    def loss_fn():
        outc = circc(Vacuum(2))
        outd = circd(Vacuum(2))
        tdetector.recompute_stochastic_channel()
        psc = tdetector(outc, cutoffs=[cutoff, cutoff])
        psd = tdetector(outd, cutoffs=[cutoff, cutoff])
        ps = tfbe.convolve_probs(psc, psd)
        return tf.norm(fake_data - ps) ** 2

    opt = Optimizer(euclidean_lr=0.001)
    opt.minimize(loss_fn, by_optimizing=[circc, circd, tdetector], max_steps=0)
    assert np.allclose(guess["sq_0"], np.sinh(S2c.trainable_parameters["euclidean"][0].numpy()) ** 2)
    assert np.allclose(guess["sq_1"], np.sinh(S2d.trainable_parameters["euclidean"][0].numpy()) ** 2)
    assert np.allclose(tdetector.efficiency, [guess["eta_s"], guess["eta_i"]])
    assert np.allclose(tdetector.dark_counts, [guess["noise_s"], guess["noise_i"]])


def test_postselection():
    """Check the correct state is heralded for a two-mode squeezed vacuum with perfect detector"""
    n_mean = 1.0
    detector = PNRDetector(modes=[0, 1], efficiency=1.0, dark_counts=0.0)
    S2 = S2gate(r=np.arcsinh(np.sqrt(n_mean)), phi=0.0)
    my_state = Vacuum(2) >> S2
    cutoff = 3
    n_measured = 1
    # outputs the ket/dm in the third mode by projecting the first and second in 1,2 photons
    proj_state, success_prob = detector(my_state, cutoffs=[cutoff, cutoff], outcomes=[n_measured, None])
    expected_prob = 1 / (1 + n_mean) * (n_mean / (1 + n_mean)) ** n_measured
    assert np.allclose(success_prob, expected_prob)
    expected_state = np.zeros([cutoff, cutoff])
    expected_state[n_measured, n_measured] = 1.0
    assert np.allclose(proj_state, expected_state)


@given(eta=st.floats(0, 1))
def test_loss_probs(eta):
    "Checks that a lossy channel is equivalent to quantum efficiency on detection probs"
    lossy_detector = PNRDetector(modes=[0, 1], efficiency=eta, dark_counts=0.0)
    ideal_detector = PNRDetector(modes=[0, 1], efficiency=1.0, dark_counts=0.0)
    S = Sgate(r=0.3, phi=[0.0, 0.7])[0, 1]
    B = BSgate(theta=1.4, phi=0.0)[0, 1]
    L = LossChannel(transmissivity=eta)[0, 1]
    dm_lossy = lossy_detector(Vacuum(2) >> S >> B, cutoffs=[20, 20])
    dm_ideal = ideal_detector(Vacuum(2) >> S >> B >> L, cutoffs=[20, 20])
    assert np.allclose(dm_ideal, dm_lossy)


@given(eta=st.floats(0, 1), n=st.integers(0, 2))
def test_projected(eta, n):
    "Checks that a lossy channel is equivalent to quantum efficiency on projected states"
    lossy_detector = PNRDetector(modes=[0, 1], efficiency=eta, dark_counts=0.0)
    ideal_detector = PNRDetector(modes=[0, 1], efficiency=1.0, dark_counts=0.0)
    S = Sgate(r=0.3, phi=[0.0, 1.5])
    B = BSgate(theta=1.0, phi=0.0)
    L = LossChannel(transmissivity=eta)
    dm_lossy, _ = lossy_detector(Vacuum(2) >> S[0, 1] >> B, cutoffs=[20, 20], outcomes=[n, None])
    dm_ideal, _ = ideal_detector(Vacuum(2) >> S[0, 1] >> B >> L[0], cutoffs=[20, 20], outcomes=[n, None])
    assert np.allclose(dm_ideal, dm_lossy)


@given(s=st.floats(min_value=0.0, max_value=10.0), X=st.floats(-10.0, 10.0))
def test_homodyne_on_2mode_squeezed_vacuum(s, X):
    homodyne = Homodyne(modes=[0], quadrature_angles=0.0, results=X)
    r = homodyne._squeezing
    prob, remaining_state = homodyne(TMSV(r=np.arcsinh(np.sqrt(abs(s)))))
    cov = (
        np.diag([1 - 2 * s / (1 / np.tanh(r) * (1 + s) + s), 1 + 2 * s / (1 / np.tanh(r) * (1 + s) - s)])
        * settings.HBAR
        / 2.0
    )
    assert np.allclose(remaining_state.cov, cov)
    means = np.array([2 * np.sqrt(s * (1 + s)) * X / (np.exp(-2 * r) + 1 + 2 * s), 0.0]) * np.sqrt(2 * settings.HBAR)
    assert np.allclose(remaining_state.means, means)


@given(s=st.floats(1.0, 20.0), X=st.floats(-10.0, 10.0), angle=st.floats(0, np.pi * 2))
def test_homodyne_on_2mode_squeezed_vacuum_with_angle(s, X, angle):
    homodyne = Homodyne(modes=[0], quadrature_angles=angle, results=X)
    r = homodyne._squeezing
    prob, remaining_state = homodyne(TMSV(r=np.arcsinh(np.sqrt(abs(s)))))
    denom = 1 + 2 * s * (s + 1) + (2 * s + 1) * np.cosh(2 * r)
    cov = (
        settings.HBAR
        / 2
        * np.array(
            [
                [
                    1 + 2 * s - 2 * s * (s + 1) * (1 + 2 * s + np.cosh(2 * r) + np.cos(angle) * np.sinh(2 * r)) / denom,
                    2 * s * (1 + s) * np.sin(angle) * np.sinh(2 * r) / denom,
                ],
                [
                    2 * s * (1 + s) * np.sin(angle) * np.sinh(2 * r) / denom,
                    (
                        1
                        + 2 * s
                        + (1 + 2 * s * (1 + s)) * np.cosh(2 * r)
                        + 2 * s * (s + 1) * np.cos(angle) * np.sinh(2 * r)
                    )
                    / denom,
                ],
            ]
        )
    )
    assert np.allclose(remaining_state.cov, cov)
    denom = 1 + 2 * s * (1 + s) + (1 + 2 * s) * np.cosh(2 * r)
    means = (
        np.array(
            [
                np.sqrt(s * (1 + s)) * X * (np.cos(angle) * (1 + 2 * s + np.cosh(2 * r)) + np.sinh(2 * r)) / denom,
                -np.sqrt(s * (1 + s)) * X * (np.sin(angle) * (1 + 2 * s + np.cosh(2 * r))) / denom,
            ]
        )
        * np.sqrt(2 * settings.HBAR)
    )
    assert np.allclose(remaining_state.means, means)


@given(
    s=st.floats(min_value=0.0, max_value=10.0),
    X=st.floats(-10.0, 10.0),
    d=arrays(np.float64, 4, elements=st.floats(-10.0, 10.0)),
)
def test_homodyne_on_2mode_squeezed_vacuum_with_displacement(s, X, d):
    S = S2gate(r=np.arcsinh(np.sqrt(abs(s))), phi=0.0)[0, 1]
    D = Dgate(x=d[:2], y=d[2:])[0, 1]
    tmsv = Vacuum(2) >> S >> D
    homodyne = Homodyne(modes=[0], quadrature_angles=0.0, results=X)
    r = homodyne._squeezing
    prob, remaining_state = homodyne(tmsv)
    xb, xa, pb, pa = d
    means = (
        np.array(
            [
                xa + (2 * np.sqrt(s * (s + 1)) * (X - xb)) / (1 + 2 * s + np.cosh(2 * r) - np.sinh(2 * r)),
                pa + (2 * np.sqrt(s * (s + 1)) * pb) / (1 + 2 * s + np.cosh(2 * r) + np.sinh(2 * r)),
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
def test_heterodyne_on_2mode_squeezed_vacuum_with_displacement(s, x, y, d):  # TODO: check if this is correct
    S = S2gate(r=np.arcsinh(np.sqrt(abs(s))), phi=0.0)
    D = Dgate(x=d[:2], y=d[2:])
    tmsv = Vacuum(2) >> S[0, 1] >> D[0, 1]
    heterodyne = Heterodyne(modes=[0], x=x, y=y)
    prob, remaining_state = heterodyne(tmsv)
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

import pytest

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
import tensorflow as tf
from scipy.stats import poisson

from mrmustard import Dgate, Sgate, S2gate, LossChannel, BSgate
from mrmustard import Circuit, Optimizer
from mrmustard import Vacuum
from mrmustard import PNRDetector, Homodyne, Heterodyne
from mrmustard.plugins import gaussian

np.random.seed(137)


@pytest.mark.parametrize("alpha", np.random.rand(3) + 1j * np.random.rand(3))
@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("dc", [0.0, 0.2])
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
    tfbe = gaussian.backend
    circc = Circuit()
    circd = Circuit()
    r1 = np.arcsinh(np.sqrt(guess["sq_0"]))
    r2 = np.arcsinh(np.sqrt(guess["sq_1"]))
    S2c = S2gate(modes=[0, 1], r=r1, phi=0.0)
    S2d = S2gate(modes=[0, 1], r=r2, phi=0.0)
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

    outc = circc(Vacuum(num_modes=2))
    outd = circd(Vacuum(num_modes=2))
    tdetector.recompute_stochastic_channel()
    psc = tdetector(outc, cutoffs=[cutoff, cutoff])
    psd = tdetector(outd, cutoffs=[cutoff, cutoff])
    fake_data = tfbe.convolve_probs(psc, psd)

    def loss_fn():
        outc = circc(Vacuum(num_modes=2))
        outd = circd(Vacuum(num_modes=2))
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
    S2 = S2gate(modes=[0, 1], r=np.arcsinh(np.sqrt(n_mean)), phi=0.0)
    my_state = S2(Vacuum(num_modes=2))

    cutoff = 3
    n_measured = 1

    # outputs the ket/dm in the third mode by projecting the first and second in 1,2 photons
    proj_state, success_prob = detector(my_state, cutoffs=[cutoff, cutoff], outcomes=[n_measured, None])
    expected_prob = 1 / (1 + n_mean) * (n_mean / (1 + n_mean)) ** n_measured
    assert np.allclose(success_prob, expected_prob)
    expected_state = np.zeros([cutoff, cutoff])
    expected_state[n_measured, n_measured] = 1.0
    assert np.allclose(proj_state, expected_state)


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.9, 1.0])
def test_loss_probs(eta):
    "Checks that a lossy channel is equivalent to quantum efficiency on detection probs"
    lossy_detector = PNRDetector(modes=[0, 1], efficiency=eta, dark_counts=0.0)
    ideal_detector = PNRDetector(modes=[0, 1], efficiency=1.0, dark_counts=0.0)
    S = Sgate(modes=[0, 1], r=0.3, phi=[0.0, 0.7])
    B = BSgate(modes=[0, 1], theta=1.4, phi=0.0)
    L = LossChannel(modes=[0, 1], transmissivity=eta)

    dm_lossy = lossy_detector(B(S(Vacuum(2))), cutoffs=[20, 20])
    dm_ideal = ideal_detector(L(B(S(Vacuum(2)))), cutoffs=[20, 20])

    assert np.allclose(dm_ideal, dm_lossy)


@pytest.mark.parametrize("eta", [0.2, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_projected(eta, n):
    "Checks that a lossy channel is equivalent to quantum efficiency on projected states"
    lossy_detector = PNRDetector(modes=[0, 1], efficiency=eta, dark_counts=0.0)
    ideal_detector = PNRDetector(modes=[0, 1], efficiency=1.0, dark_counts=0.0)
    S = Sgate(modes=[0, 1], r=0.3, phi=[0.0, 1.5])
    B = BSgate(modes=[0, 1], theta=1.0, phi=0.0)
    L = LossChannel(modes=[0], transmissivity=eta)

    dm_lossy, _ = lossy_detector(B(S(Vacuum(2))), cutoffs=[20, 20], outcomes=[n, None])
    dm_ideal, _ = ideal_detector(L(B(S(Vacuum(2)))), cutoffs=[20, 20], outcomes=[n, None])

    assert np.allclose(dm_ideal, dm_lossy)


@given(s=st.floats(min_value=0.0, max_value=10.0), X=st.floats(-10.0, 10.0))
def test_homodyne_on_2mode_squeezed_vacuum(s, X):
    S = S2gate(modes=[0, 1], r=np.arcsinh(np.sqrt(abs(s))), phi=0.0)
    tmsv = S(Vacuum(2))
    homodyne = Homodyne(modes=[0], quadrature_angles=0.0, results=X)
    r = homodyne._squeezing
    prob, remaining_state = homodyne(tmsv)
    cov = np.diag([1 - 2 * s / (1 / np.tanh(r) * (1 + s) + s), 1 + 2 * s / (1 / np.tanh(r) * (1 + s) - s)]) * tmsv.hbar / 2.0
    assert np.allclose(remaining_state.cov, cov)
    means = np.array([2 * np.sqrt(s * (1 + s)) * X / (np.exp(-2 * r) + 1 + 2 * s), 0.0]) * np.sqrt(2 * tmsv.hbar)
    assert np.allclose(remaining_state.means, means)


@given(s=st.floats(1.0, 20.0), X=st.floats(-10.0, 10.0), angle=st.floats(0, np.pi * 2))
def test_homodyne_on_2mode_squeezed_vacuum_with_angle(s, X, angle):
    S = S2gate(modes=[0, 1], r=np.arcsinh(np.sqrt(abs(s))), phi=0.0)
    tmsv = S(Vacuum(2))
    homodyne = Homodyne(modes=[0], quadrature_angles=angle, results=X)
    r = homodyne._squeezing
    prob, remaining_state = homodyne(tmsv)
    denom = 1 + 2 * s * (s + 1) + (2 * s + 1) * np.cosh(2 * r)
    cov = (
        tmsv.hbar
        / 2
        * np.array(
            [
                [
                    1 + 2 * s - 2 * s * (s + 1) * (1 + 2 * s + np.cosh(2 * r) + np.cos(angle) * np.sinh(2 * r)) / denom,
                    2 * s * (1 + s) * np.sin(angle) * np.sinh(2 * r) / denom,
                ],
                [
                    2 * s * (1 + s) * np.sin(angle) * np.sinh(2 * r) / denom,
                    (1 + 2 * s + (1 + 2 * s * (1 + s)) * np.cosh(2 * r) + 2 * s * (s + 1) * np.cos(angle) * np.sinh(2 * r)) / denom,
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
        * np.sqrt(2 * tmsv.hbar)
    )
    assert np.allclose(remaining_state.means, means)


@given(s=st.floats(min_value=0.0, max_value=10.0), X=st.floats(-10.0, 10.0), d=arrays(np.float64, 4, elements=st.floats(-10.0, 10.0)))
def test_homodyne_on_2mode_squeezed_vacuum_with_displacement(s, X, d):
    S = S2gate(modes=[0, 1], r=np.arcsinh(np.sqrt(abs(s))), phi=0.0)
    D = Dgate(modes=[0, 1], x=d[:2], y=d[2:])
    tmsv = D(S(Vacuum(2)))
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
        * np.sqrt(2 * tmsv.hbar)
    )
    assert np.allclose(remaining_state.means, means)


# Test heterodyne now
@given(
    s=st.floats(min_value=0.0, max_value=10.0),
    x=st.floats(-10.0, 10.0),
    y=st.floats(-10.0, 10.0),
    d=arrays(np.float64, 4, elements=st.floats(-10.0, 10.0)),
)
def test_heterodyne_on_2mode_squeezed_vacuum_with_displacement(s, x, y, d):
    S = S2gate(modes=[0, 1], r=np.arcsinh(np.sqrt(abs(s))), phi=0.0)
    D = Dgate(modes=[0, 1], x=d[:2], y=d[2:])
    tmsv = D(S(Vacuum(2)))
    heterodyne = Heterodyne(modes=[0], x=x, y=y, hbar=tmsv.hbar)
    prob, remaining_state = heterodyne(tmsv)
    cov = tmsv.hbar / 2 * np.array([[1, 0], [0, 1]])
    assert np.allclose(remaining_state.cov, cov)
    xb, xa, pb, pa = d
    means = (
        np.array([xa * (1 + s) + np.sqrt(s * (1 + s)) * (x - xb), pa * (1 + s) + np.sqrt(s * (1 + s)) * (pb - y)])
        * np.sqrt(2 * tmsv.hbar)
        / (1 + s)
    )
    assert np.allclose(remaining_state.means, means, atol=1e-5)

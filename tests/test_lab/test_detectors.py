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

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import poisson

from mrmustard import math, physics, settings
from mrmustard.lab import (
    TMSV,
    Attenuator,
    BSgate,
    Coherent,
    Dgate,
    Fock,
    Heterodyne,
    Homodyne,
    PNRDetector,
    S2gate,
    Sgate,
    SqueezedVacuum,
    State,
    Vacuum,
)
from tests.random import none_or_

from ..conftest import skip_np


hbar = settings.HBAR


class TestPNRDetector:
    """tests related to PNR detectors"""

    @given(
        alpha=st.complex_numbers(min_magnitude=0, max_magnitude=1),
        eta=st.floats(0, 1),
        dc=st.floats(0, 0.2),
    )
    def test_detector_coherent_state(self, alpha, eta, dc):
        """Tests the correct Poisson statistics are generated when a coherent state hits an imperfect detector"""
        skip_np()

        detector = PNRDetector(efficiency=eta, dark_counts=dc, modes=[0])
        ps = Coherent(x=alpha.real, y=alpha.imag) << detector
        expected = poisson.pmf(k=np.arange(len(ps)), mu=eta * np.abs(alpha) ** 2 + dc)
        assert np.allclose(ps, expected)

    @given(
        r=st.floats(0, 0.5),
        phi=st.floats(0, 2 * np.pi),
        eta=st.floats(0, 1),
        dc=st.floats(0, 0.2),
    )
    def test_detector_squeezed_state(self, r, phi, eta, dc):
        """Tests the correct mean and variance are generated when a squeezed state hits an imperfect detector"""
        skip_np()

        S = Sgate(r=r, phi=phi)
        ps = Vacuum(1) >> S >> PNRDetector(efficiency=eta, dark_counts=dc)
        assert np.allclose(np.sum(ps), 1.0)
        mean = np.arange(len(ps)) @ math.asnumpy(ps)
        expected_mean = eta * np.sinh(r) ** 2 + dc
        assert np.allclose(mean, expected_mean)
        variance = np.arange(len(ps)) ** 2 @ math.asnumpy(ps) - mean**2
        expected_variance = (
            eta * np.sinh(r) ** 2 * (1 + eta * (1 + 2 * np.sinh(r) ** 2)) + dc
        )
        assert np.allclose(variance, expected_variance)

    @given(
        r=st.floats(0, 0.5),
        phi=st.floats(0, 2 * np.pi),
        eta_s=st.floats(0, 1),
        eta_i=st.floats(0, 1),
        dc_s=st.floats(0, 0.2),
        dc_i=st.floats(0, 0.2),
    )
    def test_detector_two_mode_squeezed_state(self, r, phi, eta_s, eta_i, dc_s, dc_i):
        """Tests the correct mean and variance are generated when a two mode squeezed state hits an imperfect detector"""
        skip_np()

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
        var_s = np.sum(ps, axis=1) @ n**2 - mean_s**2
        var_i = np.sum(ps, axis=0) @ n**2 - mean_i**2
        expected_var_s = n_s * (n_s + 1) + dc_s
        expected_var_i = n_i * (n_i + 1) + dc_i
        covar = n @ math.asnumpy(ps) @ n - mean_s * mean_i
        expected_covar = eta_s * eta_i * (np.sinh(r) * np.cosh(r)) ** 2
        assert np.allclose(mean_s, expected_mean_s)
        assert np.allclose(mean_i, expected_mean_i)
        assert np.allclose(var_s, expected_var_s)
        assert np.allclose(var_i, expected_var_i)
        assert np.allclose(covar, expected_covar)

    def test_postselection(self):
        """Check the correct state is heralded for a two-mode squeezed vacuum with perfect detector"""
        skip_np()

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
    def test_loss_probs(self, eta):
        "Checks that a lossy channel is equivalent to quantum efficiency on detection probs"
        skip_np()

        ideal_detector = PNRDetector(efficiency=1.0, dark_counts=0.0)
        lossy_detector = PNRDetector(efficiency=eta, dark_counts=0.0)
        S = Sgate(r=0.2, phi=[0.0, 0.7])
        BS = BSgate(theta=1.4, phi=0.0)
        L = Attenuator(transmissivity=eta)
        dms_lossy = Vacuum(2) >> S[0, 1] >> BS[0, 1] >> lossy_detector[0]
        dms_ideal = Vacuum(2) >> S[0, 1] >> BS[0, 1] >> L[0] >> ideal_detector[0]
        assert np.allclose(dms_lossy, dms_ideal, atol=1e-6)


class TestHomodyneDetector:
    """tests related to homodyne detectors"""

    @pytest.mark.parametrize(
        "outcome", [None] + np.random.uniform(-5, 5, size=(10, 2)).tolist()
    )
    def test_homodyne_mode_kwargs(self, outcome):
        """Test that S gates and Homodyne mesurements are applied to the correct modes via the
        `modes` kwarg.

        Here the initial state is a "diagonal" (angle=pi/2) squeezed state in mode 0, a "vertical"
        (angle=0) squeezed state in mode 1 and vacuum in mode 2. Because the modes are separable,
        measuring modes 1 and 2 should leave the state in the mode 0 unchaged.

        Also checks postselection ensuring the x-quadrature value is consistent with the
        postselected value.
        """
        S1 = Sgate(modes=[0], r=1, phi=np.pi / 2)
        S2 = Sgate(modes=[1], r=1, phi=0)
        initial_state = Vacuum(3) >> S1 >> S2
        detector = Homodyne(result=outcome, quadrature_angle=[0.0, 0.0], modes=[1, 2])
        final_state = initial_state << detector

        expected_state = Vacuum(1) >> S1

        assert np.allclose(final_state.dm(), expected_state.dm())

        if outcome is not None:
            # checks postselection ensuring the x-quadrature
            # value is consistent with the postselected value
            x_outcome = math.asnumpy(detector.outcome)[:2]
            assert np.allclose(x_outcome, outcome)

    @given(
        s=st.floats(min_value=0.0, max_value=10.0),
        outcome=none_or_(st.floats(-10.0, 10.0)),
    )
    def test_homodyne_on_2mode_squeezed_vacuum(self, s, outcome):
        """Check that homodyne detection on TMSV for q-quadrature (``quadrature_angle=0.0``)"""
        r = settings.HOMODYNE_SQUEEZING
        detector = Homodyne(quadrature_angle=0.0, result=outcome, r=r)
        remaining_state = TMSV(r=np.arcsinh(np.sqrt(abs(s)))) << detector[0]

        # assert expected covariance matrix
        cov = (hbar / 2.0) * np.diag(
            [
                1 - 2 * s / (1 / np.tanh(r) * (1 + s) + s),
                1 + 2 * s / (1 / np.tanh(r) * (1 + s) - s),
            ]
        )
        assert np.allclose(remaining_state.cov, cov)

        # assert expected means vector, not tested when sampling (i.e. ``outcome == None``)
        # because we cannot access the sampled outcome value
        if outcome is not None:
            means = np.array(
                [2 * np.sqrt(s * (1 + s)) * outcome / (np.exp(-2 * r) + 1 + 2 * s), 0.0]
            )
            assert np.allclose(math.asnumpy(remaining_state.means), means)

    @given(
        s=st.floats(1.0, 10.0),
        outcome=none_or_(st.floats(-2, 2)),
        angle=st.floats(0, np.pi),
    )
    def test_homodyne_on_2mode_squeezed_vacuum_with_angle(self, s, outcome, angle):
        """Check that homodyne detection on TMSV works with an arbitrary quadrature angle"""
        r = settings.HOMODYNE_SQUEEZING
        detector = Homodyne(quadrature_angle=angle, result=outcome, r=r)
        remaining_state = TMSV(r=np.arcsinh(np.sqrt(abs(s)))) << detector[0]
        denom = 1 + 2 * s * (s + 1) + (2 * s + 1) * np.cosh(2 * r)
        cov = (hbar / 2) * np.array(
            [
                [
                    1
                    + 2 * s
                    - 2
                    * s
                    * (s + 1)
                    * (1 + 2 * s + np.cosh(2 * r) + np.cos(2 * angle) * np.sinh(2 * r))
                    / denom,
                    2 * s * (1 + s) * np.sin(2 * angle) * np.sinh(2 * r) / denom,
                ],
                [
                    2 * s * (1 + s) * np.sin(2 * angle) * np.sinh(2 * r) / denom,
                    (
                        1
                        + 2 * s
                        + (1 + 2 * s * (1 + s)) * np.cosh(2 * r)
                        + 2 * s * (s + 1) * np.cos(2 * angle) * np.sinh(2 * r)
                    )
                    / denom,
                ],
            ]
        )
        assert np.allclose(math.asnumpy(remaining_state.cov), cov, atol=1e-5)

    @given(
        s=st.floats(min_value=0.0, max_value=1.0),
        X=st.floats(-1.0, 1.0),
        d=arrays(np.float64, 4, elements=st.floats(-1.0, 1.0)),
    )
    def test_homodyne_on_2mode_squeezed_vacuum_with_displacement(self, s, X, d):
        """Check that homodyne detection on displaced TMSV works"""
        tmsv = TMSV(r=np.arcsinh(np.sqrt(s))) >> Dgate(x=d[:2], y=d[2:])
        r = settings.HOMODYNE_SQUEEZING
        detector = Homodyne(modes=[0], quadrature_angle=0.0, result=X, r=r)
        remaining_state = tmsv << detector
        xb, xa, pb, pa = np.sqrt(2 * hbar) * d
        expected_means = np.array(
            [
                xa
                + (2 * np.sqrt(s * (s + 1)) * (X - xb))
                / (1 + 2 * s + np.cosh(2 * r) - np.sinh(2 * r)),
                pa
                + (2 * np.sqrt(s * (s + 1)) * pb)
                / (1 + 2 * s + np.cosh(2 * r) + np.sinh(2 * r)),
            ]
        )

        means = math.asnumpy(remaining_state.means)
        assert np.allclose(means, expected_means)

    N_MEAS = 150  # number of homodyne measurements to perform
    NUM_STDS = 10.0
    std_10 = NUM_STDS / np.sqrt(N_MEAS)

    @pytest.mark.parametrize(
        "state, kwargs, mean_expected, var_expected",
        [
            (Vacuum, {"num_modes": 1}, 0.0, settings.HBAR / 2),
            (
                Coherent,
                {"x": 2.0, "y": 0.5},
                2.0 * np.sqrt(2 * settings.HBAR),
                settings.HBAR / 2,
            ),
            (SqueezedVacuum, {"r": 0.25, "phi": 0.0}, 0.0, 0.25 * settings.HBAR / 2),
        ],
    )
    @pytest.mark.parametrize("gaussian_state", [True, False])
    @pytest.mark.parametrize("normalization", [1, 1 / 3])
    def test_sampling_mean_and_var(
        self, state, kwargs, mean_expected, var_expected, gaussian_state, normalization
    ):
        """Tests that the mean and variance estimates of many homodyne
        measurements are in agreement with the expected values for the states"""
        state = state(**kwargs)

        if not gaussian_state:
            state = State(dm=state.dm(cutoffs=[40]) * normalization)
        detector = Homodyne(0.0)

        results = np.zeros((self.N_MEAS, 2))
        for i in range(self.N_MEAS):
            _ = state << detector
            results[i] = math.asnumpy(detector.outcome)

        mean = results.mean(axis=0)
        assert np.allclose(mean[0], mean_expected, atol=self.std_10, rtol=0)
        var = results.var(axis=0)
        assert np.allclose(var[0], var_expected, atol=self.std_10, rtol=0)

    def test_homodyne_squeezing_setting(self):
        r"""Check default homodyne squeezing on settings leads to the correct generaldyne
        covarince matrix: one that has tends to :math:`diag(1/\sigma[1,1],0)`."""

        sigma = np.identity(2)
        sigma_m = math.asnumpy(SqueezedVacuum(r=settings.HOMODYNE_SQUEEZING, phi=0).cov)

        inverse_covariance = np.linalg.inv(sigma + sigma_m)
        assert np.allclose(inverse_covariance, np.diag([1 / sigma[1, 1], 0]))


class TestHeterodyneDetector:
    """tests related to heterodyne detectors"""

    @pytest.mark.parametrize(
        "xy", [[None, None]] + np.random.uniform(-10, 10, size=(5, 2, 2)).tolist()
    )
    def test_heterodyne_mode_kwargs(self, xy):
        """Test that S gates and Heterodyne mesurements are applied to the correct modes via the `modes` kwarg.

        Here the initial state is a "diagonal" (angle=pi/2) squeezed state in mode 0,
        a "vertical" (angle=0) squeezed state in mode 1 and vacumm state in mode 2.

        Because the modes are separable, measuring in mode 1 and 2 should leave the state in the
        0th mode unchaged.
        """
        x, y = xy

        S1 = Sgate(modes=[0], r=1, phi=np.pi / 2)
        S2 = Sgate(modes=[1], r=1, phi=0)
        initial_state = Vacuum(3) >> S1 >> S2
        final_state = initial_state << Heterodyne(x, y, modes=[1, 2])

        expected_state = Vacuum(1) >> S1

        assert np.allclose(final_state.dm(), expected_state.dm())

    @given(
        s=st.floats(min_value=0.0, max_value=10.0),
        x=none_or_(st.floats(-10.0, 10.0)),
        y=none_or_(st.floats(-10.0, 10.0)),
        d=arrays(np.float64, 4, elements=st.floats(-10.0, 10.0)),
    )
    def test_heterodyne_on_2mode_squeezed_vacuum_with_displacement(
        self, s, x, y, d
    ):  # TODO: check if this is correct
        """Check that heterodyne detection on TMSV works with an arbitrary displacement"""
        if x is None or y is None:
            x, y = None, None

        tmsv = TMSV(r=np.arcsinh(np.sqrt(s))) >> Dgate(x=d[:2], y=d[2:])
        heterodyne = Heterodyne(modes=[0], x=x, y=y)
        remaining_state = tmsv << heterodyne[0]

        # assert expected covariance
        cov = hbar / 2 * np.array([[1, 0], [0, 1]])
        assert np.allclose(math.asnumpy(remaining_state.cov), cov)

        # assert expected means vector, not tested when x or y is None
        # because we cannot access the sampled outcome value
        if (x is not None) and (y is not None):
            xb, xa, pb, pa = d * np.sqrt(2 * hbar)
            means = np.array(
                [
                    xa * (1 + s) + np.sqrt(s * (1 + s)) * (x - xb),
                    pa * (1 + s) + np.sqrt(s * (1 + s)) * (pb - y),
                ]
            ) / (1 + s)
            assert np.allclose(remaining_state.means, means, atol=1e-5)


class TestNormalization:
    """tests evaluating normalization of output states after projection"""

    def test_norm_1mode(self):
        """Checks that projecting a single mode coherent state onto a number state
        returns the expected norm."""
        assert np.allclose(
            Coherent(2.0) << Fock(3),
            np.abs((2.0**3) / np.sqrt(6) * np.exp(-0.5 * 4.0)) ** 2,
        )

    @pytest.mark.parametrize(
        "normalize, expected_norm",
        ([True, 1.0], [False, (2.0**3) / np.sqrt(6) * np.exp(-0.5 * 4.0)]),
    )
    def test_norm_2mode(self, normalize, expected_norm):
        """Checks that projecting a two-mode coherent state onto a number state
        produces a state with the expected norm."""
        leftover = Coherent(x=[2.0, 2.0]) << Fock(3, normalize=normalize)[0]
        assert np.isclose(
            expected_norm * np.sqrt(settings.AUTOCUTOFF_PROBABILITY),
            physics.norm(leftover),
            rtol=1 - settings.AUTOCUTOFF_PROBABILITY,
        )

    def test_norm_2mode_gaussian_normalized(self):
        """Checks that after projection the norm of the leftover state is as expected."""
        leftover = Coherent(x=[2.0, 2.0]) << Coherent(x=1.0, normalize=True)[0]
        assert np.isclose(1.0, physics.norm(leftover), atol=1e-5)


class TestProjectionOnState:
    r"""Tests the cases that the projection state is given."""

    def test_vacuum_project_on_vacuum(self):
        """Tests that the probability of Vacuum that projects on Vacuum is 1.0."""
        assert np.allclose(Vacuum(3) << Vacuum(3), 1.0)
        assert np.allclose(Vacuum(3) << Coherent([0, 0, 0]), 1.0)
        assert np.allclose(Vacuum(3) << Fock([0, 0, 0]), 1.0)

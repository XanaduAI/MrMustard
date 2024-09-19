# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the sampler."""

# pylint: disable=missing-function-docstring

import numpy as np

from mrmustard import math, settings
from mrmustard.lab_dev.samplers import PNRSampler, HomodyneSampler
from mrmustard.lab_dev import Coherent, Number, Vacuum, QuadratureEigenstate


class TestPNRSampler:
    r"""
    Tests ``PNRSampler`` objects.
    """

    def test_init(self):
        sampler = PNRSampler(cutoff=10)
        assert sampler.meas_outcomes == list(range(10))
        assert sampler.povms == Number([0], 0)

    def test_probabilities(self):
        atol = 1e-4

        sampler = PNRSampler(cutoff=10)
        vac_prob = [1.0] + [0.0] * 99
        assert math.allclose(sampler.probabilities(Vacuum([0, 1])), vac_prob)

        coh_state = Coherent([0, 1], x=[0.5, 1])
        exp_probs = [
            (coh_state >> Number([0], n0).dual >> Number([1], n1).dual) ** 2
            for n0 in range(10)
            for n1 in range(10)
        ]
        assert math.allclose(sampler.probabilities(coh_state), exp_probs, atol)

    def test_sample(self):
        n_samples = 1000
        sampler = PNRSampler(cutoff=10)

        assert not np.any(sampler.sample(Vacuum([0])))
        assert not np.any(sampler.sample_prob_dist(Vacuum([0])))
        assert not np.any(sampler.sample(Vacuum([0, 1])))
        assert not np.any(sampler.sample_prob_dist(Vacuum([0, 1])))

        state = Coherent([0], x=[0.1])
        samples = sampler.sample(state, n_samples)

        count = np.zeros_like(sampler.meas_outcomes)
        for sample in samples:
            idx = sampler.meas_outcomes.index(sample)
            count[idx] += 1
        probs = count / n_samples

        assert np.allclose(probs, sampler.probabilities(state), atol=1e-2)


class TestHomodyneSampler:
    r"""
    Tests ``HomodyneSampler`` objects.
    """

    def test_init(self):
        sampler = HomodyneSampler(phi=0.5, bounds=(-5, 5), num=100)
        assert sampler.povms == QuadratureEigenstate([0], x=0, phi=0.5)
        assert math.allclose(sampler.meas_outcomes, list(np.linspace(-5, 5, 100)))

    def test_probabilties(self):
        sampler = HomodyneSampler()

        state = Coherent([0], x=[0.1])

        exp_probs = (
            state.quadrature_distribution(sampler.meas_outcomes)
            * sampler._step  # pylint: disable=protected-access
        )
        assert math.allclose(sampler.probabilities(state), exp_probs)

        sampler2 = HomodyneSampler(phi=np.pi / 2)

        exp_probs = (
            state.quadrature_distribution(sampler2.meas_outcomes, sampler2.povms[0].phi.value[0])
            * sampler2._step  # pylint: disable=protected-access
        )
        assert math.allclose(sampler2.probabilities(state), exp_probs)

    def test_sample(self):
        n_samples = 1000
        sampler = HomodyneSampler()
        state = Coherent([0], x=[0.1])
        samples = sampler.sample(state, n_samples)

        count = np.zeros_like(sampler.meas_outcomes)
        for sample in samples:
            idx = sampler.meas_outcomes.index(sample)
            count[idx] += 1
        probs = count / n_samples

        assert np.allclose(probs, sampler.probabilities(state), atol=1e-2)

    def test_sample_mean_coherent(self):
        r"""
        Porting test from strawberry fields:
        https://github.com/XanaduAI/strawberryfields/blob/master/tests/backend/test_homodyne.py#L56
        """
        N_MEAS = 300
        NUM_STDS = 10.0
        std_10 = NUM_STDS / np.sqrt(N_MEAS)
        alpha = 1.0 + 1.0j
        x = np.empty(0)
        tol = settings.ATOL

        state = Coherent([0], x=math.real(alpha), y=math.imag(alpha))
        sampler = HomodyneSampler()

        for _ in range(N_MEAS):
            meas_result = sampler.sample(state, 1)[0]
            x = np.append(x, meas_result)

        assert math.allclose(x.mean(), 2 * alpha.real, atol=std_10 + tol)

    def test_sample_mean_and_std_vacuum(self):
        r"""
        Porting test from strawberry fields:
        https://github.com/XanaduAI/strawberryfields/blob/master/tests/backend/test_homodyne.py#L40
        """
        N_MEAS = 300
        NUM_STDS = 10.0
        std_10 = NUM_STDS / np.sqrt(N_MEAS)
        x = np.empty(0)
        tol = settings.ATOL

        state = Vacuum([0])
        sampler = HomodyneSampler()

        for _ in range(N_MEAS):
            meas_result = sampler.sample(state, 1)[0]
            x = np.append(x, meas_result)

        assert np.allclose(x.mean(), 0.0, atol=std_10 + tol, rtol=0)
        assert np.allclose(x.std(), 1.0, atol=std_10 + tol, rtol=0)

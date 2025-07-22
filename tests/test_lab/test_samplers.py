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

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab import Coherent, Number, Vacuum
from mrmustard.lab.samplers import HomodyneSampler, PNRSampler


class TestPNRSampler:
    r"""
    Tests ``PNRSampler`` objects.
    """

    def test_init(self):
        sampler = PNRSampler(cutoff=10)
        assert sampler.meas_outcomes == list(range(10))
        assert sampler.povms == Number(0, 0)

    def test_probabilities(self):
        atol = 1e-4

        sampler = PNRSampler(cutoff=10)
        vac_prob = [1.0] + [0.0] * 99
        assert math.allclose(sampler.probabilities(Vacuum((0, 1))), vac_prob)

        coh_state = Coherent(0, x=0.5) >> Coherent(1, x=1)
        exp_probs = [
            (coh_state >> Number(0, n0).dual >> Number(1, n1).dual) ** 2
            for n0 in range(10)
            for n1 in range(10)
        ]
        assert math.allclose(sampler.probabilities(coh_state), exp_probs, atol)

    def test_sample(self):
        n_samples = 1000
        sampler = PNRSampler(cutoff=10)

        assert not np.any(sampler.sample(Vacuum(0)))
        assert not np.any(sampler.sample_prob_dist(Vacuum(0))[0])
        assert not np.any(sampler.sample(Vacuum((0, 1))))
        assert not np.any(sampler.sample_prob_dist(Vacuum((0, 1)))[0])

        state = Coherent(0, x=0.1)
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
        assert sampler.povms is None
        assert sampler._phi == 0.5
        assert math.allclose(sampler.meas_outcomes, list(np.linspace(-5, 5, 100)))

    def test_povm_error(self):
        sampler = HomodyneSampler()
        with pytest.raises(ValueError, match="no POVMs"):
            sampler._get_povm(0, 0)

    def test_probabilties(self):
        sampler = HomodyneSampler()

        state = Coherent(0, x=0.1)

        exp_probs = (
            state.quadrature_distribution(math.astensor(sampler.meas_outcomes)) * sampler._step
        )
        assert math.allclose(sampler.probabilities(state), exp_probs)

        sampler2 = HomodyneSampler(phi=np.pi / 2)

        exp_probs = (
            state.quadrature_distribution(math.astensor(sampler2.meas_outcomes), phi=sampler2._phi)
            * sampler2._step
        )
        assert math.allclose(sampler2.probabilities(state), exp_probs)

    def test_probabilities_cat(self):
        state = (Coherent(mode=0, x=2) + Coherent(mode=0, x=-2)).normalize()
        sampler = HomodyneSampler(phi=0, bounds=(-10, 10), num=1000)
        exp_probs = (
            state.quadrature_distribution(math.astensor(sampler.meas_outcomes), phi=sampler._phi)
            * sampler._step
        )
        assert math.allclose(sampler.probabilities(state), exp_probs)

    def test_sample_mean_coherent(self):
        r"""
        Porting test from strawberry fields:
        https://github.com/XanaduAI/strawberryfields/blob/master/tests/backend/test_homodyne.py#L56
        """
        N_MEAS = 300
        NUM_STDS = 10.0
        std_10 = NUM_STDS / np.sqrt(N_MEAS)
        alpha = 1.0 + 1.0j
        tol = settings.ATOL

        state = Coherent(0, x=math.real(alpha), y=math.imag(alpha)) >> Coherent(
            1,
            x=math.real(alpha),
            y=math.imag(alpha),
        )
        sampler = HomodyneSampler()

        meas_result = sampler.sample(state, N_MEAS)
        assert math.allclose(
            meas_result.mean(axis=0),
            settings.HBAR * math.real(math.astensor([alpha, alpha])),
            atol=std_10 + tol,
        )

    def test_sample_mean_and_std_vacuum(self):
        r"""
        Porting test from strawberry fields:
        https://github.com/XanaduAI/strawberryfields/blob/master/tests/backend/test_homodyne.py#L40
        """
        N_MEAS = 300
        NUM_STDS = 10.0
        std_10 = NUM_STDS / np.sqrt(N_MEAS)
        tol = settings.ATOL

        state = Vacuum((0, 1))
        sampler = HomodyneSampler()

        meas_result = sampler.sample(state, N_MEAS)
        assert math.allclose(meas_result.mean(axis=0), [0.0, 0.0], atol=std_10 + tol)
        assert math.allclose(meas_result.std(axis=0), [1.0, 1.0], atol=std_10 + tol)

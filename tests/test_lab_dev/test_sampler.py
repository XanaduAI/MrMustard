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

from itertools import product

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.sampler import Sampler, PNRSampler, HomodyneSampler
from mrmustard.lab_dev import Coherent, Number, Vacuum, BtoQ


class TestSampler:
    r"""
    Tests ``Sampler`` objects.
    """

    def test_init(self):
        meas_outcomes = list(range(3))
        meas_ops = [Number([0], n) for n in range(3)]
        sampler = Sampler(meas_outcomes, meas_ops)
        assert sampler.meas_outcomes == meas_outcomes
        assert sampler.meas_ops == meas_ops

    def test_probabilities(self):
        meas_outcomes = list(range(3))
        meas_ops = [Number([0], n) for n in range(3)]
        sampler = Sampler(meas_outcomes, meas_ops)
        state = Vacuum([0])
        assert all(sampler.probabilities(state) == [1, 0, 0])

        with pytest.raises(ValueError, match="incompatible"):
            sampler_two_mode = Sampler(meas_outcomes, [Number([0, 1], n) for n in range(3)])
            sampler_two_mode.probabilities(state)

    def test_sample(self):
        meas_outcomes = list(range(3))
        meas_ops = [Number([0], n) for n in range(3)]
        sampler = Sampler(meas_outcomes, meas_ops)
        assert all(sampler.sample(Vacuum([0])) == 0)


class TestPNRSampler:
    r"""
    Tests ``PNRSampler`` objects.
    """

    def test_init(self):
        sampler = PNRSampler([0, 1], cutoff=10)
        assert sampler.meas_outcomes == list(product(range(10), repeat=2))
        assert sampler.meas_ops == Number([0, 1], 0)

    def test_probabilities(self):
        atol = 1e-4

        sampler = PNRSampler([0, 1], cutoff=10)
        vac_prob = [1.0] + [0.0] * 99
        assert math.allclose(sampler.probabilities(Vacuum([0, 1])), vac_prob)
        assert math.allclose(sampler.probabilities(Vacuum([0, 1, 2])), vac_prob)

        coh_state = Coherent([0, 1], x=[0.5, 1])
        exp_probs = [
            (coh_state >> Number([0], n0).dual >> Number([1], n1).dual) ** 2
            for n0 in range(10)
            for n1 in range(10)
        ]
        assert math.allclose(sampler.probabilities(coh_state), exp_probs, atol)

        sampler2 = PNRSampler([1], cutoff=10)
        exp_probs2 = [(coh_state >> Number([1], n).dual).probability for n in range(10)]
        assert math.allclose(sampler2.probabilities(coh_state), exp_probs2, atol)


class TestHomodyneSampler:
    r"""
    Tests ``HomodyneSampler`` objects.
    """

    def test_init(self):
        sampler = HomodyneSampler([0, 1], bounds=(-5, 5), num=100)
        assert sampler.meas_ops == BtoQ([0, 1])
        assert math.allclose(
            sampler.meas_outcomes, list(product(np.linspace(-5, 5, 100), repeat=2))
        )

    def test_probabilties(self):
        sampler = HomodyneSampler([0, 1])
        assert math.allclose(
            sampler.probabilities(Vacuum([0, 1])), sampler.probabilities(Vacuum([0, 1, 2]))
        )

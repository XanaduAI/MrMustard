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

"""
Samplers for measurement devices.
"""

from __future__ import annotations
from itertools import product

from abc import ABC, abstractmethod

from typing import Any, Sequence

import numpy as np

from mrmustard import math, settings

from .states import State, Number, QuadratureEigenstate
from .circuit_components import CircuitComponent
from .circuit_components_utils import BtoQ

__all__ = ["Sampler", "PNRSampler", "HomodyneSampler"]


class Sampler(ABC):
    r"""
    A sampler for measurements of quantum circuits.

    Args:
        meas_outcomes: The measurement outcomes for this sampler.
        povms: The POVMs of this sampler.
    """

    def __init__(
        self,
        meas_outcomes: Sequence[Any],
        povms: CircuitComponent | Sequence[CircuitComponent],
    ):
        self._povms = povms
        self._meas_outcomes = meas_outcomes

    @property
    def povms(self) -> CircuitComponent | Sequence[CircuitComponent]:
        r"""
        The POVMs of this sampler.
        """
        return self._povms

    @property
    def meas_outcomes(self) -> Sequence[Any]:
        r"""
        The measurement outcomes of this sampler.
        """
        return self._meas_outcomes

    @abstractmethod
    def probabilities(self, state: State, atol: float = 1e-4) -> Sequence[float]:
        r"""
        Returns the probability distribution of a state w.r.t. measurement outcomes.

        Args:
            state: The state to generate the probability distribution of.
            atol: The absolute tolerance used for validating the computed probability
                distribution.
        """

    def sample(self, state: State, n_samples: int = 1000, seed: int | None = None) -> np.ndarray:
        r"""
        Returns an array of samples given a state.

        Args:
            state: The state to sample.
            n_samples: The number of samples to generate.
            seed: An optional seed for random sampling.
        """
        initial_mode = state.modes[0]
        initial_samples = self.sample_prob_dist(state[initial_mode], n_samples, seed)

        if len(state.modes) == 1:
            return initial_samples

        unique_samples, counts = np.unique(initial_samples, return_counts=True)
        ret = []
        for unique_sample, counts in zip(unique_samples, counts):
            meas_op = self.povms[self.meas_outcomes.index(unique_sample)].on([initial_mode])
            reduced_state = (state >> meas_op.dual).normalize()
            samples = self.sample(reduced_state, counts)
            for sample in samples:
                ret.append(np.append([unique_sample], sample))
        return np.array(ret)

    def sample_prob_dist(
        self, state: State, n_samples: int = 1000, seed: int | None = None
    ) -> np.ndarray:
        r"""
        Samples a a state by computing the probability distribution.

        Args:
            state: The state to sample.
            n_samples: The number of samples to generate.
            seed: An optional seed for random sampling.
        """
        rng = np.random.default_rng(seed) if seed else settings.rng
        return rng.choice(
            a=list(product(self.meas_outcomes, repeat=len(state.modes))),
            p=self.probabilities(state),
            size=n_samples,
        )

    def _validate_probs(self, probs: Sequence[float], dx: float, atol: float) -> Sequence[float]:
        r"""
        Validates that the given probability distribution sums to `1.0` within some
        tolerance and returns a renormalized probability distribution to account for
        small numerical errors.

        Args:
            probs: The probability distribution to validate.
            dx: The uniform differential for the probability distribution.
            atol: The absolute tolerance to validate with.
        """
        atol = atol or settings.ATOL
        probs_dx = probs * dx
        prob_sum = sum(probs_dx)
        if not math.allclose(prob_sum, 1, atol):
            raise ValueError(f"Probabilities sum to {prob_sum} and not 1.0.")
        return math.real(probs_dx / prob_sum)


class PNRSampler(Sampler):
    r"""
    A sampler for photon-number resolving (PNR) detectors.

    Args:
        cutoff: The photon number cutoff.
    """

    def __init__(self, cutoff: int) -> None:
        super().__init__(list(range(cutoff)), [Number([0], n) for n in range(cutoff)])
        self._cutoff = cutoff

    def probabilities(self, state, atol=1e-4):
        fock_state = state.dm().to_fock(self._cutoff)
        probs = math.astensor(
            [
                fock_state.representation.data[0][(ns * 2)]
                for ns in product(self.meas_outcomes, repeat=len(state.modes))
            ]
        )
        return self._validate_probs(probs, 1, atol)


class HomodyneSampler(Sampler):
    r"""
    A sampler for homodyne measurements.

    Args:
        phi: The quadrature angle where ``0`` corresponds to ``x`` and ``\pi/2`` to ``p``.
        bounds: The range of values to discretize over.
        num: The number of points to discretize over.
    """

    def __init__(
        self,
        phi: float = 0,
        bounds: tuple[float, float] = (-10, 10),
        num: int = 1000,
    ) -> None:
        meas_outcomes, step = np.linspace(*bounds, num, retstep=True)
        super().__init__(
            list(meas_outcomes), [QuadratureEigenstate([0], x=x, phi=phi) for x in meas_outcomes]
        )
        self._step = step

    def probabilities(self, state, atol=1e-4):
        q_state = state.dm() >> BtoQ(state.modes, phi=self.povms[0].phi.value[0])
        z = [x * 2 for x in product(self.meas_outcomes, repeat=len(state.modes))]
        probs = q_state.representation(z) * math.sqrt(settings.HBAR)
        return self._validate_probs(probs, self._step ** len(state.modes), atol)

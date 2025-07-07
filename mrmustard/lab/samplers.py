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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from mrmustard import math, settings

from .circuit_components import CircuitComponent
from .circuit_components_utils import BtoQ
from .states import Ket, Number, State

__all__ = ["HomodyneSampler", "PNRSampler", "Sampler"]


class Sampler(ABC):
    r"""
    A sampler for measurements of quantum circuits.

    Args:
        meas_outcomes: The measurement outcomes for this sampler.
        povms: The (optional) POVMs of this sampler.
    """

    def __init__(
        self,
        meas_outcomes: Sequence[Any],
        povms: CircuitComponent | Sequence[CircuitComponent] | None = None,
    ):
        self._povms = povms
        self._meas_outcomes = meas_outcomes
        self._outcome_arg = None

    @property
    def povms(self) -> CircuitComponent | Sequence[CircuitComponent] | None:
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
            state: The state to generate the probability distribution of. Note: the
                input state must be normalized.
            atol: The absolute tolerance used for validating that the computed
                probability distribution sums to ``1``.
        """

    def sample(self, state: State, n_samples: int = 1000, seed: int | None = None) -> np.ndarray:
        r"""
        Returns an array of samples given a state.

        Args:
            state: The state to sample.
            n_samples: The number of samples to generate.
            seed: An optional seed for random sampling.

        Returns:
            An array of samples such that the shape is ``(n_samples, n_modes)``.
        """
        if len(state.modes) == 1:
            return self.sample_prob_dist(state, n_samples, seed)[0]

        initial_mode = state.modes[0]
        initial_samples, probs = self.sample_prob_dist(state[initial_mode], n_samples, seed)

        unique_samples, idxs, counts = np.unique(
            initial_samples,
            return_index=True,
            return_counts=True,
        )
        ret = []
        for unique_sample, idx, count in zip(unique_samples, idxs, counts):
            meas_op = self._get_povm(unique_sample, initial_mode).dual
            prob = probs[idx]
            norm = math.sqrt(prob) if isinstance(state, Ket) else prob
            reduced_state = (state >> meas_op) / norm
            samples = self.sample(reduced_state, count)
            ret.extend(np.append([unique_sample], sample) for sample in samples)
        return np.array(ret)

    def sample_prob_dist(
        self,
        state: State,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Samples a state by computing the probability distribution.

        Args:
            state: The state to sample.
            n_samples: The number of samples to generate.
            seed: An optional seed for random sampling.

        Returns:
            A tuple of the generated samples and the probability
            of obtaining the sample.
        """
        rng = np.random.default_rng(seed) if seed else settings.rng
        probs = self.probabilities(state)
        meas_outcomes = list(product(self.meas_outcomes, repeat=len(state.modes)))
        samples = rng.choice(
            a=meas_outcomes,
            p=probs,
            size=n_samples,
        )
        return samples, np.array([probs[meas_outcomes.index(tuple(sample))] for sample in samples])

    def _get_povm(self, meas_outcome: Any, mode: int) -> CircuitComponent:
        r"""
        Returns the POVM associated with a given outcome on a given mode.

        Args:
            meas_outcome: The measurement outcome.
            mode: The mode.

        Returns:
            The POVM circuit component.

        Raises:
            ValueError: If this sampler has no POVMs.
        """
        if self._povms is None:
            raise ValueError("This sampler has no POVMs defined.")
        if isinstance(self.povms, CircuitComponent):
            kwargs = self.povms.parameters.to_dict()
            kwargs[self._outcome_arg] = meas_outcome
            return self.povms.__class__(mode, **kwargs)
        return self.povms[self.meas_outcomes.index(meas_outcome)].on([mode])

    def _validate_probs(self, probs: Sequence[float], atol: float) -> Sequence[float]:
        r"""
        Validates that the given probability distribution sums to ``1`` within some
        tolerance and returns a renormalized probability distribution to account for
        small numerical errors.

        Args:
            probs: The probability distribution to validate.
            atol: The absolute tolerance to validate with.
        """
        atol = atol or settings.ATOL
        probs = math.abs(probs)
        prob_sum = math.sum(probs)
        math.error_if(
            prob_sum,
            not math.allclose(prob_sum, 1, atol=atol),
            f"Probabilities sum to {prob_sum} and not 1.0.",
        )
        return probs / prob_sum


class PNRSampler(Sampler):
    r"""
    A sampler for photon-number resolving (PNR) detectors.

    Args:
        cutoff: The photon number cutoff.
    """

    def __init__(self, cutoff: int) -> None:
        super().__init__(list(range(cutoff)), Number(0, 0, cutoff))
        self._cutoff = cutoff
        self._outcome_arg = "n"

    def probabilities(self, state, atol=1e-4):
        return self._validate_probs(state.fock_distribution(self._cutoff), atol)


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
        super().__init__(list(meas_outcomes))
        self._step = step
        self._phi = phi

    def probabilities(self, state, atol=1e-4):
        probs = state.quadrature_distribution(
            math.astensor(self.meas_outcomes),
            phi=self._phi,  # TODO: revisit meas_outcomes
        ) * self._step ** len(state.modes)
        return self._validate_probs(probs, atol)

    def sample(self, state: State, n_samples: int = 1000, seed: int | None = None) -> np.ndarray:
        if len(state.modes) == 1:
            return self.sample_prob_dist(state, n_samples, seed)[0]

        initial_mode = state.modes[0]
        initial_samples, probs = self.sample_prob_dist(state[initial_mode], n_samples, seed)

        unique_samples, idxs, counts = np.unique(
            initial_samples,
            return_index=True,
            return_counts=True,
        )
        ret = []
        for unique_sample, idx, count in zip(unique_samples, idxs, counts):
            # Use partial_eval to evaluate the ansatz at the first mode only
            reduced_ansatz = (state >> BtoQ([initial_mode], phi=self._phi)).ansatz(unique_sample)
            reduced_state = state.from_bargmann(state.modes[1:], reduced_ansatz.triple)
            prob = probs[idx] / self._step
            norm = math.sqrt(prob) if isinstance(state, Ket) else prob
            normalized_reduced_state = reduced_state / norm
            samples = self.sample(normalized_reduced_state, count)
            ret.extend(np.append([unique_sample], sample) for sample in samples)
        return np.array(ret)

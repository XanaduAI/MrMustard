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

from typing import Sequence

import numpy as np

from mrmustard import math

from .states import State, Number
from .circuit_components import CircuitComponent
from .circuit_components_utils import BtoQ

__all__ = ["Sampler", "PNRSampler", "HomodyneSampler"]


class Sampler:
    r"""
    A sampler for measurements of quantum circuits.

    Args:
        meas_outcomes: The measurement outcomes for this sampler.
        meas_ops: The optional measurement operators of this sampler.
        probs: An optional probability distribution for this sampler.
    """

    def __init__(
        self,
        meas_outcomes: list[any],
        meas_ops: CircuitComponent | list[CircuitComponent] | None = None,
        probs: list[float] | None = None,
    ):
        self._meas_ops = meas_ops
        self._meas_outcomes = meas_outcomes
        self._probs = probs

    @property
    def meas_ops(self) -> CircuitComponent | list[CircuitComponent]:
        r"""
        The measurement operators of this sampler.
        """
        return self._meas_ops

    @property
    def meas_outcomes(self) -> list[any]:
        r"""
        The measurement outcomes of this sampler.
        """
        return self._meas_outcomes

    def sample(self, state: State, n_samples: int) -> list[any]:
        r"""
        Returns a list of measurement samples on a specified state.

        Args:
            state: The state to generate samples of.
            n_samples: The number of samples to generate.
        """
        rng = np.random.default_rng()
        return rng.choice(a=self._meas_outcomes, p=self.probabilities(state), size=n_samples)

    def probabilities(self, state: State | None = None) -> list[float]:
        r"""
        Returns the probability distribution of this sampler. If ``probs`` was
        not specified when initializing then a probability distribution is generated
        using ``state``.

        Args:
            state: The state to generate the probability distribution with.
        """
        if self._probs is None:
            states = [state >> meas_op.dual for meas_op in self.meas_ops]
            probs = [
                state.probability if isinstance(state, State) else math.real(state) ** 2
                for state in states
            ]
            return probs / sum(probs)
        return self._probs


class PNRSampler(Sampler):
    r"""
    A sampler for photon-number resolving (PNR) detectors.

    Args:
        modes: The measured modes.
        cutoff: The photon number cutoff.
    """

    def __init__(self, modes: Sequence[int], cutoff: int) -> None:
        super().__init__(list(range(cutoff)), [Number(modes, n).dm() for n in range(cutoff)])


class HomodyneSampler(Sampler):
    r"""
    A sampler for homodyne measurements.

    Args:
        modes: The measured modes.
        xbounds: The range of ``x`` values.
        num: The number of measurement outcomes.
    """

    def __init__(
        self, modes: Sequence[int], xbounds: tuple[float, float] = (-5, 5), num: int = 100
    ) -> None:
        super().__init__(list(np.linspace(*xbounds, num)), BtoQ(modes, phi=0))

    def probabilities(self, state: State | None = None):
        if self._probs is None:
            q_state = state >> self.meas_ops
            probs = [math.real(q_state.representation([[q]])[0]) ** 2 for q in self._meas_outcomes]
            probs /= sum(probs)
            return probs
        return self._probs

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

from typing import Sequence, Iterable

import numpy as np

from mrmustard import math, settings

from .states import State, Number
from .circuit_components import CircuitComponent
from .circuit_components_utils import BtoQ, TraceOut

__all__ = ["Sampler", "PNRSampler", "HomodyneSampler"]


class Sampler:
    r"""
    A sampler for measurements of quantum circuits.

    Args:
        meas_outcomes: The measurement outcomes for this sampler.
        meas_ops: The optional measurement operators of this sampler.
        prob_dist: An optional probability distribution for this sampler.
    """

    def __init__(
        self,
        meas_outcomes: list[any],
        meas_ops: CircuitComponent | list[CircuitComponent] | None = None,
        prob_dist: list[float] | None = None,
    ):
        self._meas_ops = meas_ops
        self._meas_outcomes = meas_outcomes
        self._prob_dist = prob_dist

    @property
    def meas_ops(self) -> CircuitComponent | list[CircuitComponent] | None:
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

    @property
    def prob_dist(self) -> list[float] | None:
        r"""
        The probability distribution of this sampler.
        """
        return self._prob_dist

    def probabilities(self, state: State | None = None) -> list[float] | None:
        r"""
        Returns the probability distribution of this sampler. If ``state`` is provided
        then will compute the probability distribution w.r.t. the state.

        Args:
            state: The state to generate the probability distribution with.
        """
        self._validate_state(state)
        if state is not None:
            states = [state.dm() >> meas_op.dual for meas_op in self.meas_ops]
            probs = [
                state.probability if isinstance(state, State) else math.real(state)
                for state in states
            ]
            return probs
        return self.prob_dist

    def sample(self, state: State | None = None, n_samples: int = 1000) -> list[any]:
        r"""
        Returns a list of measurement samples on a specified state. If ``self.probabilities`` is
        ``None`` then uses a uniform probability distribution.

        Args:
            state: The state to generate samples of.
            n_samples: The number of samples to generate.
        """
        rng = settings.rng
        return rng.choice(a=self._meas_outcomes, p=self.probabilities(state), size=n_samples)

    def _validate_state(self, state: State | None = None):
        r"""
        Validates that the modes of ``state`` and ``self.meas_ops`` are compatible with one another.

        Args:
            state: The state to validate.
        """
        if self.meas_ops and state is not None:
            meas_op_modes = (
                self.meas_ops[0].modes
                if isinstance(self.meas_ops, Iterable)
                else self.meas_ops.modes
            )
            if not set(state.modes) >= set(meas_op_modes):
                raise ValueError(
                    f"State with modes {state.modes} is incompatible with the measurement operators with modes {meas_op_modes}."
                )


class PNRSampler(Sampler):
    r"""
    A sampler for photon-number resolving (PNR) detectors.

    Args:
        modes: The measured modes.
        cutoff: The photon number cutoff.
    """

    def __init__(self, modes: Sequence[int], cutoff: int) -> None:
        super().__init__(list(range(cutoff)), [Number(modes, n) for n in range(cutoff)])


class HomodyneSampler(Sampler):
    r"""
    A sampler for homodyne measurements.

    Args:
        modes: The measured modes.
        phi: The quadrature angle where ``0`` corresponds to ``x`` and ``\pi/2`` to ``p``.
        bounds: The range of values to discretize over.
        num: The number of points to discretize over.
    """

    def __init__(
        self,
        modes: Sequence[int],
        phi: float = 0,
        bounds: tuple[float, float] = (-5, 5),
        num: int = 100,
    ) -> None:
        super().__init__(list(np.linspace(*bounds, num)), BtoQ(modes, phi=phi))

    def probabilities(self, state: State | None = None):
        self._validate_state(state)
        if state is not None:
            disjoint_modes = [mode for mode in state.modes if mode not in self.meas_ops.modes]
            dm_state = state.dm() >> TraceOut(disjoint_modes) if disjoint_modes else state.dm()
            q_state = dm_state >> self.meas_ops
            z = [[x] * q_state.representation.ansatz.num_vars for x in self.meas_outcomes]
            probs = math.real(q_state.representation(z)) * math.sqrt(settings.HBAR)
            return probs / sum(probs)
        return self.prob_dist

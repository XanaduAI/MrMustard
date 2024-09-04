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
        meas_ops: The measurement operators of this sampler.
        prob_dist: An optional probability distribution for this sampler.
    """

    def __init__(
        self,
        meas_outcomes: Sequence[any],
        meas_ops: CircuitComponent | Sequence[CircuitComponent],
        prob_dist: Sequence[float] | None = None,
    ):
        self._meas_ops = meas_ops
        self._meas_outcomes = meas_outcomes
        self._prob_dist = prob_dist

    @property
    def meas_ops(self) -> CircuitComponent | Sequence[CircuitComponent]:
        r"""
        The measurement operators of this sampler.
        """
        return self._meas_ops

    @property
    def meas_outcomes(self) -> Sequence[any]:
        r"""
        The measurement outcomes of this sampler.
        """
        return self._meas_outcomes

    @property
    def prob_dist(self) -> Sequence[float] | None:
        r"""
        The probability distribution of this sampler.
        """
        return self._prob_dist

    def probabilities(
        self, state: State | None = None, atol: float = 1e-4
    ) -> Sequence[float] | None:
        r"""
        Returns the probability distribution of this sampler. If ``state`` is provided
        then will compute the probability distribution w.r.t. the state.

        Args:
            state: The state to generate the probability distribution with.
            atol: The absolute tolerance used for validating the computed probability
                distribution.
        """
        self._validate_state(state)
        if state is not None:
            dm_state = state.dm()
            states = [dm_state >> meas_op.dual for meas_op in self.meas_ops]
            probs = [
                state.probability if isinstance(state, State) else math.real(state)
                for state in states
            ]
            return self._validate_probs(probs, 1, atol)
        return self.prob_dist

    def sample(self, state: State | None = None, n_samples: int = 1000) -> np.ndarray:
        r"""
        Returns a list of measurement samples on a specified state. If ``self.probabilities`` is
        ``None`` then uses a uniform probability distribution.

        Args:
            state: The state to generate samples of.
            n_samples: The number of samples to generate.
        """
        rng = settings.rng
        return rng.choice(a=self._meas_outcomes, p=self.probabilities(state), size=n_samples)

    def _trace_modes(self, state: State) -> list[int]:
        r"""
        Computes a list of modes to trace out based on the given state
        and the set of measurement operators.

        Args:
            state: The state to trace out.
        """
        meas_op = self.meas_ops[0] if isinstance(self.meas_ops, Iterable) else self.meas_ops
        return [mode for mode in state.modes if mode not in meas_op.modes]

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
        prob_sum = sum(probs * dx)
        if not math.allclose(prob_sum, 1, atol):
            raise ValueError(f"Probabilities sum to {prob_sum} and not 1.0.")
        return math.real(probs / prob_sum)

    def _validate_state(self, state: State | None = None):
        r"""
        Validates that the modes of ``state`` and ``self.meas_ops`` are compatible with one another.

        Args:
            state: The state to validate.
        """
        if state is not None:
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
        super().__init__(list(product(range(cutoff), repeat=len(modes))), Number(modes, 0))
        self._cutoff = cutoff

    def probabilities(self, state=None, atol=1e-4):
        self._validate_state(state)
        if state:
            fock_state = state.dm().to_fock(self._cutoff)
            probs = math.astensor([self._fock_prob(fock_state, ns) for ns in self.meas_outcomes])
            return self._validate_probs(probs, 1, atol)
        return self._prob_dist

    def _fock_prob(self, fock_state: State, ns: tuple[int, ...]) -> float:
        r"""
        Compute the fock amplitude for a given tuple of photon numbers.
        E.g. (1, 3) computes photon number `1` on the first mode and
        `3` on the second.

        Args:
            fock_state: The state in the Fock representation.
            ns: The photon number tuple.
        """
        trace_modes = self._trace_modes(fock_state)
        trace_wires = fock_state.wires[trace_modes]
        fock_rep = (
            fock_state.representation.trace(trace_wires.bra.indices, trace_wires.ket.indices)
            if trace_modes
            else fock_state.representation
        )
        fock_array = fock_rep.data[0]
        return fock_array[*(ns * 2)]


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
        bounds: tuple[float, float] = (-10, 10),
        num: int = 1000,
    ) -> None:
        meas_outcomes, step = np.linspace(*bounds, num, retstep=True)
        super().__init__(list(product(meas_outcomes, repeat=len(modes))), BtoQ(modes, phi=phi))
        self._step = step

    def probabilities(self, state=None, atol=1e-4):
        self._validate_state(state)
        if state is not None:
            trace_modes = self._trace_modes(state)
            dm_state = state.dm() >> TraceOut(trace_modes) if trace_modes else state.dm()
            q_state = dm_state >> self.meas_ops
            z = [x * 2 for x in self.meas_outcomes]
            probs = q_state.representation(z) * math.sqrt(settings.HBAR)
            return self._validate_probs(probs, self._step**2, atol)
        return self.prob_dist

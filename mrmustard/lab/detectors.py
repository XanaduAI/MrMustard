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

"""
This module implements the set of detector classes that perform measurements on quantum circuits.
"""

from typing import Iterable, List, Optional, Tuple, Union

from mrmustard import settings
from mrmustard.math import Math
from mrmustard.physics import fock, gaussian
from mrmustard.training import Parametrized
from mrmustard.typing import RealMatrix, RealVector

from .abstract import FockMeasurement, Measurement, State
from .gates import Rgate
from .states import Coherent, DisplacedSqueezed

math = Math()

__all__ = ["PNRDetector", "ThresholdDetector", "Generaldyne", "Homodyne", "Heterodyne"]


# pylint: disable=no-member
class PNRDetector(Parametrized, FockMeasurement):
    r"""Photon Number Resolving detector.

    If ``len(modes) > 1`` the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, the parallel instances of the detector share that parameter.

    To apply mode-specific parmeters use a list of floats. The number of modes is determined (in order of priority)
    by the modes parameter, the cutoffs parameter, or the length of the efficiency and dark counts parameters.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    It can be supplied the full stochastic channel, or it will compute it from
    the quantum efficiency (binomial) and the dark count probability (possonian).

    Args:
        efficiency (float or List[float]): list of quantum efficiencies for each detector
        efficiency_trainable (bool): whether the efficiency is trainable
        efficiency_bounds (Tuple[float, float]): bounds for the efficiency
        dark_counts (float or List[float]): list of expected dark counts
        dark_counts_trainable (bool): whether the dark counts are trainable
        dark_counts_bounds (Tuple[float, float]): bounds for the dark counts
        stochastic_channel (Optional 2d array): if supplied, this stochastic_channel will be used for belief propagation
        modes (Optional List[int]): list of modes to apply the detector to
        cutoffs (int or List[int]): largest phton number measurement cutoff for each mode
    """

    def __init__(
        self,
        efficiency: Union[float, List[float]] = 1.0,
        dark_counts: Union[float, List[float]] = 0.0,
        efficiency_trainable: bool = False,
        dark_counts_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_counts_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        stochastic_channel: RealMatrix = None,
        modes: List[int] = None,
        cutoffs: Union[int, List[int]] = None,
    ):
        Parametrized.__init__(
            self,
            efficiency=math.atleast_1d(efficiency),
            dark_counts=math.atleast_1d(dark_counts),
            efficiency_trainable=efficiency_trainable,
            dark_counts_trainable=dark_counts_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_counts_bounds=dark_counts_bounds,
        )

        self._stochastic_channel = stochastic_channel
        self._should_recompute_stochastic_channel = efficiency_trainable or dark_counts_trainable

        if modes is not None:
            num_modes = len(modes)
        elif cutoffs is not None:
            num_modes = len(cutoffs)
        else:
            num_modes = max(len(math.atleast_1d(efficiency)), len(math.atleast_1d(dark_counts)))

        modes = modes or list(range(num_modes))
        outcome = None
        FockMeasurement.__init__(self, outcome, modes, cutoffs)

        self.recompute_stochastic_channel()

    def should_recompute_stochastic_channel(self):
        return self._should_recompute_stochastic_channel

    def recompute_stochastic_channel(self, cutoffs: List[int] = None):
        """recompute belief using the defined `stochastic channel`"""
        if cutoffs is None:
            cutoffs = [settings.PNR_INTERNAL_CUTOFF] * len(self._modes)
        self._internal_stochastic_channel = []
        if self._stochastic_channel is not None:
            self._internal_stochastic_channel = self._stochastic_channel
        else:
            efficiency = (
                math.tile(math.atleast_1d(self.efficiency.value), [len(cutoffs)])
                if len(math.atleast_1d(self.efficiency.value)) == 1
                else self.efficiency.value
            )
            dark_counts = (
                math.tile(math.atleast_1d(self.dark_counts.value), [len(cutoffs)])
                if len(math.atleast_1d(self.dark_counts.value)) == 1
                else self.dark_counts.value
            )
            for c, qe, dc in zip(cutoffs, efficiency, dark_counts):
                dark_prior = math.poisson(max_k=settings.PNR_INTERNAL_CUTOFF, rate=dc)
                condprob = math.binomial_conditional_prob(
                    success_prob=qe, dim_in=settings.PNR_INTERNAL_CUTOFF, dim_out=c
                )
                self._internal_stochastic_channel.append(
                    math.convolve_probs_1d(
                        condprob, [dark_prior, math.eye(settings.PNR_INTERNAL_CUTOFF)[0]]
                    )
                )


# pylint: disable: no-member
class ThresholdDetector(Parametrized, FockMeasurement):
    r"""Threshold detector: any Fock component other than vacuum counts toward a click in the detector.

    If ``len(modes) > 1`` the detector is applied in parallel to all of the modes provided.

    If a parameter is a single float, its value is applied to all of the parallel instances of the detector.

    To apply mode-specific values use a list of floats.

    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (bernoulli).

    Args:
        efficiency (float or List[float]): list of quantum efficiencies for each detector
        dark_count_prob (float or List[float]): list of dark count probabilities for each detector
        efficiency_trainable (bool): whether the efficiency is trainable
        dark_count_prob_trainable (bool): whether the dark count probabilities are trainable
        efficiency_bounds (Tuple[float, float]): bounds for the efficiency
        dark_count_prob_bounds (Tuple[float, float]): bounds for the dark count probabilities
        stochastic_channel (Optional 2d array): if supplied, this stochastic_channel will be used for belief propagation
        modes (Optional List[int]): list of modes to apply the detector to
    """

    def __init__(
        self,
        efficiency: Union[float, List[float]] = 1.0,
        dark_count_prob: Union[float, List[float]] = 0.0,
        efficiency_trainable: bool = False,
        dark_count_prob_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_count_prob_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        stochastic_channel=None,
        modes: List[int] = None,
    ):
        if modes is not None:
            num_modes = len(modes)
        else:
            num_modes = max(len(math.atleast_1d(efficiency)), len(math.atleast_1d(dark_count_prob)))

        if len(math.atleast_1d(efficiency)) == 1 and num_modes > 1:
            efficiency = math.tile(math.atleast_1d(efficiency), [num_modes])
        if len(math.atleast_1d(dark_count_prob)) == 1 and num_modes > 1:
            dark_count_prob = math.tile(math.atleast_1d(dark_count_prob), [num_modes])

        modes = modes or list(range(num_modes))

        Parametrized.__init__(
            self,
            efficiency=efficiency,
            dark_count_prob=dark_count_prob,
            efficiency_trainable=efficiency_trainable,
            dark_count_prob_trainable=dark_count_prob_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_count_prob_bounds=dark_count_prob_bounds,
        )

        self._stochastic_channel = stochastic_channel

        cutoffs = [2] * num_modes
        self._should_recompute_stochastic_channel = (
            efficiency_trainable or dark_count_prob_trainable
        )

        outcome = None
        FockMeasurement.__init__(self, outcome, modes, cutoffs)

        self.recompute_stochastic_channel()

    def should_recompute_stochastic_channel(self):
        return self._should_recompute_stochastic_channel

    def recompute_stochastic_channel(self, cutoffs: List[int] = None):
        """recompute belief using the defined `stochastic channel`"""
        if cutoffs is None:
            cutoffs = [settings.PNR_INTERNAL_CUTOFF] * len(self._modes)
        self._internal_stochastic_channel = []
        if self._stochastic_channel is not None:
            self._internal_stochastic_channel = self._stochastic_channel
        else:
            for cut, qe, dc in zip(
                cutoffs,
                math.atleast_1d(self.efficiency.value)[:],
                math.atleast_1d(self.dark_count_prob.value)[:],
            ):
                row1 = math.pow(1.0 - qe, math.arange(cut)[None, :]) - math.cast(
                    dc, self.efficiency.value.dtype
                )
                row2 = 1.0 - row1
                # rest = math.zeros((cut - 2, cut), dtype=row1.dtype)
                condprob = math.concat([row1, row2], axis=0)
                self._internal_stochastic_channel.append(condprob)


class Generaldyne(Measurement):
    r"""Generaldyne measurement on given modes.

    Args:
        state (State): the Gaussian state of the measurment device
        outcome (optional or List[float]): the means of the measurement state, defaults to ``None``
        modes (List[int]): the modes on which the measurement is acting on
    """

    def __init__(
        self,
        state: State,
        outcome: Optional[RealVector] = None,
        modes: Optional[Iterable[int]] = None,
    ) -> None:
        if not state.is_gaussian:
            raise TypeError("Generaldyne measurement state must be Gaussian.")
        if outcome is not None and not outcome.shape == state.means.shape:
            raise TypeError(
                f"Expected `outcome` of size {state.means.shape} but got {outcome.shape}."
            )

        self.state = state

        if modes is None:
            modes = self.state.modes
        else:
            # ensure measurement and state act on the same modes
            self.state = state[modes]

        super().__init__(outcome, modes)

    @property
    def outcome(self) -> RealVector:
        return self.state.means

    def primal(self, other: State) -> Union[State, float]:
        if self.postselected:
            # return the projection of self.state onto other
            return self.state.primal(other)

        return super().primal(other)

    def _measure_gaussian(self, other) -> Union[State, float]:
        remaining_modes = list(set(other.modes) - set(self.modes))

        outcome, prob, new_cov, new_means = gaussian.general_dyne(
            other.cov, other.means, self.state.cov, None, modes=self.modes
        )
        self.state = State(cov=self.state.cov, means=outcome)

        return (
            prob
            if len(remaining_modes) == 0
            else State(cov=new_cov, means=new_means, modes=remaining_modes, _norm=prob)
        )

    def _measure_fock(self, other) -> Union[State, float]:
        raise NotImplementedError(f"Fock sampling not implemented for {self.__class__.__name__}")


class Heterodyne(Generaldyne):
    r"""Heterodyne measurement on given modes.

    This class is just a thin wrapper around the :class:`Coherent`.
    If neither ``x`` or ``y`` is provided then values will be sampled.

    Args:
        x (optional float or List[float]): the x-displacement of the coherent state, defaults to ``None``
        y (optional or List[float]): the y-displacement of the coherent state, defaults to ``None``
        modes (List[int]): the modes of the coherent state
    """

    def __init__(
        self,
        x: Union[float, List[float]] = 0.0,
        y: Union[float, List[float]] = 0.0,
        modes: List[int] = None,
    ):
        if (x is None) ^ (y is None):  # XOR
            raise ValueError("Both `x` and `y` arguments should be defined or set to `None`.")

        # if no x and y provided, sample the outcome
        if x is None and y is None:
            num_modes = len(modes) if modes is not None else 1
            x, y = math.zeros([num_modes]), math.zeros([num_modes])
            outcome = None
        else:
            x = math.atleast_1d(x, dtype="float64")
            y = math.atleast_1d(y, dtype="float64")
            outcome = math.concat([x, y], axis=0)  # XXPP ordering

        modes = modes or list(range(x.shape[0]))

        units_factor = math.sqrt(2.0 * settings.HBAR, dtype="float64")
        state = Coherent(x / units_factor, y / units_factor)
        super().__init__(state=state, outcome=outcome, modes=modes)


class Homodyne(Generaldyne):
    """Homodyne measurement on given modes. If ``result`` is not provided then the value
    is sampled.

    Args:
        quadrature_angle (float or List[float]): measurement quadrature angle
        result (optional float or List[float]): displacement amount
        modes (optional List[int]): the modes of the displaced squeezed state
        r (optional float or List[float]): squeezing amount (default: ``settings.HOMODYNE_SQUEEZING``)
    """

    def __init__(
        self,
        quadrature_angle: Union[float, List[float]],
        result: Optional[Union[float, List[float]]] = None,
        modes: Optional[List[int]] = None,
        r: Optional[Union[float, List[float]]] = None,
    ):
        self.r = r or settings.HOMODYNE_SQUEEZING
        self.quadrature_angle = math.atleast_1d(quadrature_angle, dtype="float64")

        # if no ``result`` provided, sample the outcome
        if result is None:
            x = math.zeros_like(self.quadrature_angle)
            y = math.zeros_like(self.quadrature_angle)
            outcome = None
        else:
            result = math.atleast_1d(result, dtype="float64")
            if result.shape[-1] == 1:
                result = math.tile(result, self.quadrature_angle.shape)

            x = result * math.cos(self.quadrature_angle)
            y = result * math.sin(self.quadrature_angle)
            outcome = math.concat([x, y], axis=0)  # XXPP ordering

        modes = modes or list(range(self.quadrature_angle.shape[0]))

        units_factor = math.sqrt(2.0 * settings.HBAR, dtype="float64")
        state = DisplacedSqueezed(
            r=self.r, phi=2 * self.quadrature_angle, x=x / units_factor, y=y / units_factor
        )
        super().__init__(state=state, outcome=outcome, modes=modes)

    def _measure_gaussian(self, other) -> Union[State, float]:
        # rotate modes to be measured to the Homodyne basis
        other >>= Rgate(-self.quadrature_angle, modes=self.modes)
        self.state >>= Rgate(-self.quadrature_angle, modes=self.modes)

        # perform homodyne measurement as a generaldyne one
        out = super()._measure_gaussian(other)

        # set p-outcomes to 0 and rotate the measurement device state back to the original basis,
        # this is in turn rotating the outcomes to the original basis
        self_state_means = math.concat(
            [self.state.means[: self.num_modes], math.zeros((self.num_modes,))], axis=0
        )
        self.state = State(cov=self.state.cov, means=self_state_means, modes=self.modes) >> Rgate(
            self.quadrature_angle, modes=self.modes
        )

        return out

    def _measure_fock(self, other) -> Union[State, float]:
        if len(self.modes) > 1:
            raise NotImplementedError(
                "Multimode Homodyne sampling for Fock representation is not yet implemented."
            )

        other_cutoffs = [
            None if m not in self.modes else other.cutoffs[other.indices(m)] for m in other.modes
        ]
        remaining_modes = list(set(other.modes) - set(self.modes))

        # create reduced state of modes to be measured on the homodyne basis
        reduced_state = other.get_modes(self.modes)

        # build pdf and sample homodyne outcome
        x_outcome, probability = fock.sample_homodyne(
            state=reduced_state.ket() if reduced_state.is_pure else reduced_state.dm(),
            quadrature_angle=self.quadrature_angle,
        )

        # Define conditional state of the homodyne measurement device and rotate back to the original basis.
        # Note: x_outcome already has units of sqrt(hbar). Here is divided by sqrt(2*hbar) to cancel the multiplication
        # factor of the displacement symplectic inside the DisplacedSqueezed state.
        x_arg = x_outcome / math.sqrt(2.0 * settings.HBAR, dtype="float64")
        self.state = DisplacedSqueezed(
            r=self.r, phi=0.0, x=x_arg, y=0.0, modes=self.modes
        ) >> Rgate(self.quadrature_angle, modes=self.modes)

        if remaining_modes == 0:
            return probability

        self_cutoffs = [other.cutoffs[other.indices(m)] for m in self.modes]
        other_cutoffs = [
            None if m not in self.modes else other.cutoffs[other.indices(m)] for m in other.modes
        ]
        out_fock = fock.contract_states(
            stateA=other.ket(other_cutoffs) if other.is_pure else other.dm(other_cutoffs),
            stateB=self.state.ket(self_cutoffs),
            a_is_dm=other.is_mixed,
            b_is_dm=False,
            modes=other.indices(self.modes),
            normalize=False,
        )

        return (
            State(dm=out_fock, modes=remaining_modes, _norm=probability)
            if other.is_mixed
            else State(ket=out_fock, modes=remaining_modes, _norm=probability)
        )

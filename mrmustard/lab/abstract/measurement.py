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

from __future__ import annotations
from abc import ABC, abstractmethod
from mrmustard.physics import gaussian, fock
from mrmustard.lab.abstract.state import State
from mrmustard.types import *
from mrmustard.utils import graphics
from mrmustard import settings
import numpy as np


class GaussianMeasurement(ABC):
    r"""
    Base class for all Gaussian measurements.
    """

    def __call__(self, state: State, **kwargs) -> Tuple[Scalar, State]:
        r"""
        Applies a general-dyne Gaussian measurement to the state, i.e. it projects
        onto the state with given cov and outcome means vector.
        Args:
            state (State): the state to be measured.
            kwargs (optional): same arguments as in the init, use them only if they are different
            from the arguments supplied at init time (e.g. for training a measurement using a state to project onto).
        Returns:
            (float, state) The measurement probabilities and the remaining post-measurement state.
            Note that the post-measurement state is trivial if all modes are measured.
        """
        if len(kwargs) > 0:
            self._project_onto = self.recompute_project_onto(**kwargs)
        prob, cov, means = gaussian.general_dyne(
            state.cov,
            state.means,
            self._project_onto.cov,
            self._project_onto.means,
            self._modes,
            settings.HBAR,
        )
        remaining_modes = [m for m in range(state.num_modes) if m not in self._modes]

        if len(remaining_modes) > 0:
            remaining_state = State(cov=cov, means=means)
            return prob, remaining_state
        else:
            return prob

    def recompute_project_onto(self, **kwargs) -> State:
        ...

    def __getitem__(self, items) -> Callable:
        r"""
        Allows measurements to be used as:
        output = meas[0,1](input)  # e.g. measuring modes 0 and 1
        """
        if isinstance(items, int):
            modes = [items]
        elif isinstance(items, slice):
            modes = list(range(items.start, items.stop, items.step))
        elif isinstance(items, (Sequence, Iterable)):
            modes = list(items)
        else:
            raise ValueError(f"{items} is not a valid slice or list of modes.")
        self.modes = modes
        return self


# TODO: push all math methods into the physics module?
class FockMeasurement(ABC):
    r"""
    A Fock measurement projecting onto a Fock measurement pattern.
    It works by representing the state in the Fock basis and then applying
    a stochastic channel matrix P(meas|n) to the Fock probabilities (belief propagation).
    It outputs the measurement probabilities and the remaining post-measurement state (if any)
    in the Fock basis.
    """

    def project(
        self, state: State, cutoffs: Sequence[int], measurement: Sequence[Optional[int]]
    ) -> Tuple[State, Tensor]:
        r"""
        Projects the state onto a Fock measurement in the form [a,b,c,...] where integers
        indicate the Fock measurement on that mode and None indicates no projection on that mode.

        Returns the measurement probability and the renormalized state (in the Fock basis) in the unmeasured modes.
        """
        if (len(cutoffs) != state.num_modes) or (len(measurement) != state.num_modes):
            raise ValueError(
                "the length of cutoffs/measurements does not match the number of modes"
            )
        dm = state.dm(cutoffs=cutoffs)
        measured = 0
        for mode, (stoch, meas) in enumerate(zip(self._stochastic_channel, measurement)):
            if meas is not None:
                # put both indices last and compute sum_m P(meas|m)rho_mm for every meas
                last = [mode - measured, mode + state.num_modes - 2 * measured]
                perm = list(set(range(dm.ndim)).difference(last)) + last
                dm = fock.math.transpose(dm, perm)
                dm = fock.math.diag_part(dm)
                dm = fock.math.tensordot(dm, stoch[meas, : dm.shape[-1]], [[-1], [0]])
            measured += 1
        probs = fock.math.sum(fock.math.all_diagonals(dm, real=False))
        return dm / probs, fock.math.abs(probs)

    def apply_stochastic_channel(self, stochastic_channel, fock_probs: Tensor) -> Tensor:
        cutoffs = [fock_probs.shape[m] for m in self._modes]
        for i, mode in enumerate(self._modes):
            if cutoffs[mode] > stochastic_channel[i].shape[1]:
                raise IndexError(
                    f"Internal cutoff ({stochastic_channel[i].shape[1]}) too low in mode {mode} (state cutoff {cutoffs[mode]}).\nYou can increase max_input_photons or reduce the cutoff of the state."
                )
        detector_probs = fock_probs
        for i, mode in enumerate(self._modes):
            detector_probs = fock.math.tensordot(
                detector_probs,
                stochastic_channel[i][: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            detector_probs = fock.math.transpose(
                detector_probs, indices[:mode] + [fock_probs.ndim - 1] + indices[mode:]
            )
        return detector_probs

    def __call__(
        self, state: State, cutoffs: List[int], outcomes: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Tensor, Tensor]:
        if outcomes is None:
            fock_probs = state.fock_probabilities(cutoffs)
            return self.apply_stochastic_channel(self._stochastic_channel, fock_probs)
        else:
            return self.project(state, cutoffs, outcomes)

    def recompute_stochastic_channel(self, **kwargs) -> State:
        ...

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
from mrmustard.math import Math

math = Math()
from mrmustard.physics import fock
from mrmustard.lab.abstract.state import State
from mrmustard.types import *
from mrmustard.utils import graphics
from mrmustard import settings
import numpy as np


class FockMeasurement(ABC):
    r"""A Fock measurement projecting onto a Fock measurement pattern.

    It works by representing the state in the Fock basis and then applying a stochastic channel
    matrix ``P(meas|n)`` to the Fock probabilities (belief propagation).

    It outputs the measurement probabilities and the remaining post-measurement state (if any)
    in the Fock basis.
    """

    def primal(self, state: State) -> Tensor:
        r"""
        Returns a tensor representing the post-measurement state in the unmeasured modes in the Fock basis.
        The first N indices of the returned tensor correspond to the Fock measurements of the N modes that
        the detector is measuring. The remaining indices correspond to the density matrix of the unmeasured modes.
        """
        cutoffs = []
        used = 0
        for mode in state.modes:
            if mode in self._modes:
                cutoffs.append(
                    max(settings.PNR_INTERNAL_CUTOFF, state.cutoffs[state.indices(mode)])
                )
                used += 1
            else:
                cutoffs.append(state.cutoffs[state.indices(mode)])
        if self.should_recompute_stochastic_channel() or math.any(
            [c > settings.PNR_INTERNAL_CUTOFF for c in state.cutoffs]
        ):
            self.recompute_stochastic_channel(cutoffs)
        dm = state.dm(cutoffs)
        for k, (mode, stoch) in enumerate(zip(self._modes, self._internal_stochastic_channel)):
            # move the mode indices to the end
            last = [mode - k, mode + state.num_modes - 2 * k]
            perm = [m for m in range(dm.ndim) if m not in last] + last
            dm = math.transpose(dm, perm)
            # compute sum_m P(meas|m)rho_mm
            dm = math.diag_part(dm)
            dm = math.tensordot(dm, stoch[: self._cutoffs[k], : dm.shape[-1]], [[-1], [1]])
        # put back the last len(self.modes) modes at the beginning
        output = math.transpose(
            dm,
            list(range(dm.ndim - len(self._modes), dm.ndim))
            + list(range(dm.ndim - len(self._modes))),
        )
        if len(output.shape) == len(self._modes):  # all modes are measured
            output = math.real(output)  # return probabilities
        return output

    def should_recompute_stochastic_channel(self) -> bool:  # override in subclasses
        return False

    def __lshift__(self, other) -> Tensor:
        if isinstance(other, State):
            self.primal(other)
        else:
            raise TypeError(
                f"unsupported operand type(s) '{type(self).__name__}' << '{type(other).__name__}'"
            )

    def __getitem__(self, items) -> Callable:
        r"""
        Allows measurements to be used as ``output = meas[0,1](input)``, e.g. measuring modes 0
        and 1.
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

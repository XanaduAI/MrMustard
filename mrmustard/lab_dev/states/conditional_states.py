# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=abstract-method

"""
The classes representing states conditional on the output of a measurement.
"""

from __future__ import annotations

from typing import Sequence, Any

import mrmustard.math as math

from .base import State

__all__ = [
    "ConditionalState",
]

class ConditionalState(State):
    r"""
    A state dependent on the outcomes of a measurement.

    Args:

    """
    def __init__(self, modes: Sequence[int], meas_outcomes: Sequence[Any], states: Sequence[State], name: str | None = None):

        self._state_outcomes = dict(zip(meas_outcomes, states))
        self._state = None

        # not sure about modes?
        super().__init__(modes_in_ket=modes, modes_out_ket=modes, name= name or "ConditionalState")

    @property
    def state(self) -> State | None:
        r"""
        """
        return self._state
    
    @property
    def state_outcomes(self) -> dict[Any:State]:
        r"""
        """
        return self._state_outcomes
    
    @property
    def representation(self):
        r"""
        """
        # once we get the meas outcome batch dim working then it should return a representation
        # with all possible states if no state is set
        return self._state.representation if self._state else None
    
    def get_batched(self):
        r"""
        """
        return math.astensor([state.representation.data for state in self.state_outcomes.values()])

    def set_state(self, meas_outcome):
        r"""
        """
        self._state = self._state_outcomes[meas_outcome]

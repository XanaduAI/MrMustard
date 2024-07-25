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
        modes: The modes of the conditional state.
        meas_outcomes: A list of measurement outcomes.
        states: The states conditioned on some measured outcome.
        name: The name of this conditional state.
    """

    def __init__(
        self,
        modes: Sequence[int],
        meas_outcomes: Sequence[Any],
        states: Sequence[State],
        name: str | None = None,
    ):

        self._state_outcomes = dict(zip(meas_outcomes, states))
        self._meas_outcomes = None

        # not sure about modes?
        super().__init__(modes_out_ket=modes, name=name or "ConditionalState")

    @property
    def meas_outcomes(self) -> Any | None:
        r""" """
        return self._meas_outcomes

    @meas_outcomes.setter
    def meas_outcomes(self, value):
        self._meas_outcomes = value

    @property
    def state(self):
        r""" """
        try:
            return self.state_outcomes[self.meas_outcomes]
        except KeyError:
            return None

    @property
    def state_outcomes(self) -> dict[Any:State]:
        r""" """
        return self._state_outcomes

    @property
    def representation(self):
        r""" """
        # once we get the meas outcome batch dim working then it should return a representation
        # with all possible states if no state is set
        try:
            return self.state.representation
        except AttributeError:
            return None

    def get_batched(self):
        r""" """
        return math.astensor([state.representation.data for state in self.state_outcomes.values()])

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

"""This module contains the implementation of the class :class:`FockMeasurement`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Union

from mrmustard import math, settings
from mrmustard.math.parameter_set import ParameterSet
from mrmustard.math.parameters import Constant, Variable
from mrmustard.utils.typing import Tensor
from .state import State


class Measurement(ABC):
    """this is an abstract class holding the common methods and properties that any measurement should
    implement

    Args:
        outcome (optional, List[float] or Tensor): the result of the measurement
        modes (List[int]): the modes on which the measurement is acting on
    """

    def __init__(self, outcome: Tensor, modes: Sequence[int]) -> None:
        super().__init__()

        if modes is None:
            raise ValueError(f"Modes not defined for {self.__class__.__name__}.")
        self._modes = modes

        self._is_postselected = False if outcome is None else True
        """used to evaluate if the measurement outcome should be
        sampled or is already defined by the user (postselection)"""

        self._parameter_set = ParameterSet()

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to a transformation.

        Args:
            parameter: The parameter to add.
        """
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    @property
    def parameter_set(self):
        r"""
        The set of parameters for this transformation.
        """
        return self._parameter_set

    @property
    def modes(self):
        r"""returns the modes being measured"""
        return self._modes

    @property
    def num_modes(self):
        r"""returns the number of modes being measured"""
        return len(self.modes)

    @property
    def postselected(self):
        r"""returns whether the measurement is postselected, i.e, a outcome has been provided"""
        return self._is_postselected

    @property
    @abstractmethod
    def outcome(self):
        """Returns outcome of the measurement. If no measurement has been carried out returns `None`."""
        ...

    @abstractmethod
    def _measure_fock(self, other: State) -> Union[State, float]:
        ...

    @abstractmethod
    def _measure_gaussian(self, other: State) -> Union[State, float]:
        ...

    def primal(self, other: State) -> Union[State, float]:
        """performs the measurement procedure according to the representation of the incoming state"""
        if other.is_gaussian:
            return self._measure_gaussian(other)

        return self._measure_fock(other)

    def __lshift__(self, other) -> Union[State, float]:
        if isinstance(other, State):
            self.primal(other)

        raise TypeError(
            f"Cannot apply Measurement '{self.__qualname__}' to '{other.__qualname__}'."
        )

    def __getitem__(self, items) -> Measurement:
        """Assign modes via the getitem syntax: allows measurements to be used as
        ``output = meas[0,1](input)``, e.g. measuring modes 0 and 1.
        """
        if isinstance(items, int):
            modes = [items]
        elif isinstance(items, slice):
            modes = list(range(items.start, items.stop, items.step))
        elif isinstance(items, (Sequence, Iterable)):
            modes = list(items)
        else:
            raise ValueError(f"{items} is not a valid slice or list of modes.")
        self._modes = modes

        return self


class FockMeasurement(Measurement):
    """A Fock measurement projecting onto a Fock measurement pattern.

    It works by representing the state in the Fock basis and then applying a stochastic channel
    matrix ``P(meas|n)`` to the Fock probabilities (belief propagation).

    It outputs the measurement probabilities and the remaining post-measurement state (if any)
    in the Fock basis.
    """

    def __init__(
        self, outcome: Tensor, modes: Sequence[int], cutoffs: Sequence[int]
    ) -> None:
        self._cutoffs = cutoffs or [settings.PNR_INTERNAL_CUTOFF] * len(modes)
        super().__init__(outcome, modes)

    @property
    def outcome(self):
        raise NotImplementedError

    def _measure_gaussian(self, other: State) -> Union[State, float]:
        return self._measure_fock(other)

    def _measure_fock(self, other: State) -> Union[State, float]:
        r"""
        Returns a tensor representing the post-measurement state in the unmeasured modes in the Fock basis.
        The first `N` indices of the returned tensor correspond to the Fock measurements of the `N` modes that
        the detector is measuring. The remaining indices correspond to the density matrix of the unmeasured modes.

        Args
            other (State): the quantum state
        Returns
            Tensor: a tensor representing the post-measurement state
        """
        cutoffs = []
        for mode in other.modes:
            if mode in self._modes:
                cutoffs.append(
                    max(
                        settings.PNR_INTERNAL_CUTOFF, other.cutoffs[other.indices(mode)]
                    )
                )
            else:
                cutoffs.append(other.cutoffs[other.indices(mode)])
        if self.should_recompute_stochastic_channel() or math.any(
            [c > settings.PNR_INTERNAL_CUTOFF for c in other.cutoffs]
        ):
            self.recompute_stochastic_channel(cutoffs)
        dm = other.dm(cutoffs)
        for k, (mode, stoch) in enumerate(
            zip(self._modes, self._internal_stochastic_channel)
        ):
            # move the mode indices to the end
            last = [mode - k, mode + other.num_modes - 2 * k]
            perm = [m for m in range(dm.ndim) if m not in last] + last
            dm = math.transpose(dm, perm)
            # compute sum_m P(meas|m)rho_mm
            dm = math.diag_part(dm)
            dm = math.tensordot(
                dm, stoch[: self._cutoffs[k], : dm.shape[-1]], [[-1], [1]]
            )
        # put back the last len(self.modes) modes at the beginning
        output = math.transpose(
            dm,
            list(range(dm.ndim - len(self._modes), dm.ndim))
            + list(range(dm.ndim - len(self._modes))),
        )
        if len(output.shape) == len(self._modes):  # all modes are measured
            output = math.real(output)  # return probabilities
        return output

    #  pylint: disable=no-self-use
    def should_recompute_stochastic_channel(self) -> bool:  # override in subclasses
        """Returns `True` if the stochastic channel has to be recomputed.

        This method should be overriden by subclasses as needed.
        """
        return False

    def recompute_stochastic_channel(self, cutoffs: Sequence[int]) -> None:
        """Recomputes the stochastic channel.

        This method should be overriden by subclasses as needed.
        """
        raise NotImplementedError

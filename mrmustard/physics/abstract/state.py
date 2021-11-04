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
from mrmustard.utils.types import *
from mrmustard.utils import graphics
from mrmustard import settings
import numpy as np


class State:
    r"""Base class for quantum states"""

    def __init__(self, mixed: bool = None, cov=None, means=None, fock=None):
        r"""
        Initializes the state. Either supply the cov,means pair of the fock tensor.
        Arguments:
            mixed (bool): whether the state is mixed
            cov (Matrix): the covariance matrix
            means (Vector): the means vector
            fock (Array): the Fock representation
        """
        _covmeans = cov is not None and means is not None
        _fock = fock is not None
        if not _covmeans and not _fock:
            raise ValueError("either cov and means or fock must be supplied")
        self._num_modes = None
        self._is_mixed = bool(mixed)
        self._fock = fock
        self._means = means
        self._cov = cov

    @classmethod
    def from_gaussian(cls, cov: Matrix, means: Vector, mixed: bool) -> State:
        r"""
        Returns a state from a Gaussian distribution.
        Arguments:
            cov (Matrix): the covariance matrix
            means (Vector): the means vector
            mixed (bool): whether the state is mixed
        Returns:
            State: the state
        """
        return cls(mixed=mixed, cov=cov, means=means)

    @classmethod
    def from_fock(cls, fock: Array, mixed: bool) -> State:
        r"""
        Returns a state from a Fock representation.
        Arguments:
            fock (Array): the Fock representation
            mixed (bool): whether the state is mixed
        Returns:
            State: the state
        """
        return cls(mixed=mixed, fock=fock)

    @property
    def is_gaussian(self) -> bool:
        r"""
        Returns `True` if the state is Gaussian.
        """
        return self.cov is not None and self.means is not None

    @property
    def num_modes(self) -> int:
        r"""
        Returns the number of modes in the state.
        """
        if self._num_modes is None:
            if self._fock is not None:
                num_indices = len(self._fock.shape)
                self._num_modes = num_indices if self.is_pure else num_indices // 2
            else:
                self._num_modes = self.means.shape[-1] // 2
        return self._num_modes

    @property
    def is_pure(self):
        return not self.is_mixed

    @property
    def is_mixed(self):
        return self._is_mixed

    @property
    def means(self) -> Optional[Vector]:
        r"""
        Returns the means vector of the state.
        """
        return self._means

    @property
    def cov(self) -> Optional[Matrix]:
        r"""
        Returns the covariance matrix of the state.
        """
        return self._cov

    @property
    def fock(self) -> Optional[Array]:
        r"""
        Returns the Fock representation of the state.
        """
        if self._fock is None:
            self._fock = fock.fock_representation(self.cov, self.means, mixed=self.is_mixed)
        return self._fock

    @property
    def cutoffs(self) -> List[int]:
        r"""
        Returns the cutoff dimensions for each mode.
        """
        if self.is_gaussian:
            return None
        else:
            return [int(s) for s in self.fock.shape]

    @property
    def number_means(self) -> Vector:
        r"""
        Returns the mean photon number for each mode
        """
        try:
            return gaussian.number_means(self.cov, self.means, settings.HBAR)
        except ValueError:
            return gaussian.number_means(self._fock)

    @property
    def number_cov(self) -> Matrix:
        r"""
        Returns the complete photon number covariance matrix
        """
        try:
            return gaussian.number_cov(self.cov, self.means, settings.HBAR)
        except ValueError:
            return gaussian.number_cov(self._fock)

    def ket(self, cutoffs: Sequence[int]) -> Optional[Tensor]:
        r"""
        Returns the ket of the state in Fock representation or `None` if the state is mixed.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the ket
        """
        if self.is_pure:
            if self.means is not None:  # TODO: this can trigger recomputation of means: find a better way
                self._fock = fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False)
            return self._fock
        else:
            return None

    def dm(self, cutoffs: List[int]) -> Tensor:
        r"""
        Returns the density matrix of the state in Fock representation.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the density matrix
        """
        if self.is_pure:
            ket = self.ket(cutoffs=cutoffs)
            return fock.ket_to_dm(ket)
        else:
            if self.means is not None:  # TODO: this may trigger recomputation of means: find a better way
                self._fock = fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=True)
            return self._fock

    def fock_probabilities(self, cutoffs: Sequence[int]) -> Tensor:
        r"""
        Returns the probabilities in Fock representation. If the state is pure, they are
        the absolute value squared of the ket amplitudes. If the state is mixed they are
        the diagonals of the density matrix.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Array: the probabilities
        """
        if self.is_mixed:
            dm = self.dm(cutoffs=cutoffs)
            return fock.dm_to_probs(dm)
        else:
            ket = self.ket(cutoffs=cutoffs)
            return fock.ket_to_probs(ket)

    def __and__(self, other: State) -> State:
        r"""
        Concatenates two states.
        """
        cov = gaussian.join_covs([self.cov, other.cov])
        means = gaussian.join_means([self.means, other.means])
        return State.from_gaussian(cov, means, self.is_mixed or other.is_mixed)

    def __getitem__(self, item):
        r"""
        Returns the state on the given modes.
        """
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")
        cov, _, _ = gaussian.partition_cov(self.cov, item)
        means, _ = gaussian.partition_means(self.means, item)
        return State.from_gaussian(cov, means, gaussian.is_mixed_cov(cov))

    def __eq__(self, other):
        r"""
        Returns whether the states are equal.
        """
        if not np.allclose(self.means, other.means):
            return False
        if not np.allclose(self.cov, other.cov):
            return False
        return True

    def __repr__(self):
        info = f"num_modes={self.num_modes} | pure={self.is_pure} | representation={'gaussian' if self.is_gaussian else 'fock'}\n"
        detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n" if settings.DEBUG else ""
        if self.num_modes == 1:
            if self._fock is not None:
                cutoffs = self._fock.shape if self.is_pure else self._fock.shape[:1]
            else:
                cutoffs = [20]
            graphics.mikkel_plot(self.dm(cutoffs=cutoffs))
        return info + "-" * len(info) + detailed_info

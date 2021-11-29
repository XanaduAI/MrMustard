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
from mrmustard.math import Math

math = Math()

from mrmustard.utils.types import *
from mrmustard.utils import graphics
from mrmustard import settings
import numpy as np
from rich.table import Table
from rich import print as rprint
from mrmustard.utils.xptensor import XPMatrix, XPVector


class State:
    r"""Base class for quantum states"""

    def __init__(self, cov: Matrix = None, means: Vector = None, fock: Array = None, is_mixed: bool = None):
        r"""
        Initializes the state. Either supply the fock tensor or the cov and means.
        Arguments:
            cov (Matrix): the covariance matrix
            means (Vector): the means vector
            fock (Array): the Fock representation
            is_mixed (optional, bool): whether the state is mixed
        """

        _fock = fock is not None
        if (cov is None or means is None) and not _fock:
            raise ValueError("Please supply either (S, eig), (cov and means) or fock")
        self._num_modes = len(means) // 2 if means is not None else (len(fock.shape) // 2 if is_mixed else len(fock.shape))
        self._is_mixed = is_mixed
        self._purity = 1.0 if not is_mixed else None
        self._fock = fock
        self._cov = cov
        self._means = means
        self._fock_probabilities = None
        self._cutoffs = None
        self._eigenvalues = None

    @property
    def is_mixed(self):
        if self._is_mixed is None:
            self._is_mixed = gaussian.is_mixed_cov(self.cov)
        return self._is_mixed

    @property
    def is_pure(self):
        return not self.is_mixed

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
    def is_gaussian(self) -> bool:
        r"""
        Returns `True` if the state is Gaussian.
        """
        return self._cov is not None and self._means is not None

    @property
    def num_modes(self) -> int:
        r"""
        Returns the number of modes in the state.
        """
        if self._num_modes is None:
            if self.is_gaussian:
                self._num_modes = self._means.shape[-1] // 2
            else:
                num_indices = len(self._fock.shape)
                self._num_modes = num_indices if self.is_pure else num_indices // 2
        return self._num_modes

    @property
    def purity(self) -> float:
        r"""
        Returns the purity of the state.
        """
        if self._purity is None:
            if self.is_gaussian:
                self._purity = gaussian.purity(self.cov, settings.HBAR)
            else:
                self._purity = fock.purity(self._fock)  # dm
        return self._purity

    @property
    def modes(self) -> List[int]:
        r"""
        Returns the modes of the state.
        By default states are in modes 0, ..., num_modes-1
        """
        try:
            if self._modes is None:
                self._modes = list(range(self.num_modes))
            return self._modes
        except AttributeError:
            return list(range(self.num_modes))

    @property
    def number_stdev(self) -> Vector:
        r"""
        Returns the square root of the photon number variances
        (standard deviation) in each mode.
        """
        if self.is_gaussian:
            return math.sqrt(math.diag_part(self.number_cov))
        else:
            return math.sqrt(fock.number_variances(self.fock, is_dm=self.is_mixed))

    @property
    def cutoffs(self) -> List[int]:
        r"""
        Returns the cutoff dimensions for each mode.
        """
        if self._fock is None:
            return fock.autocutoffs(self.number_stdev, self.number_means)
        else:
            if self._cutoffs is not None:
                return self._cutoffs
            return [s for s in self._fock.shape[: self.num_modes]]

    @property
    def shape(self) -> List[int]:
        r"""
        Returns the shape of the state, accounting for ket/dm representation.
        If the state is in Gaussian representation, the shape is inferred from
        the first two moments of the number operator.
        """
        return self.cutoffs if self.is_pure else self.cutoffs + self.cutoffs

    @property
    def fock(self) -> Array:
        r"""
        Returns the Fock representation of the state.
        """
        if self._fock is None:
            self._fock = fock.fock_representation(self.cov, self.means, shape=self.shape, return_dm=self.is_mixed)
        return self._fock

    @property
    def number_means(self) -> Vector:
        r"""
        Returns the mean photon number for each mode.
        """
        if self.is_gaussian:
            return gaussian.number_means(self.cov, self.means, settings.HBAR)
        else:
            return fock.number_means(tensor=self.fock, is_dm=self.is_mixed)

    @property
    def number_cov(self) -> Matrix:
        r"""
        Returns the complete photon number covariance matrix.
        """
        if self.is_gaussian:
            return gaussian.number_cov(self.cov, self.means, settings.HBAR)
        else:
            raise NotImplementedError("number_cov not implemented for non-gaussian states")

    def ket(self, cutoffs: Sequence[Optional[int]]) -> Optional[Tensor]:
        r"""
        Returns the ket of the state in Fock representation or `None` if the state is mixed.
        Arguments:
            cutoffs List[int or None]: the cutoff dimensions for each mode. If a mode cutoff is None,
                it's guessed automatically.
        Returns:
            Tensor: the ket
        """
        if cutoffs is None:
            cutoffs = self.cutoffs
        else:
            cutoffs = [c if c is not None else self.cutoffs[i] for i, c in enumerate(cutoffs)]
        if self.is_mixed:
            return None
        if self.is_gaussian:
            self._fock = fock.fock_representation(self.cov, self.means, shape=cutoffs, return_dm=False)
        else:  # only fock representation is available
            current_cutoffs = [s for s in self._fock.shape[: self.num_modes]]
            if cutoffs != current_cutoffs:
                paddings = [(0, max(0, new - old)) for new, old in zip(cutoffs, current_cutoffs)]
                if any(p != (0, 0) for p in paddings):
                    padded = fock.math.pad(self._fock, paddings, mode="constant")
                else:
                    padded = self._fock
                return padded[tuple([slice(s) for s in cutoffs])]
        return self._fock

    def dm(self, cutoffs: List[int] = None) -> Tensor:
        r"""
        Returns the density matrix of the state in Fock representation.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode. If a mode cutoff is None,
                it's automatically computed.
        Returns:
            Tensor: the density matrix
        """
        if cutoffs is None:
            cutoffs = self.cutoffs
        else:
            cutoffs = [c if c is not None else self.cutoffs[i] for i, c in enumerate(cutoffs)]
        if self.is_pure:
            ket = self.ket(cutoffs=cutoffs)
            return fock.ket_to_dm(ket)
        else:
            if self.is_gaussian:
                self._fock = fock.fock_representation(self.cov, self.means, shape=cutoffs * 2, return_dm=True)
            elif cutoffs != (current_cutoffs := [s for s in self._fock.shape[: self.num_modes]]):
                paddings = [(0, max(0, new - old)) for new, old in zip(cutoffs, current_cutoffs)]
                if any(p != (0, 0) for p in paddings):
                    padded = fock.math.pad(self._fock, paddings + paddings, mode="constant")
                else:
                    padded = self._fock
                return padded[tuple([slice(s) for s in cutoffs + cutoffs])]
        return self._fock

    def fock_probabilities(self, cutoffs: Sequence[int]) -> Tensor:
        r"""
        Returns the probabilities in Fock representation.
        If the state is pure, they are the absolute value squared of the ket amplitudes.
        If the state is mixed they are the multi-dimensional diagonals of the density matrix.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Array: the probabilities
        """
        if self._fock_probabilities is None:
            if self.is_mixed:
                dm = self.dm(cutoffs=cutoffs)
                self._fock_probabilities = fock.dm_to_probs(dm)
            else:
                ket = self.ket(cutoffs=cutoffs)
                self._fock_probabilities = fock.ket_to_probs(ket)
        return self._fock_probabilities

    def __call__(self, other: Union[State, Transformation]) -> State:
        r"""
        Returns the post-measurement state after `other` is projected onto `self`:
        self(state) -> state projected onto self.
        If `other` is a `Transformation`, it returns the dual of the transformation applied to `self`:
        self(transformation) -> transformation^dual(self)
        """
        if issubclass(other.__class__, State):
            remaining_modes = [m for m in range(other.num_modes) if m not in self._modes]

            if self.is_gaussian and other.is_gaussian:
                prob, cov, means = gaussian.general_dyne(other.cov, other.means, self.cov, self.means, self._modes, settings.HBAR)
                if len(remaining_modes) > 0:
                    return State(means=means, cov=cov, is_mixed=gaussian.is_mixed_cov(cov))
                else:
                    return prob
            else:  # either self or other is not gaussian
                other_cutoffs = []
                used = 0
                for m in range(other.num_modes):
                    if m in self._modes:
                        other_cutoffs.append(self.cutoffs[used])
                        used += 1
                    else:
                        other_cutoffs.append(other.cutoffs[m])
                try:
                    out_fock = self.__preferred_projection(other, other_cutoffs, self._modes)
                except AttributeError:
                    other_fock = other.ket(other_cutoffs) if other.is_pure else other.dm(other_cutoffs)
                    self_cutoffs = [other_cutoffs[m] for m in range(self.num_modes)]
                    self_fock = self.ket(self_cutoffs) if self.is_pure else self.dm(self_cutoffs)
                    out_fock = fock.contract_states(
                        stateA=other_fock,
                        stateB=self_fock if self.is_pure else self.dm(self_cutoffs),
                        a_is_mixed=other.is_mixed,
                        b_is_mixed=self.is_mixed,
                        modes=self._modes,
                        normalize=self._normalize,
                    )
                if len(remaining_modes) > 0:
                    output_is_mixed = not (self.is_pure and other.is_pure)  # TODO: this may fail?
                    return State(
                        fock=out_fock if self._normalize == False else fock.normalize(out_fock, is_mixed=output_is_mixed),
                        is_mixed=output_is_mixed,
                    )
                else:
                    return fock.math.abs(out_fock) ** 2 if other.is_pure and self.is_pure else fock.math.abs(out_fock)
        else:
            try:
                return other.dual_channel(self)
            except AttributeError:
                raise TypeError(f"Cannot apply {other.__class__.__qualname__} to {self.__class__.__qualname__}")

    def __and__(self, other: State) -> State:
        r"""
        Concatenates two states.
        """
        if self.is_gaussian and other.is_gaussian:
            cov = gaussian.join_covs([self.cov, other.cov])
            means = gaussian.join_means([self.means, other.means])
            return State(cov=cov, means=means, is_mixed=self.is_mixed or other.is_mixed)
        else:
            fock_joined = fock.join_focks([self.fock, other.fock])  # TODO: write this method
            return State(fock=fock_joined, is_mixed=self.is_mixed or other.is_mixed)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")
        if len(item) != self.num_modes:
            raise ValueError(f"there are {self.num_modes} modes (item has {len(item)} elements, perhaps you're looking for .get_modes()?)")
        self._modes = item
        return self

    def get_modes(self, item):
        r"""
        Returns the state on the given modes.
        """
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")
        if self.is_gaussian:
            cov, _, _ = gaussian.partition_cov(self.cov, item)
            means, _ = gaussian.partition_means(self.means, item)
            return State(cov=cov, means=means, is_mixed=gaussian.is_mixed_cov(cov))
        else:
            fock_partitioned = fock.trace(self.dm(self.cutoffs), [m for m in range(self.num_modes) if m not in item])
            return State(fock=fock_partitioned, is_mixed=fock.is_mixed_dm(fock_partitioned))

    def __eq__(self, other):
        r"""
        Returns whether the states are equal.
        """
        if self.num_modes != other.num_modes:
            return False
        if self.purity != other.purity:
            return False
        if self.is_gaussian and other.is_gaussian:
            if not np.allclose(self.means, other.means, atol=1e-6):
                return False
            if not np.allclose(self.cov, other.cov, atol=1e-6):
                return False
            return True
        if self.is_pure and other.is_pure:
            return np.allclose(self.ket(cutoffs=other.cutoffs), other.ket(cutoffs=other.cutoffs), atol=1e-6)
        else:
            return np.allclose(self.dm(cutoffs=other.cutoffs), other.dm(cutoffs=other.cutoffs), atol=1e-6)

    def __rshift__(self, other):
        r"""
        Applies other (a Transformation) to self (a State).
        e.g. Coherent(x=0.1) >> Sgate(r=0.1)
        """
        if issubclass(other.__class__, State):
            raise TypeError(
                f"Cannot apply {other.__class__.__qualname__} to a state.\nBut we can project a state on a state: are you looking for the << operator?"
            )
        return other(self)

    def __lshift__(self, other):
        r"""
        Implements
        e.g. Dgate << psi
        """
        return other(self)

    def __add__(self, other: State):
        r"""
        Implements a mixture of states.
        """
        if not isinstance(other, State):
            raise TypeError(f"Cannot add {other.__class__.__qualname__} to a state")
        return State(fock=self.dm(self.cutoffs) + other.dm(self.cutoffs), is_mixed=True)  # TODO: gaussian implementation

    def __rmul__(self, other):
        r"""
        Implements multiplication by a scalar from the right.
        e.g. 0.5 * psi
        """
        if not isinstance(other, (int, float)):
            raise TypeError(f"Cannot multiply {other.__class__.__qualname__} by a state")
        if self.is_gaussian:
            return State(cov=self.cov * other, means=self.means, is_mixed=True)
        else:
            return State(fock=self.dm() * other, is_mixed=True)

    def __repr__(self):
        table = Table(title=str(self.__class__.__qualname__))
        table.add_column("Purity", justify="center")
        table.add_column("Num modes", justify="center")
        table.add_column("Bosonic size", justify="center")
        table.add_column("Gaussian", justify="center")
        table.add_column("Fock", justify="center")
        table.add_row(
            f"{(self.purity):.3f}",
            str(self.num_modes),
            "1" if self.is_gaussian else "N/A",
            "✅" if self.is_gaussian else "❌",
            "✅" if self._fock is not None else "❌",
        )
        rprint(table)
        if self.num_modes == 1:
            graphics.mikkel_plot(self.dm(cutoffs=self.cutoffs))
        detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n" if settings.DEBUG else " "
        return detailed_info

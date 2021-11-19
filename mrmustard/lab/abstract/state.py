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
from rich.table import Table
from rich import print as rprint
from mrmustard.utils.xptensor import XPMatrix, XPVector


class State:
    r"""Base class for quantum states"""

    def __init__(
        self, cov: Matrix = None, means: Vector = None, fock: Array = None, is_mixed: bool = None
    ):
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
        self._num_modes = None
        self._is_mixed = is_mixed
        self._purity = 1.0 if not is_mixed else None
        self._fock = fock
        self._means = means
        self._fock_probabilities = None
        self._cutoffs = None
        self.__maybe_modes = None
        self._eigenvalues = None

    @property
    def symplectic(self) -> XPMatrix:
        r"""
        Returns the symplectic matrix of the state.
        """
        return self._symplectic

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
    def is_pure(self):
        return not self.is_mixed

    @property
    def is_mixed(self):
        if self._is_mixed is None:
            self._is_mixed = gaussian.is_mixed_cov(self.cov)
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
    def cutoffs(self) -> List[int]:
        r"""
        Returns the cutoff dimensions for each mode.
        """
        if self._fock is None:
            return None  # TODO: fock.autocutoffs(self.number_cov, self.number_means)  NOTE: <-- should we?
        else:
            return [s for s in self._fock.shape[: self.num_modes]]

    @property
    def number_means(self) -> Vector:
        r"""
        Returns the mean photon number for each mode.
        """
        if self.is_gaussian:
            return gaussian.number_means(self.cov, self.means, settings.HBAR)
        else:
            raise NotImplementedError("number_means not implemented for non-gaussian states")

    @property
    def number_cov(self) -> Matrix:
        r"""
        Returns the complete photon number covariance matrix.
        """
        if self.is_gaussian:
            return gaussian.number_cov(self.cov, self.means, settings.HBAR)
        else:
            raise NotImplementedError("number_cov not implemented for non-gaussian states")

    def ket(self, cutoffs: Sequence[int], from_cache=False) -> Optional[Tensor]:
        r"""
        Returns the ket of the state in Fock representation or `None` if the state is mixed.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the ket
        """
        if self.is_mixed:
            return None
        if from_cache and self._fock is not None:
            return self._fock
        if self.is_gaussian:
            self._fock = fock.fock_representation(
                self.cov, self.means, shape=cutoffs, is_mixed=False
            )
        else:  # only fock representation is available
            if cutoffs != self.cutoffs:
                try:
                    shape = cutoffs if self.is_pure else cutoffs * 2
                    shape_tuple = [slice(s) for s in shape]
                    return self._fock.__getitem__(*shape_tuple)
                except IndexError:
                    raise IndexError(
                        f"This state in Fock representation does not have a ket of shape {shape}"
                    )
        return self._fock

    def dm(self, cutoffs: List[int], from_cache=False) -> Tensor:
        r"""
        Returns the density matrix of the state in Fock representation.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the density matrix
        """
        if from_cache and self._fock is not None:
            return self._fock
        if self.is_pure:
            ket = self.ket(cutoffs=cutoffs)
            self._fock = fock.ket_to_dm(ket)
        else:
            if self.is_gaussian:
                self._fock = fock.fock_representation(
                    self.cov, self.means, shape=cutoffs * 2, is_mixed=True
                )
            elif cutoffs != self.cutoffs:
                try:
                    shape_tuple = [slice(s) for s in cutoffs * 2]  # NOTE: we know it's mixed
                    return self._fock.__getitem__(shape_tuple)
                except IndexError:
                    raise IndexError(f"This state does not have amplitudes of shape {shape}")
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

    def __call__(self, other: State) -> State:
        r"""
        Returns the post-measurement state after other is projected onto self.
        I.e. self acts as a measurement.
        """
        if isinstance(other, State):
            if self.is_gaussian:
                Sinv = XPMatrix(gaussian.symplectic_inverse(other.symplectic), modes=([], []))
            self.symplectic = Sinv @ self.symplectic
        else:
            raise TypeError(f"Cannot project {type(other)} onto {type(self)}")

    def __and__(
        self, other: State
    ) -> State:  # TODO: keep lazy variables when mixed-representation is supported
        r"""
        Concatenates two states.
        """
        if self.is_gaussian and other.is_gaussian:
            cov = gaussian.join_covs([self.cov, other.cov])
            means = gaussian.join_means([self.means, other.means])
            return State(cov=cov, means=means, is_mixed=self.is_mixed or other.is_mixed)
        else:
            fock = fock.join_focks([self.fock, other.fock])  # TODO: write this method
            return State(fock=fock, is_mixed=self.is_mixed or other.is_mixed)

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
        self.__maybe_modes = item
        cov, _, _ = gaussian.partition_cov(self.cov, item)
        means, _ = gaussian.partition_means(self.means, item)
        return State(cov=cov, means=means, is_mixed=gaussian.is_mixed_cov(cov))

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
            return np.allclose(
                self.ket(cutoffs=other.cutoffs), other.ket(cutoffs=other.cutoffs), atol=1e-6
            )
        else:
            print("here")
            return np.allclose(
                self.dm(cutoffs=other.cutoffs), other.dm(cutoffs=other.cutoffs), atol=1e-6
            )

    def __rshift__(self, other):
        r"""
        Implements piping a state through a transformation or a measurement.
        e.g. psi >> Dgate or psi >> Coherent
        """
        return other(self)

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
            if self._fock is not None:
                cutoffs = self._fock.shape if self.is_pure else self._fock.shape[:1]
            else:
                cutoffs = [20]
            graphics.mikkel_plot(self.dm(cutoffs=cutoffs))
        detailed_info = (
            f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n" if settings.DEBUG else " "
        )
        return detailed_info

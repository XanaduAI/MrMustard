from __future__ import annotations
from mrmustard._typing import *
from mrmustard.plugins import fock, gaussian, graphics
from mrmustard.experimental import XPMatrix, XPVector
from numpy import allclose


class State:
    def __init__(self, hbar: float, mixed: bool = None, cov=None, means=None, fock=None):
        self._num_modes = None
        self._hbar = float(hbar)
        if mixed is not None:
            self.is_mixed: bool = mixed
        self._fock = fock
        self._means = means
        self._cov = cov

    @classmethod
    def from_gaussian(cls, cov: Matrix, means: Vector, mixed: bool, hbar: float = 2.0) -> State:
        r"""
        Returns a state from a Gaussian distribution.
        Arguments:
            cov Matrix: the covariance matrix
            means Vector: the means vector
            mixed bool: whether the state is mixed
            hbar float: the hbar value
        Returns:
            State: the state
        """
        return cls(hbar, mixed, cov, means)

    @classmethod
    def from_fock(cls, fock: Tensor, mixed: bool) -> State:
        r"""
        Returns a state from a Fock representation.
        Arguments:
            fock Tensor: the Fock representation
            mixed bool: whether the state is mixed
            hbar float: the hbar value
        Returns:
            State: the state
        """
        return cls(hbar=2.0, mixed=mixed, fock=fock)

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
    def number_means(self):
        r"""
        Returns the mean photon number for each mode
        """
        try:
            return gaussian.number_means(self.cov, self.means, self._hbar)
        except ValueError:
            return fock.number_means(self._fock, self._hbar)

    @property
    def number_cov(self):
        r"""
        Returns the complete photon number covariance matrix
        """
        try:
            return gaussian.number_cov(self.cov, self.means, self._hbar)
        except ValueError:
            return fock.number_cov(self._fock, self._hbar)

    def ket(self, cutoffs: Sequence[int]) -> Optional[Tensor]:
        r"""
        Returns the ket of the state in Fock representation or `None` if the state is mixed.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the ket
        """
        if self.is_pure:
            if self.means is not None:  # TODO: this may trigger recomputation of means: find a better way
                self._fock = fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False, hbar=self._hbar)
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
                self._fock = fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=True, hbar=self._hbar)
            return self._fock

    def fock_probabilities(self, cutoffs: Sequence[int]) -> Tensor:
        r"""
        Returns the probabilities in Fock representation. If the state is pure, they are
        the absolute value squared of the ket amplitudes. If the state is mixed they are
        the diagonals of the density matrix.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the probabilities
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
        if self._hbar != other._hbar:
            raise ValueError("States must have the same hbar")
        cov = gaussian.join_covs([self.cov, other.cov])
        means = gaussian.join_means([self.means, other.means])
        return State.from_gaussian(cov, means, self.is_mixed or other.is_mixed, self._hbar)

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
        return State.from_gaussian(cov, means, gaussian.is_mixed_cov(cov), self._hbar)

    def __eq__(self, other):
        r"""
        Returns whether the states are equal.
        """
        if self._hbar != other._hbar:
            return False
        if not allclose(self.means, other.means):
            return False
        if not allclose(self.cov, other.cov):
            return False
        return True

    def __repr__(self):
        info = f"num_modes={self.num_modes} | hbar={self._hbar} | pure={self.is_pure}\n"
        detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n"
        if self.num_modes == 1:
            if self._fock is not None:
                cutoffs = self._fock.shape if self.is_pure else self._fock.shape[:1]
            else:
                cutoffs = [20]
            graphics.mikkel_plot(self.dm(cutoffs=cutoffs))
        return info + "-" * len(info) + detailed_info

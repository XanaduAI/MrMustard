from abc import ABC
from typing import List
from mrmustard.core.backends import MathBackendInterface, StateBackendInterface


class State(ABC):
    _math_backend: MathBackendInterface
    _state_backend: StateBackendInterface

    def __init__(self, num_modes: int, hbar: float = 2.0, mixed=False):
        self.num_modes = num_modes
        self.hbar = hbar
        self.mixed = mixed

    def __repr__(self):
        return "covariance:\n" + repr(self.cov) + "\nmeans:\n" + repr(self.means)

    def ket(self, cutoffs: List[int]):
        r"""
        Returns the ket of the state in Fock representation.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        """
        if not self.mixed:
            A, B, C = self._state_backend.ABC(self.cov, self.means, mixed=self.mixed, hbar=self.hbar)
            return self._state_backend.fock_state(A, B, C, cutoffs=cutoffs)

    def dm(self, cutoffs: List[int]):
        r"""
        Returns the density matrix of the state in Fock representation.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        """
        A, B, C = self._state_backend.ABC(self.cov, self.means, mixed=self.mixed, hbar=self.hbar)
        fock_state = self._state_backend.fock_state(A, B, C, cutoffs=cutoffs)
        if self.mixed:
            return fock_state
        else:
            return self._math_backend.outer(self._math_backend.conj(fock_state), fock_state)

    def fock_probabilities(self, cutoffs: List[int]):
        r"""
        Returns the probabilities in Fock representation. If the state is pure, they are
        the absolute value squared of the ket amplitudes. If the state is mixed they are
        the diagonals of the density matrix.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        """
        if self.mixed:
            rho = self.dm(cutoffs=cutoffs)
            return self._math_backend.all_diagonals(rho, real=True)
        else:
            psi = self.ket(cutoffs=cutoffs)
            return self._math_backend.abs(psi) ** 2

    @property
    def number_means(self):
        r"""
        Returns the mean photon number for each mode
        """
        return self._state_backend.number_means(self.cov, self.means, self.hbar)

    @property
    def number_cov(self):
        r"""
        Returns the complete photon number covariance matrix
        """
        return self._state_backend.number_cov(self.cov, self.means, self.hbar)

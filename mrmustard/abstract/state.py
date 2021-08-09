from abc import ABC
from mrmustard.typing import *
from mrmustard.plugins import FockPlugin, GaussianPlugin

class State(ABC):
    _fock: FockPlugin
    _gaussian: GaussianPlugin

    def __init__(self, num_modes: int, hbar: float, mixed: bool):
        self.num_modes = num_modes
        self.hbar = hbar
        self.mixed = mixed
        self.cov: Matrix = None
        self.means: Vector = None

    def __repr__(self):
        return "covariance:\n" + repr(self.cov) + "\nmeans:\n" + repr(self.means)

    def ket(self, cutoffs: Sequence[int]) -> Optional[Tensor]:
        r"""
        Returns the ket of the state in Fock representation or `None` if the state is mixed.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the ket
        """
        if not self.mixed:
            return self._fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False, hbar=self.hbar)

    def dm(self, cutoffs: List[int]) -> Tensor:
        r"""
        Returns the density matrix of the state in Fock representation.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the density matrix
        """
        return self._fock.dm(self.cov, self.means, cutoffs=cutoffs, mixed=self.mixed, hbar=self.hbar)

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
        if self.mixed:
            dm = self.dm(cutoffs=cutoffs)
            return self._fock.dm_to_probs(dm, real=True)
        else:
            ket = self.ket(cutoffs=cutoffs)
            return self._fock.ket_to_probs(ket, real=True)

    @property
    def number_means(self):
        r"""
        Returns the mean photon number for each mode
        """
        return self._fock.number_means(self.cov, self.means, self.hbar)

    @property
    def number_cov(self):
        r"""
        Returns the complete photon number covariance matrix
        """
        return self._fock.number_cov(self.cov, self.means, self.hbar)

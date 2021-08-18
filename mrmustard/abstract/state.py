from mrmustard._typing import *
from mrmustard import FockPlugin, GaussianPlugin


class State:  # NOTE: this is not an ABC
    _fock = FockPlugin()
    _gaussian = GaussianPlugin()

    def __init__(self, num_modes: int, hbar: float, mixed: bool, cov: Optional[Matrix] = None, means: Optional[Vector] = None):
        if cov is not None:
            assert cov.shape[-1] == num_modes * 2
            assert cov.shape[-2] == num_modes * 2
        self.cov: Optional[Matrix] = cov
        if means is not None:
            assert means.shape[-1] == num_modes * 2
        self.means: Optional[Vector] = means

        self.num_modes = num_modes
        self.hbar = hbar
        self.isMixed: bool = mixed

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
        if not self.isMixed:
            return self._fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False, hbar=self.hbar)
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
        if not self.isMixed:
            ket = self._fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False, hbar=self.hbar)
            return self._fock.ket_to_dm(ket)
        else:
            return self._fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=True, hbar=self.hbar)

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
        if self.isMixed:
            dm = self.dm(cutoffs=cutoffs)
            return self._fock.dm_to_probs(dm)
        else:
            ket = self.ket(cutoffs=cutoffs)
            return self._fock.ket_to_probs(ket)

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

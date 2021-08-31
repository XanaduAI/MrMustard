from mrmustard._typing import *
from mrmustard.plugins import fock, gaussian


class State:  # NOTE: this is not an ABC
    def __init__(self, hbar: float, mixed: bool, cov: Optional[Matrix] = None, means: Optional[Vector] = None):
        self.cov: Optional[Matrix] = cov
        self.means: Optional[Vector] = means
        self.num_modes = cov.shape[-1] // 2 if cov is not None else 0
        self.hbar = hbar
        self.isMixed: bool = mixed

    @property
    def isPure(self):
        return not self.isMixed

    def __repr__(self):
        info = f"num_modes={self.num_modes} | hbar={self.hbar} | pure={self.isPure}\n"
        detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n"
        if self.num_modes <= 4:
            return info + "-" * len(info) + detailed_info
        else:
            return info

    def ket(self, cutoffs: Sequence[int]) -> Optional[Tensor]:
        r"""
        Returns the ket of the state in Fock representation or `None` if the state is mixed.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the ket
        """
        if not self.isMixed:
            return fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False, hbar=self.hbar)
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
            ket = fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False, hbar=self.hbar)
            return fock.ket_to_dm(ket)
        else:
            return fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=True, hbar=self.hbar)

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
            return fock.dm_to_probs(dm)
        else:
            ket = self.ket(cutoffs=cutoffs)
            return fock.ket_to_probs(ket)

    @property
    def number_means(self):
        r"""
        Returns the mean photon number for each mode
        """
        return fock.number_means(self.cov, self.means, self.hbar)

    @property
    def number_cov(self):
        r"""
        Returns the complete photon number covariance matrix
        """
        return fock.number_cov(self.cov, self.means, self.hbar)

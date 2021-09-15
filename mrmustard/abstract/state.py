from mrmustard._typing import *
from mrmustard.functionality import fock, gaussian


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

    def concat(self, other: State):
        r"""
        Returns a new state that is the concatenation of this state and the other state.
        The new state is a mixed state.
        Arguments:
            other State: the other state
        Returns:
            State: the concatenated state
        """
        if self.hbar != other.hbar:
            raise ValueError("hbar must be the same for both states")
        if self.isPure != other.isPure:
            raise ValueError("states must be either mixed or pure, but not a combination")
        cov = gaussian.join_covs(self.cov, other.cov)
        means = gaussian.join_means(self.means, other.means)
        return State(self.hbar, mixed=True, cov=cov, means=means)

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

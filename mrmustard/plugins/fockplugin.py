from mrmustard.backends import BackendInterface
from mrmustard.typing import *


class FockPlugin:
    r"""
    A plugin that interfaces the phase space representation with the Fock representation.

    It implements:
    - fock_representation and its gradient
    - number cov and means
    - classical stochastic channels
    """
    backend: BackendInterface

    # ~~~~~~
    # states
    # ~~~~~~

    def number_means(self, cov: Matrix, means: Vector, hbar: float) -> Vector:
        r"""
        Returns the photon number means vector
        given a Wigenr covariance matrix and a means vector.
        Args:
            cov: The Wigner covariance matrix.
            means: The Wigner means vector.
            hbar: The value of the Plank constant.
        Returns:
            The photon number means vector.
        """
        N = means.shape[-1] // 2
        return (means[:N] ** 2 + means[N:] ** 2
                + self.backend.diag_part(cov[:N, :N])
                + self.backend.diag_part(cov[N:, N:]) - hbar
                ) / (2 * hbar)

    def number_cov(self, cov: Matrix, means: Vector, hbar: float) -> Matrix:
        r"""
        Returns the photon number covariance matrix
        given a Wigenr covariance matrix and a means vector.
        Args:
            cov: The Wigner covariance matrix.
            means: The Wigner means vector.
            hbar: The value of the Plank constant.
        Returns:
            The photon number covariance matrix.
        """
        N = means.shape[-1] // 2
        mCm = cov * means[:, None] * means[None, :]
        dd = self.backend.diag(
            self.backend.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])
            ) / (2 * hbar ** 2)
        CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
        return CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * self.backend.eye(N)

    def fock_representation(self, A: Matrix, B: Vector, C: Scalar, cutoffs: Sequence[int]) -> Tensor:
        r"""
        Returns the fock state given the A, B, C matrices and a list of cutoff indices.
        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            cutoffs: The cutoff indices.
        Returns:
            The fock state.
        """
        mixed = len(B) == 2 * len(cutoffs)
        cutoffs_minus_1 = tuple([c - 1 for c in cutoffs] + [c - 1 for c in cutoffs] * mixed)
        state = self.backend.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed, dtype=self.backend.complex128)
        state[(0,) * (len(cutoffs) + len(cutoffs) * mixed)] = C
        state = self.backend.fill_amplitudes(state, A, B, cutoffs_minus_1)  # implemented in MathBackendInterface
        return state

    def grad_fock_representation(self,
                                 dout: Tensor,
                                 state: Tensor,
                                 A: Matrix,
                                 B: Vector,
                                 C: Scalar,
                                 cutoffs: Sequence[int]
                                 ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Returns the gradient of the fock state given the A, B, C matrices and a list of cutoff indices.
        Args:
            dout: The output gradient.
            state: The fock state.
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            cutoffs: The cutoff indices.
        Returns:
            The gradient of the fock state.
        """
        mixed = len(B) == 2 * len(cutoffs)
        dA = self.backend.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + A.shape, dtype=self.backend.complex128)
        dB = self.backend.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + B.shape, dtype=self.backend.complex128)
        cutoffs_minus_1 = tuple([c - 1 for c in cutoffs] + [c - 1 for c in cutoffs] * mixed)
        dA, dB = self.backend.fill_gradients(dA, dB, state, A, B, cutoffs_minus_1)  # implemented in MathBackendInterface
        dC = state / C
        dLdA = self.backend.sum(dout[..., None, None] * self.backend.conj(dA), axis=tuple(range(dout.ndim)))
        dLdB = self.backend.sum(dout[..., None] * self.backend.conj(dB), axis=tuple(range(dout.ndim)))
        dLdC = self.backend.sum(dout * self.backend.conj(dC), axis=tuple(range(dout.ndim)))
        return dLdA, dLdB, dLdC

    def PNRdetectorStochasticChannel(self, efficiency: Scalar, dark_counts: Scalar) -> Matrix:
        r"""
        Returns the stochastic matrix of the PNR detector.
        Args:
            efficiency: The detector efficiency.
            dark_counts: The dark counts.
        Returns:
            The stochastic matrix.
        """




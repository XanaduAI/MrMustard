from typing import Tuple, Sequence

from mrmustard.core.plugins import MathBackendInterface
from mrmustard.core.utils import Tensor


class FockPlugin:
    r"""
    A plugin that interfaces the phase space representation with the Fock representation.
    """
    math: MathBackendInterface

    # ~~~~~~
    # states
    # ~~~~~~

    def number_means(self, cov: Tensor, means: Tensor, hbar: float) -> Tensor:
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
                + self.math.diag_part(cov[:N, :N])
                + self.math.diag_part(cov[N:, N:]) - hbar
                ) / (2 * hbar)

    def number_cov(self, cov: Tensor, means: Tensor, hbar: float) -> Tensor:
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
        dd = self.math.diag(
            self.math.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])
            ) / (2 * hbar ** 2)
        CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
        return CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * self.math.eye(N)

    def ABC(self, cov: Tensor, means: Tensor, mixed: bool, hbar: float) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Returns the A matrix, B vector and C scalar
        given a Wigner covariance matrix and a means vector.
        The A, B, C triple is needed to compute the fock representation of the state.
        Args:
            cov: The Wigner covariance matrix.
            means: The Wigner means vector.
            mixed: Whether the state vector is mixed or not.
            hbar: The value of the Plank constant.
        Returns:
            The A matrix, B vector and C scalar.
        """
        num_modes = means.shape[-1] // 2

        # cov and means in the amplitude basis
        R = self.math.rotmat(num_modes)
        sigma = self.math.matmul(R,  cov/hbar, self.math.dagger(R))
        beta = self.math.matvec(R, means/self.math.sqrt(hbar))

        sQ = sigma + 0.5 * self.math.eye(2 * num_modes)
        sQinv = self.math.inv(sQ)
        X = self.math.Xmat(num_modes)
        A = self.math.matmul(X, self.math.eye(2 * num_modes) - sQinv)
        B = self.math.matvec(self.math.transpose(sQinv), self.math.conj(beta))
        exponent = -0.5 * self.math.sum(self.math.conj(beta)[:, None] * sQinv * beta[None, :])
        T = self.math.exp(exponent) / self.math.sqrt(self.math.det(sQ))
        N = num_modes - num_modes * mixed
        return (
            A[N:, N:],
            B[N:],
            T ** (0.5 + 0.5 * mixed),
        )  # will be off by global phase because T is real even for pure states

    def fock_representation(self, A: Tensor, B: Tensor, C: Tensor, cutoffs: Sequence[int]) -> Tensor:
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
        state = self.math.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed, dtype=self.math.complex128)
        state[(0,) * (len(cutoffs) + len(cutoffs) * mixed)] = C
        state = self.math.fill_amplitudes(state, A, B, cutoffs_minus_1)  # implemented in MathBackendInterface
        return state

    def grad_fock_representation(self,
                                 dout: Tensor,
                                 state: Tensor,
                                 A: Tensor,
                                 B: Tensor,
                                 C: Tensor,
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
        dA = self.math.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + A.shape, dtype=self.math.complex128)
        dB = self.math.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + B.shape, dtype=self.math.complex128)
        cutoffs_minus_1 = tuple([c - 1 for c in cutoffs] + [c - 1 for c in cutoffs] * mixed)
        dA, dB = self.math.fill_gradients(dA, dB, state, A, B, cutoffs_minus_1)  # implemented in MathBackendInterface
        dC = state / C
        dLdA = self.math.sum(dout[..., None, None] * self.math.conj(dA), axis=tuple(range(dout.ndim)))
        dLdB = self.math.sum(dout[..., None] * self.math.conj(dB), axis=tuple(range(dout.ndim)))
        dLdC = self.math.sum(dout * self.math.conj(dC), axis=tuple(range(dout.ndim)))
        return dLdA, dLdB, dLdC

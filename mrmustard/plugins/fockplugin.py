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

    def fock_representation(self, A: Matrix, B: Vector, C: Scalar, shape: Sequence[int]) -> Tensor:
        r"""
        Returns the fock state given the A, B, C matrices and a list of cutoff indices.
        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The cutoff indices.
        Returns:
            The fock state.
        """
        # mixed = len(B) == 2 * len(cutoffs)
        # cutoffs_minus_1 = tuple([c - 1 for c in cutoffs] + [c - 1 for c in cutoffs] * mixed)
        # state = self.backend.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed, dtype=self.backend.complex128)
        # state[(0,) * (len(cutoffs) + len(cutoffs) * mixed)] = C
        # state = self.backend.fill_amplitudes(state, A, B, cutoffs_minus_1)  # implemented in MathBackendInterface
        # return state

        return self._backend.hermite_renormalized(A, B, C, shape)

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
        dA, dB = self.backend.fill_gradients(dA, dB, state, A, B, cutoffs_minus_1)  # implemented in BackendInterface
        dC = state / C
        dLdA = self.backend.sum(dout[..., None, None] * self.backend.conj(dA), axis=tuple(range(dout.ndim)))
        dLdB = self.backend.sum(dout[..., None] * self.backend.conj(dB), axis=tuple(range(dout.ndim)))
        dLdC = self.backend.sum(dout * self.backend.conj(dC), axis=tuple(range(dout.ndim)))
        return dLdA, dLdB, dLdC

    def hermite_parameters(self, cov: Matrix, means: Vector, mixed: bool, hbar: float) -> Tuple[Matrix, Vector, Scalar]:  # TODO: move to FockPlugin?
        r"""
        Returns the A matrix, B vector and C scalar given a Wigner covariance matrix and a means vector.
        The A, B, C triple is needed to compute the Fock representation of the state.
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
        R = self.rotmat(num_modes)
        sigma = self.matmul(R, cov/hbar, self.dagger(R))
        beta = self.matvec(R, means/self.sqrt(hbar))

        sQ = sigma + 0.5 * self.eye(2 * num_modes)
        sQinv = self.inv(sQ)
        X = self.Xmat(num_modes)
        A = self.matmul(X, self.eye(2 * num_modes) - sQinv)
        B = self.matvec(self.transpose(sQinv), self.conj(beta))
        exponent = -0.5 * self.sum(self.conj(beta)[:, None] * sQinv * beta[None, :])
        T = self.exp(exponent) / self.sqrt(self.det(sQ))
        N = num_modes - num_modes * mixed
        return (
            A[N:, N:],
            B[N:],
            T ** (0.5 + 0.5 * mixed),
        )  # will be off by global phase because T is real even for pure states
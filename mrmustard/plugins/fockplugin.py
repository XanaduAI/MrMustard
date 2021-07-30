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

    def number_means(self, cov: Matrix, means: Vector, hbar: float) -> Vector:
        r"""
        Returns the photon number means vector
        given a Wigenr covariance matrix and a means vector.
        Args:
            cov: The Wigner covariance matrix.
            means: The Wigner means vector.
            hbar: The value of the Planck constant.
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
            hbar: The value of the Planck constant.
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

    def fock_representation(cov: Matrix, means: Vector, shape: Sequence[int], mixed: bool, hbar: float) -> Tensor:
        r"""
        Returns the Fock representation of the phase space representation
        given a Wigenr covariance matrix and a means vector.
        Args:
            cov: The Wigner covariance matrix.
            means: The Wigner means vector.
            shape: The shape of the tensor.
            mixed: Whether the state vector is mixed or not.
            hbar: The value of the Planck constant.
        Returns:
            The Fock representation of the phase space representation.
        """
        A, B, C = self.hermite_parameters(cov, means, mixed, hbar)
        return self.backend.hermite_renormalized(A, B, C, shape)


    def hermite_parameters(self, cov: Matrix, means: Vector, mixed: bool, hbar: float) -> Tuple[Matrix, Vector, Scalar]:  # TODO: move to FockPlugin?
        r"""
        Returns the A matrix, B vector and C scalar given a Wigner covariance matrix and a means vector.
        The A, B, C triple is needed to compute the Fock representation of the state.
        Args:
            cov: The Wigner covariance matrix.
            means: The Wigner means vector.
            mixed: Whether the state vector is mixed or not.
            hbar: The value of the Planck constant.
        Returns:
            The A matrix, B vector and C scalar.
        """
        num_indices = means.shape[-1]

        # cov and means in the amplitude basis
        R = self.backend.rotmat(num_indices//2)
        sigma = self.backend.matmul(self.backend.matmul(R, cov/hbar), self.backend.dagger(R))
        beta = self.backend.matvec(R, means/self.sqrt(hbar))

        sQ = sigma + 0.5 * self.eye(len(sigma))
        sQinv = self.inv(sQ)
        X = self.backend.Xmat(num_indices//2)
        A = self.backend.matmul(X, self.backend.eye(len(X)) - sQinv)
        B = self.backend.matvec(self.backend.transpose(sQinv), self.backend.conj(beta))
        exponent = -0.5 * self.sum(self.backend.conj(beta)[:, None] * sQinv * beta[None, :])
        T = self.backend.exp(exponent) / self.backend.sqrt(self.det(sQ))
        N = 0 if mixed else num_indices
        return (
            A[N:, N:],
            B[N:],
            T ** (1.0 if mixed else 0.5),
        )  # will be off by global phase because T is real even for pure states

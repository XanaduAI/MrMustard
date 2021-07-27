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

    def remove_subsystems(self, array: tf.Tensor, subsystems: List[int]) -> tf.Tensor:
        cutoffs = array.shape[: array.ndim // 2]
        # move the axes of subsystems to the end
        subsystems = [i for i in range(array.ndim) if i not in subsystems] + subsystems
        return tf.trace(tf.transpose(array, subsystems))
        array = tf.transpose(array, bla)
        array = tf.reshape(array, (np.prod(cutoffs), np.prod(cutoffs)))
        return tf.linalg.trace(array, axis1=subsystems, axis2=subsystems)

    def trace_all_systems(self, array: tf.Tensor) -> tf.Tensor:
        cutoffs = array.shape[: array.ndim // 2]
        array = tf.reshape(array, (np.prod(cutoffs), np.prod(cutoffs)))
        return tf.linalg.trace(array)
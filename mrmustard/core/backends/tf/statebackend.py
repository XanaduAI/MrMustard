import tensorflow as tf
import numpy as np
from typing import Tuple, Sequence
from mrmustard.backends import StateBackendInterface
from mrmustard.core import utils


class StateBackend(StateBackendInterface):
    def number_means(self, cov: tf.Tensor, means: tf.Tensor, hbar: float) -> tf.Tensor:
        N = means.shape[-1] // 2
        return (
            means[:N] ** 2
            + means[N:] ** 2
            + tf.linalg.diag_part(cov[:N, :N])
            + tf.linalg.diag_part(cov[N:, N:])
            - hbar
        ) / (2 * hbar)

    def number_cov(self, cov, means, hbar) -> tf.Tensor:
        N = means.shape[-1] // 2
        mCm = cov * means[:, None] * means[None, :]
        dd = tf.linalg.diag(
            tf.linalg.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])
        ) / (2 * hbar ** 2)
        CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
        return CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * np.identity(N)

    def ABC(
        self, cov: tf.Tensor, means: tf.Tensor, mixed: bool = False, hbar: float = 2.0
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        num_modes = means.shape[-1] // 2

        # cov and means in the amplitude basis
        R = utils.rotmat(num_modes)
        sigma = R @ tf.cast(cov / hbar, tf.complex128) @ tf.math.conj(tf.transpose(R))
        beta = tf.linalg.matvec(R, tf.cast(means / np.sqrt(hbar), tf.complex128))

        sQ = sigma + 0.5 * tf.eye(2 * num_modes, dtype=tf.complex128)
        sQinv = tf.linalg.inv(sQ)

        A = tf.cast(utils.Xmat(num_modes), tf.complex128) @ (np.identity(2 * num_modes) - sQinv)
        B = tf.linalg.matvec(tf.transpose(sQinv), tf.math.conj(beta))
        exponent = -0.5 * tf.reduce_sum(tf.math.conj(beta)[:, None] * sQinv * beta[None, :])
        T = tf.math.exp(exponent) / tf.math.sqrt(tf.linalg.det(sQ))
        N = num_modes - num_modes * mixed
        return (
            A[N:, N:],
            B[N:],
            T ** (0.5 + 0.5 * mixed),
        )  # will be off by global phase because T is real

    @tf.custom_gradient
    def fock_state(self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: Sequence[int]):
        mixed = len(B) == 2 * len(cutoffs)
        cutoffs_minus_1 = tuple([c - 1 for c in cutoffs] + [c - 1 for c in cutoffs] * mixed)
        state = np.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed, dtype=np.complex128)
        state[(0,) * (len(cutoffs) + len(cutoffs) * mixed)] = C
        state = utils.fill_amplitudes(state, A, B, cutoffs_minus_1)

        def grad(dy):
            dA = np.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + A.shape, dtype=np.complex128)
            dB = np.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + B.shape, dtype=np.complex128)
            dA, dB = utils.fill_gradients(dA, dB, state, A, B, cutoffs_minus_1)
            dC = state / C
            dLdA = np.sum(dy[..., None, None] * np.conj(dA), axis=tuple(range(dy.ndim)))
            dLdB = np.sum(dy[..., None] * np.conj(dB), axis=tuple(range(dy.ndim)))
            dLdC = np.sum(dy * np.conj(dC), axis=tuple(range(dy.ndim)))
            return dLdA, dLdB, dLdC

        return state, grad

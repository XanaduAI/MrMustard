import tensorflow as tf
from typing import Tuple

from mrmustard.backends import SymplecticBackendInterface


class SymplecticBackend(SymplecticBackendInterface):
    def loss_X(self, transmissivity: tf.Tensor) -> tf.Tensor:
        r"""Returns the X matrix for the lossy bosonic channel.
        The channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = tf.math.sqrt(transmissivity)
        return tf.linalg.diag(tf.concat([D, D], axis=0))

    def loss_Y(self, transmissivity: tf.Tensor, hbar: float) -> tf.Tensor:
        r"""Returns the Y (noise) matrix for the lossy bosonic channel.
        The channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = (1.0 - transmissivity) * hbar / 2.0
        return tf.linalg.diag(tf.concat([D, D], axis=0))

    def thermal_X(self, nbar: tf.Tensor, hbar: float) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def thermal_Y(self, nbar: tf.Tensor, hbar: float) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def displacement(self, x: tf.Tensor, y: tf.Tensor, hbar: float) -> tf.Tensor:
        return tf.cast(tf.math.sqrt(2 * hbar), x.dtype) * tf.concat([x, y], axis=0)

    def beam_splitter_symplectic(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        r"""Beam-splitter.
        Args:
            theta: transmissivity parameter
            phi: phase parameter
        Returns:
            array: symplectic-orthogonal transformation matrix of an interferometer with angles theta and phi
        """
        ct = tf.math.cos(theta)
        st = tf.math.sin(theta)
        cp = tf.math.cos(phi)
        sp = tf.math.sin(phi)
        return tf.convert_to_tensor(
            [
                [ct, -cp * st, 0, -sp * st],
                [cp * st, ct, -sp * st, 0],
                [0, sp * st, ct, -cp * st],
                [sp * st, 0, cp * st, ct],
            ]
        )

    def rotation_symplectic(self, phi: tf.Tensor) -> tf.Tensor:
        r"""Rotation gate.
        Args:
            phi: rotation angles
        Returns:
            array: rotation matrices by angle theta
        """
        num_modes = phi.shape[-1]
        x = tf.math.cos(phi)
        y = tf.math.sin(phi)
        return (
            tf.linalg.diag(tf.concat([x, x], axis=0))
            + tf.linalg.diag(-y, k=num_modes)
            + tf.linalg.diag(y, k=-num_modes)
        )

    def squeezing_symplectic(self, r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        r"""Squeezing. In fock space this corresponds to \exp(\tfrac{1}{2}r e^{i \phi} (a^2 - a^{\dagger 2}) ).

        Args:
            r: squeezing magnitude
            phi: rotation parameter
        Returns:
            array: symplectic transformation matrix
        """
        # pylint: disable=assignment-from-no-return
        num_modes = phi.shape[-1]
        cp = tf.math.cos(phi)
        sp = tf.math.sin(phi)
        ch = tf.math.cosh(r)
        sh = tf.math.sinh(r)
        return (
            tf.linalg.diag(tf.concat([ch - cp * sh, ch + cp * sh], axis=0))
            + tf.linalg.diag(-sp * sh, k=num_modes)
            + tf.linalg.diag(-sp * sh, k=-num_modes)
        )

    def two_mode_squeezing_symplectic(self, r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        r"""Two-mode squeezing.
        Args:
            r: squeezing magnitude
            phi: rotation parameter
        Returns:
            array: symplectic transformation matrix
        """
        # pylint: disable=assignment-from-no-return
        cp = tf.math.cos(phi)
        sp = tf.math.sin(phi)
        ch = tf.math.cosh(r)
        sh = tf.math.sinh(r)
        return tf.convert_to_tensor(
            [
                [ch, cp * sh, 0, sp * sh],
                [cp * sh, ch, sp * sh, 0],
                [0, sp * sh, ch, -cp * sh],
                [sp * sh, 0, -cp * sh, ch],
            ]
        )

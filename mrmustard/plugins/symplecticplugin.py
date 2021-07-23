from mrmustard.backends import BackendInterface
from typing import Tuple, Union
from mrmustard.backends import Vector, Matrix, Scalar


class SymplecticPlugin:
    r"""
    A plugin for all things symplectic.
    It relies on a math plugin (implementing the MathBackend interface) to do the actual math.

    The SymplecticPlugin implements:
      - Gaussian states
      - Gaussian mixture states [upcoming]
      - Gaussian unitary transformations
      - Gaussian CPTP channels
      - Gaussian CP channels [upcoming]
    """
    backend: BackendInterface  # will be loaded at runtime

    #  ~~~~~~
    #  States
    #  ~~~~~~

    def vacuum_state(self, num_modes: int, hbar: float) -> Tuple[Matrix, Vector]:
        r"""Returns the covariance matrix and displacement vector of the vacuum state.
        Args:
            num_modes (int): number of modes
            hbar (float): value of hbar
        Returns:
            Matrix: vacuum covariance matrix
            Vector: vacuum displacement vector
        """
        cov = self.backend.eye([num_modes*2], dtype=self.backend.float64) * hbar/2
        disp = self.backend.zeros([num_modes*2], dtype=self.backend.float64)
        return cov, disp

    def coherent_state(self, x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float) -> Tuple[Matrix, Vector]:
        r"""Returns the covariance matrix and displacement vector of a coherent state.
        The dimension depends on the dimensions of x and y.
        Args:
            x (scalar or vector): real part of the displacement
            y(scalar or vector): imaginary part of the displacement
            hbar: value of hbar
        Returns:
            Matrix: coherent state covariance matrix
            Vector: coherent state displacement vector
        """
        num_modes = self.backend.atleast_1d(x).shape[-1]
        cov = self.backend.eye([num_modes*2], dtype=self.backend.float64) * hbar/2
        disp = self.backend.concat([x, y], axis=-1) * self.backend.sqrt(2 * hbar)
        return cov, disp

    def squeezed_vacuum_state(self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector], hbar: float) -> Tuple[Matrix, Vector]:
        r"""Returns the covariance matrix and displacement vector of a squeezed vacuum state.
        The dimension depends on the dimensions of r and phi.
        Args:
            r (scalar or vector): squeezing magnitude
            phi (scalar or vector): squeezing angle
            hbar: value of hbar
        Returns:
            Matrix: squeezed state covariance matrix
            Vector: squeezed state displacement vector
        """
        S = self.squeezing_symplectic(r, phi)
        cov = self.backend.matmul(S, self.backend.transpose(S)) * hbar/2
        _, disp = self.coherent_state(0, 0, hbar)
        return cov, disp

    def thermal_state(self, nbar: Union[Scalar, Vector], hbar: float) -> Tuple[Matrix, Vector]:
        r"""Returns the covariance matrix and displacement vector of a thermal state.
        The dimension depends on the dimensions of nbar.
        Args:
            nbar (scalar or vector): average number of photons per mode
            hbar: value of hbar
        Returns:
            Matrix: thermal state covariance matrix
            Vector: thermal state displacement vector
        """
        g = (2*nbar + 1) * hbar/2
        cov = self.backend.diag(self.backend.concat([g, g], axis=-1), dtype=self.backend.float64)
        disp = self.backend.zeros([nbar*2], dtype=self.backend.float64)
        return cov, disp

    def displaced_squeezed_state(self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector], x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float) -> Tuple[Matrix, Vector]:
        r"""Returns the covariance matrix and displacement vector of a displaced squeezed state.
        The dimension depends on the dimensions of r, phi, x and y.
        Args:
            r   (scalar or vector): squeezing magnitude
            phi (scalar or vector): squeezing angle
            x   (scalar or vector): real part of the displacement
            y   (scalar or vector): imaginary part of the displacement
            hbar: value of hbar
        Returns:
            Matrix: displaced squeezed state covariance matrix
            Vector: displaced squeezed state displacement vector
        """
        S = self.squeezing_symplectic(r, phi)
        cov = self.backend.matmul(S, self.backend.transpose(S)) * hbar/2
        disp = self.backend.concat([x, y], axis=-1)
        return cov, disp

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    #  Unitary transformations
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    def rotation_symplectic(self, angle: Union[Scalar, Vector]) -> Matrix:
        r"""Symplectic matrix of a rotation gate.
        The dimension depends on the dimension of the angle.
        Args:
            angle (scalar or vector): rotation angles
        Returns:
            Tensor: symplectic matrix of a rotation gate
        """
        angle = self.backend.atleast_1d(angle)
        num_modes = angle.shape[-1]
        x = self.backend.cos(angle)
        y = self.backend.sin(angle)
        return (
            self.backend.diag(self.backend.concat([x, x], axis=0))
            + self.backend.diag(-y, k=num_modes)
            + self.backend.diag(y, k=-num_modes)
        )

    def squeezing_symplectic(self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector]) -> Matrix:
        r"""Symplectic matrix of a squeezing gate.
        The dimension depends on the dimension of r and phi.
        Args:
            r (scalar or vector): squeezing magnitude
            phi (scalar or vector): rotation parameter
        Returns:
            Tensor: symplectic matrix of a squeezing gate
        """
        r = self.backend.atleast_1d(r)
        phi = self.backend.atleast_1d(phi)
        num_modes = phi.shape[-1]
        cp = self.backend.cos(phi)
        sp = self.backend.sin(phi)
        ch = self.backend.cosh(r)
        sh = self.backend.sinh(r)
        return (
            self.backend.diag(self.backend.concat([ch - cp * sh, ch + cp * sh], axis=0))
            + self.backend.diag(-sp * sh, k=num_modes)
            + self.backend.diag(-sp * sh, k=-num_modes)
        )

    def displacement(self, x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float) -> Vector:
        r"""Returns the displacement vector for a displacement by alpha = x + iy.
        The dimension depends on the dimensions of x and y.
        Args:
            x (scalar or vector): real part of displacement
            y (scalar or vector): imaginary part of displacement
            hbar: value of hbar
        Returns:
            Vector: displacement vector of a displacement gate
        """
        return self.backend.sqrt(2 * hbar) * self.backend.concat([x, y], axis=0)

    def beam_splitter_symplectic(self, theta: Scalar, phi: Scalar) -> Matrix:
        r"""Symplectic matrix of a Beam-splitter gate.
        The dimension is 4x4.
        Args:
            theta: transmissivity parameter
            phi: phase parameter
        Returns:
            Matrix: symplectic (orthogonal) matrix of a beam-splitter gate
        """
        ct = self.backend.cos(theta)
        st = self.backend.sin(theta)
        cp = self.backend.cos(phi)
        sp = self.backend.sin(phi)
        zero = self.backend.zeros_like(theta)
        return self.backend.Tensor(
            [
                [ct, -cp * st, zero, -sp * st],
                [cp * st, ct, -sp * st, zero],
                [zero, sp * st, ct, -cp * st],
                [sp * st, zero, cp * st, ct],
            ]
        )

    def mz_symplectic(self, phi_a: Scalar, phi_b: Scalar, internal: bool = False) -> Matrix:
        r"""Symplectic matrix of a Mach-Zehnder gate.
        It supports two conventions:
        if `internal=True`, both phases act inside the interferometer:
            `phi_a` on the upper arm, `phi_b` on the lower arm;
        if `internal = False` (default), both phases act on the upper arm:
            `phi_a` before the first BS, `phi_b` after the first BS.
        Args:
            phi_a (float): first phase
            phi_b (float): second phase
            internal (bool): whether phases are in the internal arms (default is False)
        Returns:
            Matrix: symplectic (orthogonal) matrix of a Mach-Zehnder interferometer
        """
        ca = self.backend.cos(phi_a)
        sa = self.backend.sin(phi_a)
        cb = self.backend.cos(phi_b)
        sb = self.backend.sin(phi_b)
        cp = self.backend.cos(phi_a + phi_b)
        sp = self.backend.sin(phi_a + phi_b)

        if internal:
            return 0.5 * self.backend.Tensor(
                [
                    [ca - cb, -sa - sb, sb - sa, -ca - cb],
                    [-sa - sb, cb - ca, -ca - cb, sa - sb],
                    [sa - sb, ca + cb, ca - cb, -sa - sb],
                    [ca + cb, sb - sa, -sa - sb, cb - ca],
                ]
            )
        else:
            return 0.5 * self.backend.Tensor(
                [
                    [cp - ca, -sb, sa - sp, -1 - cb],
                    [-sa - sp, 1 - cb, -ca - cp, sb],
                    [sp - sa, 1 + cb, cp - ca, -sb],
                    [cp + ca, -sb, -sa - sp, 1 - cb],
                ]
            )

    def two_mode_squeezing_symplectic(self, r: Scalar, phi: Scalar) -> Matrix:
        r"""Symplectic matrix of a two-mode squeezing gate.
        The dimension is 4x4.
        Args:
            r (float): squeezing magnitude
            phi (float): rotation parameter
        Returns:
            Matrix: symplectic matrix of a two-mode squeezing gate
        """
        cp = self.backend.cos(phi)
        sp = self.backend.sin(phi)
        ch = self.backend.cosh(r)
        sh = self.backend.sinh(r)
        zero = self.backend.zeros_like(r)
        return self.backend.astensor(
            [
                [ch, cp * sh, zero, sp * sh],
                [cp * sh, ch, sp * sh, zero],
                [zero, sp * sh, ch, -cp * sh],
                [sp * sh, zero, -cp * sh, ch],
            ]
        )

    # ~~~~~~~~~~~~~
    # CPTP channels
    # ~~~~~~~~~~~~~

    def loss_X(self, transmissivity: Union[Scalar, Vector]) -> Matrix:
        r"""Returns the X matrix for the lossy bosonic channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = self.backend.sqrt(transmissivity)
        return self.backend.diag(self.backend.concat([D, D], axis=0))

    def loss_Y(self, transmissivity: Union[Scalar, Vector], hbar: float) -> Matrix:
        r"""Returns the Y (noise) matrix for the lossy bosonic channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = (1.0 - transmissivity) * hbar/2
        return self.backend.diag(self.backend.concat([D, D], axis=0))

    def thermal_X(self, nbar: Union[Scalar, Vector], hbar: float) -> Matrix:
        r"""Returns the X matrix for the thermal lossy channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        raise NotImplementedError

    def thermal_Y(self, nbar: Union[Scalar, Vector], hbar: float) -> Matrix:
        r"""Returns the Y (noise) matrix for the thermal lossy channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~
    # non-TP channels
    # ~~~~~~~~~~~~~~~

    def homodyne(self, angle: Union[Scalar, Vector], measurement: Union[Scalar, Vector]) -> Matrix:
        r"""Returns the homodyne channel operator.
        """
        raise NotImplementedError

    def heterodyne(self, angle: Union[Scalar, Vector], measurement: Union[Scalar, Vector]) -> Matrix:
        r"""Returns the heterodyne channel operator.
        """
        raise NotImplementedError

    def anydyne(self, angle: Union[Scalar, Vector], measurement: Union[Scalar, Vector]) -> Matrix:
        r"""Returns the anydyne channel operator.
        """
        raise NotImplementedError

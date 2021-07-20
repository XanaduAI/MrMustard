import numpy as np
from typing import Tuple

from mrmustard.core.plugins import MathPluginInterface

from typing import TypeVar

Tensor = TypeVar('Tensor')

class SymplecticPlugin:
    r"""
    A plugin for all things symplectic, including symplectic transformations and covariance matrices.
    It relies on a math plugin (implementing the MathPlugin interface)
    to do the actual math, which is set by the user.

    The SymplecticPlugin implements:
      - Gaussian states
      - Gaussian mixture states [upcoming]
      - Gaussian unitary transformations
      - Gaussian CPTP channels
      - Gaussian CP channels [upcoming]
    """
    math: MathPlugin  # will be loaded at runtime

    #  ~~~~~~
    #  States
    #  ~~~~~~
    
    def vacuum_state(self, nmodes: int, hbar: float) -> Tuple[Tensor, Tensor]:
        r"""Returns the covariance matrix and displacement vector of the vacuum state.
        Args:
            nmodes (int): number of modes
            hbar (float): value of hbar
        Returns:
            Tensor: vacuum covariance matrix
            Tensor: vacuum displacement vector
        """
        cov = self.math.eye([nmodes*2], dtype=self.math.float64) * hbar/2
        disp = self.math.zeros([nmodes*2], dtype=self.math.float64)
        return cov, disp 

    def coherent_state(self, x: Tensor, y: Tensor, hbar: float) -> Tuple[Tensor, Tensor]:
        r"""Returns the covariance matrix and displacement vector of a coherent state.
        Args:
            x: real part of the displacement
            y: imaginary part of the displacement
            hbar: value of hbar
        Returns:
            Tensor: coherent state covariance matrix
            Tensor: coherent state displacement vector
        """
        cov = self.math.eye([nmodes*2], dtype=self.math.float64) * hbar/2
        disp = self.math.concat([x, y], axis=-1) * self.math.sqrt(2 * hbar)
        return cov, disp

    def squeezed_state(self, r: Tensor, phi: Tensor, hbar: float) -> Tuple[Tensor, Tensor]:
        r"""Returns the covariance matrix and displacement vector of a squeezed vacuum state.
        Args:
            r: squeezing magnitude
            phi: squeezing angle
            hbar: value of hbar
        Returns:
            Tensor: squeezed state covariance matrix
            Tensor: squeezed state displacement vector
        """
        S = self.squeezing_symplectic(r, phi)
        cov = self.math.matmul(S, self.math.transpose(S)) * hbar/2
        _, disp = self.coherent_state(0, 0, hbar)
        return cov, disp

    def thermal_state(self, nbar: Tensor, hbar: float) -> Tuple[Tensor, Tensor]:
        r"""Returns the covariance matrix and displacement vector of a thermal state.
        Args:
            nbar: average number of photons per mode
            hbar: value of hbar
        Returns:
            Tensor: thermal state covariance matrix
            Tensor: thermal state displacement vector
        """
        g = (2*nbar + 1) * hbar/2
        cov = self.math.diag(self.math.concat([g, g], axis=-1), dtype=self.math.float64)
        disp = self.math.zeros([nbar*2], dtype=self.math.float64)
        return cov, disp

    def displaced_squeezed_state(self, r: Tensor, phi: Tensor, x: Tensor, y: Tensor, hbar: float) -> Tuple[Tensor, Tensor]:
        r"""Returns the covariance matrix and displacement vector of a displaced squeezed state.
        Note:
            The 
        Args:
            r: squeezing magnitude
            phi: squeezing angle
            x: real part of the displacement
            y: imaginary part of the displacement
            hbar: value of hbar
        Returns:
            Tensor: displaced squeezed state covariance matrix
            Tensor: displaced squeezed state displacement vector
        """
        S = self.squeezing_symplectic(r, phi)
        cov = self.math.matmul(S, self.math.transpose(S)) * hbar/2
        disp = self.math.concat([x, y], axis=-1)
        return cov, disp

    # ~~~~~~~~~~~~~~~~~~~~~~~~
    #  Unitary transformations
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    def rotation_symplectic(self, angle: Tensor) -> Tensor:
        r"""Symplectic matrix of a rotation gate.
        Args:
            angle: rotation angles
        Returns:
            Tensor: symplectic matrix of a rotation gate
        """
        angle = self.math.atleast_1d(angle)
        num_modes = angle.shape[-1]
        x = self.math.cos(angle)
        y = self.math.sin(angle)
        return self.math.diag(self.math.concat([x, x], axis=0)) + self.math.diag(-y, k=num_modes) + self.math.diag(y, k=-num_modes)

    def squeezing_symplectic(self, r: Tensor, phi: Tensor) -> Tensor:
        r"""Symplectic matrix of a squeezing gate.

        Args:
            r (Tensor): squeezing magnitude
            phi (Tensor): rotation parameter
        Returns:
            Tensor: symplectic matrix of a squeezing gate
        """
        # pylint: disable=assignment-from-no-return
        r = self.math.atleast_1d(r)
        phi = self.math.atleast_1d(phi)
        num_modes = phi.shape[-1]
        cp = self.math.cos(phi)
        sp = self.math.sin(phi)
        ch = self.math.cosh(r)
        sh = self.math.sinh(r)
        return (
            self.math.diag(self.math.concat([ch - cp * sh, ch + cp * sh], axis=0))
            + self.math.diag(-sp * sh, k=num_modes)
            + self.math.diag(-sp * sh, k=-num_modes)
        )

    def displacement(self, x: Tensor, y: Tensor, hbar: float) -> Tensor:
        r"""Returns the displacement vector for a displacement by alpha = x + iy.
        Args:
            x: real part of displacement
            y: imaginary part of displacement
            hbar: value of hbar
        Returns:
            Tensor: displacement vector of a displacement gate
        """
        return self.math.sqrt(2 * hbar) * self.math.concat([x, y], axis=0)

    def beam_splitter_symplectic(self, theta: Tensor, phi: Tensor) -> Tensor:
        r"""Symplectic matrix of a Beam-splitter gate.
        Args:
            theta: transmissivity parameter
            phi: phase parameter
        Returns:
            Tensor: symplectic (orthogonal) matrix of a beam-splitter gate
        """
        ct = self.math.cos(theta)
        st = self.math.sin(theta)
        cp = self.math.cos(phi)
        sp = self.math.sin(phi)
        zero = self.math.zeros_like(theta)
        return self.math.Tensor(
            [
                [ct, -cp * st, zero, -sp * st],
                [cp * st, ct, -sp * st, zero],
                [zero, sp * st, ct, -cp * st],
                [sp * st, zero, cp * st, ct],
            ]
        )

    def mz_symplectic(self, phi_a: Tensor, phi_b: Tensor, internal: bool = False) -> Tensor:
        r"""Symplectic matrix of a Mach-Zehnder gate.
        It supports two conventions:
        if `internal=True`, both phases act inside the interferometer: `phi_a` on the upper arm, `phi_b` on the lower arm;
        if `internal = False` (default), both phases act on the upper arm: `phi_a` before the first BS, `phi_b` after the first BS.
        Args:
            phi_a: first phase
            phi_b: second phase
            internal (bool): whether phases are in the internal arms (default is False)
        Returns:
            Tensor: symplectic (orthogonal) matrix of a Mach-Zehnder interferometer
        """
        ca = self.math.cos(phi_a)
        sa = self.math.sin(phi_a)
        cb = self.math.cos(phi_b)
        sb = self.math.sin(phi_b)
        cp = self.math.cos(phi_a + phi_b)
        sp = self.math.sin(phi_a + phi_b)

        if internal:
            return 0.5 * self.math.Tensor(
                [
                    [ca - cb, -sa - sb, sb - sa, -ca - cb],
                    [-sa - sb, cb - ca, -ca - cb, sa - sb],
                    [sa - sb, ca + cb, ca - cb, -sa - sb],
                    [ca + cb, sb - sa, -sa - sb, cb - ca],
                ]
            )
        else:
            return 0.5 * self.math.Tensor(
                [
                    [cp - ca, -sb, sa - sp, -1 - cb],
                    [-sa - sp, 1 - cb, -ca - cp, sb],
                    [sp - sa, 1 + cb, cp - ca, -sb],
                    [cp + ca, -sb, -sa - sp, 1 - cb],
                ]
            )

    def two_mode_squeezing_symplectic(self, r: Tensor, phi: Tensor) -> Tensor:
        r"""Symplectic matrix of a two-mode squeezing gate.
        Args:
            r: squeezing magnitude
            phi: rotation parameter
        Returns:
            Tensor: symplectic matrix of a two-mode squeezing gate
        """
        # pylint: disable=assignment-from-no-return
        cp = self.math.cos(phi)
        sp = self.math.sin(phi)
        ch = self.math.cosh(r)
        sh = self.math.sinh(r)
        zero = self.math.zeros_like(r)
        return self.Tensor(
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

    def loss_X(self, transmissivity: Tensor) -> Tensor:
        r"""Returns the X matrix for the lossy bosonic channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = self.math.sqrt(transmissivity)
        return self.math.diag(self.math.concat([D, D], axis=0))

    def loss_Y(self, transmissivity: Tensor, hbar: float) -> Tensor:
        r"""Returns the Y (noise) matrix for the lossy bosonic channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = (1.0 - transmissivity) * hbar/2
        return self.math.diag(self.math.concat([D, D], axis=0))

    def thermal_X(self, nbar: Tensor, hbar: float) -> Tuple[Tensor, Tensor]:
        r"""Returns the X matrix for the thermal lossy channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        raise NotImplementedError

    def thermal_Y(self, nbar: Tensor, hbar: float) -> Tuple[Tensor, Tensor]:
        r"""Returns the Y (noise) matrix for the thermal lossy channel.
        The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~
    # non-TP channels
    # ~~~~~~~~~~~~~~~

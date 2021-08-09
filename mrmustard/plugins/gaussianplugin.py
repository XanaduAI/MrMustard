from mrmustard.backends import BackendInterface
from mrmustard.typing import *


class GaussianPlugin:
    r"""
    A plugin for all things Gaussian.

    The GaussianPlugin implements:
      - Gaussian states (pure and mixed)
      - Gaussian mixture states [upcoming]
      - Gaussian unitary transformations
      - Gaussian CPTP channels
      - Gaussian CP channels [upcoming]
      - Gaussian utilities
    """
    backend: BackendInterface

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

    def CPTP(self, cov: Matrix, means: Vector, X: Matrix, Y: Matrix, d: Vector, modes: Sequence[int]) -> Tuple[Matrix, Vector]:
        r"""Returns the cov matrix of a state after undergoing a CPTP channel, computed as `cov = X \cdot cov \cdot X^T + Y`.
        If the channel is single-mode, `modes` can contain `M` modes to apply the channel to,
        otherwise it must contain as many modes as the number of modes in the channel.

        Args:
            cov (Matrix): covariance matrix
            means (Vector): means vector
            X (Matrix): the X matrix of the CPTP channel
            Y (Matrix): noise matrix of the CPTP channel
            d (Vector): displacement vector of the CPTP channel
            modes (Sequence[int]): modes on which the channel operates
        Returns:
            Tuple[Matrix, Vector]: the covariance matrix and the means vector of the state after the CPTP channel
        """
        if X.shape[-1] == Y.shape[-1] == d.shape[-1] == 2:  # single-mode channel
            X = self._backend.single_mode_to_multimode_mat(X, len(modes))
            Y = self._backend.single_mode_to_multimode_mat(Y, len(modes))
            d = self._backend.single_mode_to_multimode_vec(d, len(modes))
        cov = self._backend.left_matmul_to_modes(X, cov, modes)
        cov = self._backend.right_matmul_to_modes(cov, self._backend.transpose(X), modes)
        cov = self._backend.add_to_modes(cov, Y, modes)
        means = self._backend.left_matmul_to_modes(X, means, modes)
        means = self._backend.add_to_modes(means, d, modes)
        return cov, means

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
        The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.
        """
        raise NotImplementedError

    def thermal_Y(self, nbar: Union[Scalar, Vector], hbar: float) -> Matrix:
        r"""Returns the Y (noise) matrix for the thermal lossy channel.
        The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.
        """
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~
    # non-TP channels
    # ~~~~~~~~~~~~~~~

    def homodyne(self, *args, **kwargs) -> Tuple[Scalar, Matrix, Vector]:
        r"""
        Returns the results of a homodyne measurement.
        """
        raise NotImplementedError

    def heterodyne(self, *args, **kwargs) -> Tuple[Scalar, Matrix, Vector]:
        r"""
        Returns the results of a heterodyne measurement.
        """
        raise NotImplementedError

    def general_dyne(self, cov: Matrix, means: Vector, proj_cov: Matrix, proj_means: Vector, modes: Sequence[int]) -> Tuple[Scalar, Matrix, Vector]:
        r"""
        Returns the results of a general dyne measurement.
        Arguments:
            cov (Matrix): covariance matrix of the state being measured
            means (Vector): means vector of the state being measured
            proj_cov (Matrix): covariance matrix of the state being projected onto
            proj_means (Vector): means vector of the state being projected onto (i.e. the measurement outcome)
            modes (Sequence[int]): modes being measured
        Returns:
            Tuple[Scalar, Matrix, Vector]: the outcome probability *density*, the post-measurement cov and means vector
        """
        N, n = len(cov) // 2, len(proj_cov) // 2
        Amodes = [i for i in range(N) if i not in modes]
        A, B, AB = self.partition_cov(cov, Amodes)
        a, b = self.partition_means(means, Amodes)
        inv = self.backend.inv(B + proj_cov)
        ABinv = self.backend.matmul(AB, inv)
        new_cov = A - self.backend.matmul(ABinv, self.backend.transpose(AB))
        new_means = a + self.backend.matvec(ABinv, proj_means - b)
        prob = self.backend.exp(self.backend.matvec(self.backend.matvec(inv, proj_means - b), proj_means - b)) / (self.backend.pi**(N-n) * self.backend.sqrt(self.backend.det(B + proj_cov)))
        return prob, new_cov, new_means

    # ~~~~~~~~~
    # utilities
    # ~~~~~~~~~

    def trace(self, cov: Matrix, means: Vector, Bmodes: Sequence[int]) -> Tuple[Matrix, Vector]:
        r"""
        Returns the covariances and means after discarding the specified modes.
        Arguments:
            cov (Matrix): covariance matrix
            means (Vector): means vector
            Bmodes (Sequence[int]): modes to discard
        Returns:
            Tuple[Matrix, Vector]: the covariance matrix and the means vector after discarding the specified modes
        """
        N = len(cov) // 2
        Aindices = self.backend.astensor([i for i in range(N) if i not in Bmodes])
        A_cov_block = self.backend.gather(self.backend.gather(cov, Aindices, axis=0), Aindices, axis=1)
        A_means_vec = self.backend.gather(means, Aindices)
        return A_cov_block, A_means_vec

    def partition_cov(self, cov: Matrix, Amodes: Sequence[int]) -> Tuple[Matrix, Matrix, Matrix]:
        r"""
        Partitions the covariance matrix into the A and B subsystems and the AB coherence block.
        Arguments:
            cov (Matrix): the covariance matrix
            Amodes (Sequence[int]): the modes of system A
        Returns:
            Tuple[Matrix, Matrix, Matrix]: the cov of A, the cov of B and the AB block
        """
        N = len(cov) // 2
        Bindices = self.backend.astensor([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes])
        Aindices = self.backend.astensor(Amodes + [i + N for i in Amodes])
        
        A_block =  self.backend.gather(self.backend.gather(cov, Aindices, axis=0), Aindices, axis=1)
        B_block =  self.backend.gather(self.backend.gather(cov, Bindices, axis=0), Bindices, axis=1)
        AB_block = self.backend.gather(self.backend.gather(cov, Aindices, axis=0), Bindices, axis=1)
        return A_block, B_block, AB_block

    def partition_means(self, means: Vector, Amodes: Sequence[int]) -> Tuple[Vector, Vector]:
        r"""
        Partitions the means vector into the A and B subsystems.
        Arguments:
            means (Vector): the means vector
            Amodes (Sequence[int]): the modes of system A
        Returns:
            Tuple[Vector, Vector]: the means of A and the means of B
        """
        N = len(means) // 2
        Bindices = self.backend.astensor([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes])
        Aindices = self.backend.astensor(Amodes + [i + N for i in Amodes])
        return self.backend.gather(means, Aindices), self.backend.gather(means, Bindices)

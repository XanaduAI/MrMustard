from mrmustard import Backend
from mrmustard._typing import *
from math import pi, sqrt
from thewalrus.quantum import is_pure_cov

r"""
A plugin for all things Gaussian.

The GaussianPlugin implements:
    - Gaussian states (pure and mixed)
    - Gaussian mixture states [upcoming]
    - Gaussian unitary transformations
    - Gaussian CPTP channels
    - Gaussian CP channels [upcoming]
    - Gaussian entropies [upcoming]
    - Gaussian entanglement [upcoming]
"""
backend = Backend()

#  ~~~~~~
#  States
#  ~~~~~~


def vacuum_state(num_modes: int, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of the vacuum state.
    Args:
        num_modes (int): number of modes
        hbar (float): value of hbar
    Returns:
        Matrix: vacuum covariance matrix
        Vector: vacuum means vector
    """
    cov = backend.eye(num_modes * 2, dtype=backend.float64) * hbar / 2
    means = backend.zeros([num_modes * 2], dtype=backend.float64)
    return cov, means


def coherent_state(x: Vector, y: Vector, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of a coherent state.
    The dimension depends on the dimensions of x and y.
    Args:
        x (vector): real part of the means vector
        y (vector): imaginary part of the means vector
        hbar: value of hbar
    Returns:
        Matrix: coherent state covariance matrix
        Vector: coherent state means vector
    """
    x = backend.atleast_1d(x)
    y = backend.atleast_1d(y)
    num_modes = x.shape[-1]
    cov = backend.eye(num_modes * 2, dtype=x.dtype) * hbar / 2
    means = backend.concat([x, y], axis=0) * backend.sqrt(2 * hbar, dtype=x.dtype)
    return cov, means


def squeezed_vacuum_state(r: Vector, phi: Vector, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of a squeezed vacuum state.
    The dimension depends on the dimensions of r and phi.
    Args:
        r (vector): squeezing magnitude
        phi (vector): squeezing angle
        hbar: value of hbar
    Returns:
        Matrix: squeezed state covariance matrix
        Vector: squeezed state means vector
    """
    S = squeezing_symplectic(r, phi)
    cov = backend.matmul(S, backend.transpose(S)) * hbar / 2
    means = backend.zeros(cov.shape[-1], dtype=cov.dtype)
    return cov, means


def thermal_state(nbar: Vector, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of a thermal state.
    The dimension depends on the dimensions of nbar.
    Args:
        nbar (vector): average number of photons per mode
        hbar: value of hbar
    Returns:
        Matrix: thermal state covariance matrix
        Vector: thermal state means vector
    """
    g = (2 * backend.atleast_1d(nbar) + 1) * hbar / 2
    cov = backend.diag(backend.concat([g, g], axis=-1))
    means = backend.zeros(cov.shape[-1], dtype=cov.dtype)
    return cov, means


def displaced_squeezed_state(r: Vector, phi: Vector, x: Vector, y: Vector, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of a displaced squeezed state.
    The dimension depends on the dimensions of r, phi, x and y.
    Args:
        r   (scalar or vector): squeezing magnitude
        phi (scalar or vector): squeezing angle
        x   (scalar or vector): real part of the means
        y   (scalar or vector): imaginary part of the means
        hbar: value of hbar
    Returns:
        Matrix: displaced squeezed state covariance matrix
        Vector: displaced squeezed state means vector
    """
    S = squeezing_symplectic(r, phi)
    cov = backend.matmul(S, backend.transpose(S)) * hbar / 2
    means = displacement(x, y, hbar)
    return cov, means


def two_mode_squeezed_vacuum_state(r: Vector, phi: Vector, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of a two-mode squeezed vacuum state.
    The dimension depends on the dimensions of r and phi.
    Args:
        r (vector): squeezing magnitude
        phi (vector): squeezing angle
        hbar: value of hbar
    Returns:
        Matrix: two-mode squeezed state covariance matrix
        Vector: two-mode squeezed state means vector
    """
    S = two_mode_squeezing_symplectic(r, phi)
    cov = backend.matmul(S, backend.transpose(S)) * hbar / 2
    means = backend.zeros(cov.shape[-1], dtype=cov.dtype)
    return cov, means


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_symplectic(angle: Union[Scalar, Vector]) -> Matrix:
    r"""Symplectic matrix of a rotation gate.
    The dimension depends on the dimension of the angle.
    Args:
        angle (scalar or vector): rotation angles
    Returns:
        Tensor: symplectic matrix of a rotation gate
    """
    angle = backend.atleast_1d(angle)
    num_modes = angle.shape[-1]
    x = backend.cos(angle)
    y = backend.sin(angle)
    return backend.diag(backend.concat([x, x], axis=0)) + backend.diag(-y, k=num_modes) + backend.diag(y, k=-num_modes)


def squeezing_symplectic(r: Union[Scalar, Vector], phi: Union[Scalar, Vector]) -> Matrix:
    r"""Symplectic matrix of a squeezing gate.
    The dimension depends on the dimension of r and phi.
    Args:
        r (scalar or vector): squeezing magnitude
        phi (scalar or vector): rotation parameter
    Returns:
        Tensor: symplectic matrix of a squeezing gate
    """
    r = backend.atleast_1d(r)
    phi = backend.atleast_1d(phi)
    num_modes = phi.shape[-1]
    cp = backend.cos(phi)
    sp = backend.sin(phi)
    ch = backend.cosh(r)
    sh = backend.sinh(r)
    cpsh = cp * sh
    spsh = sp * sh
    return (
        backend.diag(backend.concat([ch - cpsh, ch + cpsh], axis=0)) + backend.diag(-spsh, k=num_modes) + backend.diag(-spsh, k=-num_modes)
    )


def displacement(x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float) -> Vector:
    r"""Returns the displacement vector for a displacement by alpha = x + iy.
    The dimension depends on the dimensions of x and y.
    Args:
        x (scalar or vector): real part of displacement
        y (scalar or vector): imaginary part of displacement
        hbar: value of hbar
    Returns:
        Vector: displacement vector of a displacement gate
    """
    x = backend.atleast_1d(x)
    y = backend.atleast_1d(y)
    if x.shape[-1] == 1:
        x = backend.tile(x, y.shape)
    if y.shape[-1] == 1:
        y = backend.tile(y, x.shape)
    return backend.sqrt(2 * hbar, dtype=x.dtype) * backend.concat([x, y], axis=0)


def beam_splitter_symplectic(theta: Scalar, phi: Scalar) -> Matrix:
    r"""Symplectic matrix of a Beam-splitter gate.
    The dimension is 4x4.
    Args:
        theta: transmissivity parameter
        phi: phase parameter
    Returns:
        Matrix: symplectic (orthogonal) matrix of a beam-splitter gate
    """
    ct = backend.cos(theta)
    st = backend.sin(theta)
    cp = backend.cos(phi)
    sp = backend.sin(phi)
    zero = backend.zeros_like(theta)
    return backend.astensor(
        [
            [ct, -cp * st, zero, -sp * st],
            [cp * st, ct, -sp * st, zero],
            [zero, sp * st, ct, -cp * st],
            [sp * st, zero, cp * st, ct],
        ]
    )


def mz_symplectic(phi_a: Scalar, phi_b: Scalar, internal: bool = False) -> Matrix:
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
    ca = backend.cos(phi_a)
    sa = backend.sin(phi_a)
    cb = backend.cos(phi_b)
    sb = backend.sin(phi_b)
    cp = backend.cos(phi_a + phi_b)
    sp = backend.sin(phi_a + phi_b)

    if internal:
        return 0.5 * backend.astensor(
            [
                [ca - cb, -sa - sb, sb - sa, -ca - cb],
                [-sa - sb, cb - ca, -ca - cb, sa - sb],
                [sa - sb, ca + cb, ca - cb, -sa - sb],
                [ca + cb, sb - sa, -sa - sb, cb - ca],
            ]
        )
    else:
        return 0.5 * backend.astensor(
            [
                [cp - ca, -sb, sa - sp, -1 - cb],
                [-sa - sp, 1 - cb, -ca - cp, sb],
                [sp - sa, 1 + cb, cp - ca, -sb],
                [cp + ca, -sb, -sa - sp, 1 - cb],
            ]
        )


def two_mode_squeezing_symplectic(r: Scalar, phi: Scalar) -> Matrix:
    r"""Symplectic matrix of a two-mode squeezing gate.
    The dimension is 4x4.
    Args:
        r (float): squeezing magnitude
        phi (float): rotation parameter
    Returns:
        Matrix: symplectic matrix of a two-mode squeezing gate
    """
    cp = backend.cos(phi)
    sp = backend.sin(phi)
    ch = backend.cosh(r)
    sh = backend.sinh(r)
    zero = backend.zeros_like(r)
    return backend.astensor(
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


def CPTP(cov: Matrix, means: Vector, X: Matrix, Y: Matrix, d: Vector, modes: Sequence[int]) -> Tuple[Matrix, Vector]:
    r"""Returns the cov matrix and means vector of a state after undergoing a CPTP channel, computed as `cov = X \cdot cov \cdot X^T + Y`
    and `d = X \cdot means + d`.
    If the channel is single-mode, `modes` can contain `M` modes to apply the channel to,
    otherwise it must contain as many modes as the number of modes in the channel.

    Args:
        cov (Matrix): covariance matrix
        means (Vector): means vector
        X (Matrix): the X matrix of the CPTP channel
        Y (Matrix): noise matrix of the CPTP channel
        d (Vector): displacement vector of the CPTP channel
        modes (Sequence[int]): modes on which the channel operates
        hbar (float): value of hbar
    Returns:
        Tuple[Matrix, Vector]: the covariance matrix and the means vector of the state after the CPTP channel
    """
    # if single-mode channel, apply to all modes indicated in `modes`
    if X is not None and X.shape[-1] == 2:
        X = backend.single_mode_to_multimode_mat(X, len(modes))
    if Y is not None and Y.shape[-1] == 2:
        Y = backend.single_mode_to_multimode_mat(Y, len(modes))
    if d is not None and d.shape[-1] == 2:
        d = backend.single_mode_to_multimode_vec(d, len(modes))
    cov = backend.left_matmul_at_modes(X, cov, modes)
    cov = backend.right_matmul_at_modes(cov, backend.transpose(X), modes)
    cov = backend.add_at_modes(cov, Y, modes)
    means = backend.matvec_at_modes(X, means, modes)
    means = backend.add_at_modes(means, d, modes)
    return cov, means


def loss_X(transmissivity: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = backend.sqrt(transmissivity)
    return backend.diag(backend.concat([D, D], axis=0))


def loss_Y(transmissivity: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = (1.0 - transmissivity) * hbar / 2
    return backend.diag(backend.concat([D, D], axis=0))


def thermal_X(nbar: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the thermal lossy channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.
    """
    raise NotImplementedError


def thermal_Y(nbar: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the thermal lossy channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.
    """
    raise NotImplementedError


def compose_channels_XYd(X1: Matrix, Y1: Matrix, d1: Vector, X2: Matrix, Y2: Matrix, d2: Vector) -> Tuple[Matrix, Matrix, Vector]:
    r"""Returns the combined X, Y, and d for two CPTP channels.
    Arguments:
        X1 (Matrix): the X matrix of the first CPTP channel
        Y1 (Matrix): the Y matrix of the first CPTP channel
        d1 (Vector): the displacement vector of the first CPTP channel
        X2 (Matrix): the X matrix of the second CPTP channel
        Y2 (Matrix): the Y matrix of the second CPTP channel
        d2 (Vector): the displacement vector of the second CPTP channel
    Returns:
        Tuple[Matrix, Matrix, Vector]: the combined X, Y, and d matrices
    """
    if X1 is None:
        X = X2
    elif X2 is None:
        X = X1
    else:
        X = backend.matmul(X2, X1)
    if Y1 is None:
        Y = Y2
    elif Y2 is None:
        Y = Y1
    else:
        Y = backend.matmul(backend.matmul(X2, Y1), X2) + Y2
    if d1 is None:
        d = d2
    elif d2 is None:
        d = d1
    else:
        d = backend.matmul(X2, d1) + d2
    return X, Y, d


# ~~~~~~~~~~~~~~~
# non-TP channels
# ~~~~~~~~~~~~~~~98


def general_dyne(
    cov: Matrix, means: Vector, proj_cov: Matrix, proj_means: Vector, modes: Sequence[int], hbar: float
) -> Tuple[Scalar, Matrix, Vector]:
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
    N = cov.shape[-1] // 2
    nB = proj_cov.shape[-1] // 2  # B is the system being measured
    nA = N - nB  # A is the leftover
    Amodes = [i for i in range(N) if i not in modes]
    A, B, AB = partition_cov(cov, Amodes)
    a, b = partition_means(means, Amodes)
    proj_cov = backend.cast(proj_cov, B.dtype)
    proj_means = backend.cast(proj_means, b.dtype)
    inv = backend.inv(B + proj_cov)
    new_cov = A - backend.matmul(backend.matmul(AB, inv), backend.transpose(AB))
    new_means = a + backend.matvec(backend.matmul(AB, inv), proj_means - b)
    prob = backend.exp(-backend.sum(backend.matvec(inv, proj_means - b) * proj_means - b)) / (
        pi ** nB * (hbar ** -nB) * backend.sqrt(backend.det(B + proj_cov))
    )  # TODO: check this (hbar part especially)
    return prob, new_cov, new_means


# ~~~~~~~~~
# utilities
# ~~~~~~~~~


def is_mixed_cov(cov: Matrix) -> bool:
    r"""
    Returns True if the covariance matrix is mixed, False otherwise.
    """
    return not is_pure_cov(backend.asnumpy(cov))


def trace(cov: Matrix, means: Vector, Bmodes: Sequence[int]) -> Tuple[Matrix, Vector]:
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
    Aindices = backend.astensor([i for i in range(N) if i not in Bmodes])
    A_cov_block = backend.gather(backend.gather(cov, Aindices, axis=0), Aindices, axis=1)
    A_means_vec = backend.gather(means, Aindices)
    return A_cov_block, A_means_vec


def partition_cov(cov: Matrix, Amodes: Sequence[int]) -> Tuple[Matrix, Matrix, Matrix]:
    r"""
    Partitions the covariance matrix into the A and B subsystems and the AB coherence block.
    Arguments:
        cov (Matrix): the covariance matrix
        Amodes (Sequence[int]): the modes of system A
    Returns:
        Tuple[Matrix, Matrix, Matrix]: the cov of A, the cov of B and the AB block
    """
    N = cov.shape[-1] // 2
    Bindices = backend.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
    Aindices = backend.cast(Amodes + [i + N for i in Amodes], "int32")
    A_block = backend.gather(backend.gather(cov, Aindices, axis=1), Aindices, axis=0)
    B_block = backend.gather(backend.gather(cov, Bindices, axis=1), Bindices, axis=0)
    AB_block = backend.gather(backend.gather(cov, Bindices, axis=1), Aindices, axis=0)
    return A_block, B_block, AB_block


def partition_means(means: Vector, Amodes: Sequence[int]) -> Tuple[Vector, Vector]:
    r"""
    Partitions the means vector into the A and B subsystems.
    Arguments:
        means (Vector): the means vector
        Amodes (Sequence[int]): the modes of system A
    Returns:
        Tuple[Vector, Vector]: the means of A and the means of B
    """
    N = len(means) // 2
    Bindices = backend.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
    Aindices = backend.cast(Amodes + [i + N for i in Amodes], "int32")
    return backend.gather(means, Aindices), backend.gather(means, Bindices)


def purity(cov: Matrix, hbar: float) -> Scalar:
    r"""
    Returns the purity of the state with the given covariance matrix.
    Arguments:
        cov (Matrix): the covariance matrix
    Returns:
        float: the purity
    """
    return 1 / backend.sqrt(backend.det((2 / hbar) * cov))


def sympletic_eigenvals(cov: Matrix) -> Any:
    r"""
    Returns the sympletic eigenspectrum of a covariance matrix.
    For a pure state, we expect the sympletic eigenvalues to be 1.
    Arguments:
        cov (Matrix): the covariance matrix.
    Returns:
        List[float]: the sympletic eigenvalues
    """
    cov = backend.cast(cov, "complex128")  # cast to complex otherwise matmul will break
    J = backend.J(cov.shape[0] // 2)  # create a sympletic form
    M = 1j * J @ cov  # compute iJ*cov
    vals = backend.eigvals(M)  # compute the eigenspectrum
    return backend.abs(vals[::2])  # return the even eigenvalues


def von_neumann_entropy(cov: Matrix) -> float:
    r"""
    Returns the Von Neumann entropy.
    For a pure state, we expect the Von Neumann entropy to be 0.
    Arguments:
        cov (Matrix): the covariance matrix
    Returns:
        List[float]: the sympletic eigenvalues
    Reference: (https://arxiv.org/pdf/1110.3234.pdf), Equations 46-47.
    """
    symp_vals = self.sympletic_eigenvals(cov)
    g = lambda x: backend.xlogy((x + 1) / 2, (x + 1) / 2) - backend.xlogy((x - 1) / 2, (x - 1) / 2 + 1e-9)
    entropy = backend.sum(g(symp_vals))
    return entropy


def fidelity(mu1: float, cov1: Matrix, mu2: float, cov2: Matrix, hbar=2.0, rtol=1e-05, atol=1e-08) -> float:
    r"""
    Returns the fidelity of two gaussian states.

    Reference: https://arxiv.org/pdf/2102.05748.pdf, Equations 95-99. Note that we compute the square of equation 98.
    """

    cov1 = backend.cast(cov1 / hbar, "complex128")  # convert to units where hbar = 1
    cov2 = backend.cast(cov2 / hbar, "complex128")  # convert to units where hbar = 1

    mu1 = backend.cast(mu1, "complex128")
    mu2 = backend.cast(mu2, "complex128")
    deltar = (mu2 - mu1) / backend.sqrt(hbar, dtype=mu1.dtype)  # convert to units where hbar = 1
    J = backend.J(cov1.shape[0] // 2)
    I = backend.eye(cov1.shape[0])
    J = backend.cast(J, "complex128")
    I = backend.cast(I, "complex128")

    cov12_inv = backend.inv(cov1 + cov2)

    V = backend.transpose(J) @ cov12_inv @ ((1 / 4) * J + cov2 @ J @ cov1)

    W = -2 * (V @ (1j * J))
    W_inv = backend.inv(W)
    matsqrtm = backend.sqrtm(I - W_inv @ W_inv)  # this also handles the case where the input matrix is close to zero
    f0_top = backend.det((matsqrtm + I) @ (W @ (1j * J)))
    f0_bot = backend.det(cov1 + cov2)

    f0 = (f0_top / f0_bot) ** (1 / 2)  # square of equation 98

    dot = backend.sum(
        backend.transpose(deltar) * backend.matvec(cov12_inv, deltar)
    )  # computing (mu2-mu1)/sqrt(hbar).T @ cov12_inv @ (mu2-mu1)/sqrt(hbar)

    fidelity = f0 * backend.exp((-1 / 2) * dot)  # square of equation 95

    return backend.cast(fidelity, "float64")

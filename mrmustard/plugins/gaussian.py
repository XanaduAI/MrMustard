from mrmustard import Backend
from mrmustard._typing import *
from math import pi, sqrt
from thewalrus.quantum import is_pure_cov
from mrmustard.experimental import XPMatrix, XPVector

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


def vacuum_state(num_modes: int, hbar: float) -> Tuple[XPMatrix, XPVector]:
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
    return XPMatrix.from_xxpp(cov, like_1=True), XPVector.from_xxpp(means)


def coherent_state(x: Vector, y: Vector, hbar: float) -> Tuple[XPMatrix, XPVector]:
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
    return CPTP(d=displacement(x, y, hbar))


def squeezed_vacuum_state(r: Vector, phi: Vector, hbar: float) -> Tuple[XPMatrix, XPVector]:
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
    return CPTP(X = squeezing_symplectic(r, phi))

def thermal_state(nbar: Vector, hbar: float) -> Tuple[XPMatrix, XPVector]:
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


def displaced_squeezed_state(r: Vector, phi: Vector, x: Vector, y: Vector, hbar: float) -> Tuple[XPMatrix, XPVector]:
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
    return CPTP(X = squeezing_symplectic(r, phi), d = displacement(x, y, hbar))


def two_mode_squeezed_vacuum_state(r: Vector, phi: Vector, hbar: float) -> Tuple[XPMatrix, XPVector]:
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
    return CPTP(X = two_mode_squeezing_symplectic(r, phi))


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_symplectic(angle: Union[Scalar, Vector]) -> XPMatrix:
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
    return XPMatrix.from_xxpp(backend.diag(backend.concat([x, x], axis=0)) + backend.diag(-y, k=num_modes) + backend.diag(y, k=-num_modes), like_1=True)


def squeezing_symplectic(r: Union[Scalar, Vector], phi: Union[Scalar, Vector]) -> XPMatrix:
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
    return XPMatrix.from_xxpp(
        backend.diag(backend.concat([ch - cpsh, ch + cpsh], axis=0)) + backend.diag(-spsh, k=num_modes) + backend.diag(-spsh, k=-num_modes),
        like_1=True,
        )


def displacement(x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float) -> XPVector:
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
    return XPVector.from_xxpp(backend.sqrt(2 * hbar, dtype=x.dtype) * backend.concat([x, y], axis=0))


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
    return XPMatrix.from_xxpp(backend.astensor(
        [
            [ct, -cp * st, zero, -sp * st],
            [cp * st, ct, -sp * st, zero],
            [zero, sp * st, ct, -cp * st],
            [sp * st, zero, cp * st, ct],
        ]
    ), like_1=True)


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
        return XPMatrix.from_xxpp(0.5 * backend.astensor(
            [
                [ca - cb, -sa - sb, sb - sa, -ca - cb],
                [-sa - sb, cb - ca, -ca - cb, sa - sb],
                [sa - sb, ca + cb, ca - cb, -sa - sb],
                [ca + cb, sb - sa, -sa - sb, cb - ca],
            ]
        ), like_1=True)
    else:
        return XPMatrix.from_xxpp(0.5 * backend.astensor(
            [
                [cp - ca, -sb, sa - sp, -1 - cb],
                [-sa - sp, 1 - cb, -ca - cp, sb],
                [sp - sa, 1 + cb, cp - ca, -sb],
                [cp + ca, -sb, -sa - sp, 1 - cb],
            ]
        ), like_1=True)


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
    return XPMatrix.from_xxpp(backend.astensor(
        [
            [ch, cp * sh, zero, sp * sh],
            [cp * sh, ch, sp * sh, zero],
            [zero, sp * sh, ch, -cp * sh],
            [sp * sh, zero, -cp * sh, ch],
        ]
    ), like_1=True)


# ~~~~~~~~~~~~~
# CPTP channels
# ~~~~~~~~~~~~~


def CPTP(cov = XPMatrix(like_1=True),
         means = XPVector(),
         X = XPMatrix(like_1=True),
         Y = XPMatrix(like_0=True),
         d = XPVector(),
         modes: Sequence[int] = []) -> Tuple[XPMatrix, XPVector]:
    r"""Returns the cov matrix and means vector of a state after undergoing a CPTP channel,
    computed as `cov = X \cdot cov \cdot X^T + Y` and `d = X \cdot means + d`.
    Single-mode channels can be applied in parallel (i.e. `modes` can contain `M` modes to apply the channel to).

    Args:
        cov (Matrix): covariance matrix
        means (Vector): means vector
        X (Matrix): the X matrix of the CPTP channel
        Y (Matrix): noise matrix of the CPTP channel
        d (Vector): displacement vector of the CPTP channel
        modes (optional, Sequence[int]): modes on which the channel operates
        hbar (float): value of hbar
    Returns:
        Tuple[Matrix, Vector]: the covariance matrix and the means vector of the state after the CPTP channel
    """
    # if single-mode channel, apply to all modes indicated in `modes`
    if len(X.inmodes) == len(X.outmodes) == 1 and len(modes) > 1:
        X = X.clone_like(cov)
    if len(Y.inmodes) == len(Y.outmodes) == 1 and len(modes) > 1:
        Y = Y.clone_like(cov)
    if len(d.outmodes) == 1 and len(modes) > 1:
        d = d.clone_like(means)
    cov = X @ cov @ X.T + Y
    means = X @ means + d
    return cov, means


def loss_X(transmissivity: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = backend.sqrt(transmissivity)
    return XPMatrix.from_xxpp(backend.diag(backend.concat([D, D], axis=0)), like_1=True)


def loss_Y(transmissivity: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = (1.0 - transmissivity) * hbar / 2
    return XPMatrix.from_xxpp(backend.diag(backend.concat([D, D], axis=0)), like_0=True)


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
        X = X2 @ X1
    if Y1 is None:
        Y = Y2
    elif Y2 is None:
        Y = Y1
    else:
        Y = X2 @ Y1 @ X2 + Y2
    if d1 is None:
        d = d2
    elif d2 is None:
        d = d1
    else:
        d = X2 @ d1 + d2
    return X, Y, d


# ~~~~~~~~~~~~~~~
# non-TP channels
# ~~~~~~~~~~~~~~~98


def general_dyne(
    cov: Matrix, means: Vector, proj_cov: Matrix, proj_means: Vector, modes: Sequence[int], hbar: float
) -> Tuple[Scalar, XPMatrix, XPVector]:
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
    unmeasured_modes = [i for i in cov.inmodes if i not in modes]
    A = cov[unmeasured_modes, unmeasured_modes]
    B = cov[modes, modes]
    AB = cov[unmeasured_modes, modes]
    a = means[unmeasured_modes]
    b = means[modes]

    inv = XPMatrix.from_xxpp(backend.inv(B + proj_cov), modes=B.modes, like_1=True)   #NOTE: not really like 1, but it's not None...

    new_cov = A - AB @ inv @ AB.T
    new_means = a - AB @ inv @ (proj_means - b)

    prob = backend.exp(-(proj_means - b) @ inv @ (proj_means - b))/(pi ** nB * (hbar ** -nB) * backend.sqrt(backend.det(B + proj_cov)))
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

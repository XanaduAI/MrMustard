# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mrmustard.utils.types import *
from thewalrus.quantum import is_pure_cov
from mrmustard.utils.xptensor import XPMatrix, XPVector
from mrmustard import settings
from numpy import pi

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  NOTE: the math backend is loaded automatically by the settings object
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def vacuum_cov(num_modes: int, hbar: float) -> Matrix:
    r"""Returns the real covariance matrix of the vacuum state.
    Args:
        num_modes (int): number of modes
        hbar (float): value of hbar
    Returns:
        Matrix: vacuum covariance matrix
    """
    return math.eye(num_modes * 2, dtype=math.float64) * hbar / 2


def vacuum_means(num_modes: int, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of the vacuum state.
    Args:
        num_modes (int): number of modes
        hbar (float): value of hbar
    Returns:
        Matrix: vacuum covariance matrix
        Vector: vacuum means vector
    """
    return displacement(math.zeros(num_modes), math.zeros(num_modes), hbar)


def squeezed_vacuum_cov(r: Vector, phi: Vector, hbar: float) -> Matrix:
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
    return math.matmul(S, math.transpose(S)) * hbar / 2


def thermal_cov(nbar: Vector, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of a thermal state.
    The dimension depends on the dimensions of nbar.
    Args:
        nbar (vector): average number of photons per mode
        hbar: value of hbar
    Returns:
        Matrix: thermal state covariance matrix
        Vector: thermal state means vector
    """
    g = (2 * math.atleast_1d(nbar) + 1) * hbar / 2
    return math.diag(math.concat([g, g], axis=-1))


def two_mode_squeezed_vacuum_cov(r: Vector, phi: Vector, hbar: float) -> Matrix:
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
    return math.matmul(S, math.transpose(S)) * hbar / 2


def gaussian_cov(symplectic: Matrix, eigenvalues: Vector = None, hbar: float = 2.0) -> Matrix:
    r"""Returns the covariance matrix of a Gaussian state.
    Args:
        symplectic (Tensor): symplectic matrix of a channel
        eigenvalues (vector): symplectic eigenvalues
        hbar (float): value of hbar
    Returns:
        Tensor: covariance matrix of the Gaussian state
    """
    if eigenvalues is None:
        return math.matmul(symplectic, math.transpose(symplectic))
    else:
        return math.matmul(math.matmul(symplectic, math.diag(eigenvalues)), math.transpose(symplectic))


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
    angle = math.atleast_1d(angle)
    num_modes = angle.shape[-1]
    x = math.cos(angle)
    y = math.sin(angle)
    return math.diag(math.concat([x, x], axis=0)) + math.diag(-y, k=num_modes) + math.diag(y, k=-num_modes)


def squeezing_symplectic(r: Union[Scalar, Vector], phi: Union[Scalar, Vector]) -> Matrix:
    r"""Symplectic matrix of a squeezing gate.
    The dimension depends on the dimension of r and phi.
    Args:
        r (scalar or vector): squeezing magnitude
        phi (scalar or vector): rotation parameter
    Returns:
        Tensor: symplectic matrix of a squeezing gate
    """
    r = math.atleast_1d(r)
    phi = math.atleast_1d(phi)
    num_modes = phi.shape[-1]
    cp = math.cos(phi)
    sp = math.sin(phi)
    ch = math.cosh(r)
    sh = math.sinh(r)
    cpsh = cp * sh
    spsh = sp * sh
    return math.diag(math.concat([ch - cpsh, ch + cpsh], axis=0)) + math.diag(-spsh, k=num_modes) + math.diag(-spsh, k=-num_modes)


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
    x = math.atleast_1d(x)
    y = math.atleast_1d(y)
    if x.shape[-1] == 1:
        x = math.tile(x, y.shape)
    if y.shape[-1] == 1:
        y = math.tile(y, x.shape)
    return math.sqrt(2 * hbar, dtype=x.dtype) * math.concat([x, y], axis=0)


def beam_splitter_symplectic(theta: Scalar, phi: Scalar) -> Matrix:
    r"""Symplectic matrix of a Beam-splitter gate.
    The dimension is 4x4.
    Args:
        theta: transmissivity parameter
        phi: phase parameter
    Returns:
        Matrix: symplectic (orthogonal) matrix of a beam-splitter gate
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    cp = math.cos(phi)
    sp = math.sin(phi)
    zero = math.zeros_like(theta)
    return math.astensor(
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
    ca = math.cos(phi_a)
    sa = math.sin(phi_a)
    cb = math.cos(phi_b)
    sb = math.sin(phi_b)
    cp = math.cos(phi_a + phi_b)
    sp = math.sin(phi_a + phi_b)

    if internal:
        return 0.5 * math.astensor(
            [
                [ca - cb, -sa - sb, sb - sa, -ca - cb],
                [-sa - sb, cb - ca, -ca - cb, sa - sb],
                [sa - sb, ca + cb, ca - cb, -sa - sb],
                [ca + cb, sb - sa, -sa - sb, cb - ca],
            ]
        )
    else:
        return 0.5 * math.astensor(
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
    cp = math.cos(phi)
    sp = math.sin(phi)
    ch = math.cosh(r)
    sh = math.sinh(r)
    zero = math.zeros_like(r)
    return math.astensor(
        [
            [ch, cp * sh, zero, sp * sh],
            [cp * sh, ch, sp * sh, zero],
            [zero, sp * sh, ch, -cp * sh],
            [sp * sh, zero, -cp * sh, ch],
        ]
    )


def two_mode_controlled_phase(g=1):
    r"""Controlled PHASE gate of two-gaussian modes.

    C_Z = \exp(ig q_1 \otimes q_2).

    Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 8.
    https://arxiv.org/pdf/1110.3234.pdf, Equation 161.

    Args:
        g (float): interaction strength
    Returns:
        the C_Z controlled phase matrix (in xpxp format)
    """

    return math.astensor(
        [
            [1, 0, 0, 0],
            [0, 1, g, 0],
            [0, 0, 1, 0],
            [g, 0, 0, 1],
        ]
    )


def two_mode_controlled_not(g=1):
    r"""Controlled NOT gate of two-gaussian modes.

    C_X = \exp(ig q_1 \otimes p_2).

    Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 9.

    Args:
        g (float): interaction strength
    Returns:
        the C_X controlled NOT matrix (in xpxp format)
    """

    return math.astensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -g],
            [g, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def controlled_phase(N: Scalar, a: Scalar, b: Scalar, g=1):
    r"""Controlled PHASE gate of N-mode gaussian.

    C_Z = \exp(ig q_A \otimes q_B).

    Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 9.

    Args:
        N (int): number of modes
        a (int): mode number A between (1, N)
        b (int): mode number B between (1, N)
        g (float): interaction strength
    Returns:
        the C_Z controlled PHASE matrix (in xpxp format)
    """

    S = math.eye(N)
    S[2 * a - 1, 2 * b - 2] = g
    S[2 * b - 1, 2 * a - 2] = g
    return math.astensor(S)


def controlled_NOT(N: Scalar, a: Scalar, b: Scalar, g=1):
    r"""Controlled NOT gate of N-mode gaussian.

    C_X = \exp(ig q_A \otimes p_B).

    Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 9.

    Args:
        N (int): number of modes
        a (int): mode number A between (1, N)
        b (int): mode number B between (1, N)
        g (float): interaction strength
    Returns:
        the C_X controlled NOT matrix (in xpxp format)
    """

    S = math.eye(N)
    S[2 * a - 1, 2 * b - 1] = -g
    S[2 * b - 2, 2 * a - 2] = g
    return math.astensor(S)


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
        X = math.single_mode_to_multimode_mat(X, len(modes))
    if Y is not None and Y.shape[-1] == 2:
        Y = math.single_mode_to_multimode_mat(Y, len(modes))
    if d is not None and d.shape[-1] == 2:
        d = math.single_mode_to_multimode_vec(d, len(modes))
    cov = math.left_matmul_at_modes(X, cov, modes)
    cov = math.right_matmul_at_modes(cov, math.transpose(X), modes)
    cov = math.add_at_modes(cov, Y, modes)
    means = math.matvec_at_modes(X, means, modes)
    means = math.add_at_modes(means, d, modes)
    return cov, means


def loss_X(transmissivity: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.

    This channel couples mode a to a thermal state b with the transformation

    a -> sqrt(t) a + \sqrt(1-t) b

    Reference: https://arxiv.org/pdf/1110.3234.pdf, Equation 113.

    Arguments:
        transmissivity (float): value of the transmissivity, must be between 0 and 1
    Returns:
        Tuple[Matrix, Vector]: the X matrix of the loss channel.
    """
    D = math.sqrt(transmissivity)
    return math.diag(math.concat([D, D], axis=0))


def loss_Y(transmissivity: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.

    This channel couples mode a to a thermal state b with the transformation

    a -> sqrt(t) a + \sqrt(1-t) b

    Reference: https://arxiv.org/pdf/1110.3234.pdf, Equation 113.

    Arguments:
        transmissivity (float): value of the transmissivity, must be between 0 and 1
        hbar (float): value of hbar
    Returns:
        Tuple[Matrix, Vector]: the Y matrix of the loss channel.
    """
    D = (1.0 - transmissivity) * hbar / 2
    return math.diag(math.concat([D, D], axis=0))


def thermal_X(transmissivity: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the thermal lossy channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.

    Note that if nbar = 0, the thermal loss channel reduces to the loss channel.

    This channel couples mode a to a thermal state b with the transformation

    a -> sqrt(t) a + \sqrt(1-t) b

    Reference: https://arxiv.org/pdf/1110.3234.pdf, Equation 113.

    Arguments:
        transmissivity (float): value of the transmissivity, must be between 0 and 1
    Returns:
        Tuple[Matrix, Vector]: the X matrix of the thermal loss channel.
    """

    D = math.sqrt(transmissivity)
    return math.diag(math.concat([D, D], axis=0))


def thermal_Y(transmissivity: Union[Scalar, Vector], nbar: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the thermal lossy channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.

    Note that if nbar = 0, the thermal loss channel reduces to the loss channel.

    This channel couples mode a to a thermal state b with the transformation

    a -> sqrt(t) a + \sqrt(1-t) b

    Reference: https://arxiv.org/pdf/1110.3234.pdf, Equation 113.

    Arguments:
        transmissivity (float): value of the transmissivity, must be between 0 and 1
        nbar (float): average number of photons per mode
        hbar (float): value of hbar
    Returns:
        Tuple[Matrix, Vector]: the Y matrix of the thermal loss channel.
    """
    D = (1.0 - transmissivity) * (hbar / 2) * (2 * nbar + 1)
    return math.diag(math.concat([D, D], axis=0))


def amp_X(transmissivity: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the amplification channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.

    This channel couples mode a to a thermal state b with the transformation

    a -> sqrt(t) a + \sqrt(t-1) b

    Reference: https://arxiv.org/pdf/1110.3234.pdf, Equation 113.

    Arguments:
        transmissivity (float): value of the transmissivity > 1
        nbar (float): average number of photons per mode
    Returns:
        Tuple[Matrix, Vector]: the Y matrix of the amplification channel.
    """
    D = math.sqrt(transmissivity)
    return math.diag(math.concat([D, D], axis=0))


def amp_Y(transmissivity: Union[Scalar, Vector], nbar: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the X matrix for the amplification channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.

    This channel couples mode a to a thermal state b with the transformation

    a -> sqrt(t) a + \sqrt(1-t) b

    Reference: https://arxiv.org/pdf/1110.3234.pdf, Equation 113.

    Arguments:
        transmissivity (float): value of the transmissivity > 1
        nbar (float): average number of photons per mode
        hbar (float): value of hbar
    Returns:
        Tuple[Matrix, Vector]: the Y matrix of the amplification channel.
    """
    D = (transmissivity - 1) * (hbar / 2) * (2 * nbar + 1)
    return math.diag(math.concat([D, D], axis=0))


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
        X = math.matmul(X2, X1)
    if Y1 is None:
        Y = Y2
    elif Y2 is None:
        Y = Y1
    else:
        Y = math.matmul(math.matmul(X2, Y1), X2) + Y2
    if d1 is None:
        d = d2
    elif d2 is None:
        d = d1
    else:
        d = math.matmul(X2, d1) + d2
    return X, Y, d


# ~~~~~~~~~~~~~~~
# non-TP channels
# ~~~~~~~~~~~~~~~


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
    proj_cov = math.cast(proj_cov, B.dtype)
    proj_means = math.cast(proj_means, b.dtype)
    inv = math.inv(B + proj_cov)
    new_cov = A - math.matmul(math.matmul(AB, inv), math.transpose(AB))
    new_means = a + math.matvec(math.matmul(AB, inv), proj_means - b)
    prob = math.exp(-math.sum(math.matvec(inv, proj_means - b) * proj_means - b)) / (
        pi ** nB * (hbar ** -nB) * math.sqrt(math.det(B + proj_cov))
    )  # TODO: check this (hbar part especially)
    return prob, new_cov, new_means


# ~~~~~~~~~
# utilities
# ~~~~~~~~~
def number_means(cov: Matrix, means: Vector, hbar: float) -> Vector:
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
    return (means[:N] ** 2 + means[N:] ** 2 + math.diag_part(cov[:N, :N]) + math.diag_part(cov[N:, N:]) - hbar) / (2 * hbar)


def number_cov(cov: Matrix, means: Vector, hbar: float) -> Matrix:
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
    dd = math.diag(math.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])) / (2 * hbar ** 2)
    CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
    return CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * math.eye(N, dtype=CC.dtype)


def is_mixed_cov(cov: Matrix) -> bool:  # TODO: deprecate
    r"""
    Returns True if the covariance matrix is mixed, False otherwise.
    """
    return not is_pure_cov(math.asnumpy(cov), hbar=settings.HBAR)


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
    Aindices = math.astensor([i for i in range(N) if i not in Bmodes])
    A_cov_block = math.gather(math.gather(cov, Aindices, axis=0), Aindices, axis=1)
    A_means_vec = math.gather(means, Aindices)
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
    Bindices = math.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
    Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
    A_block = math.gather(math.gather(cov, Aindices, axis=1), Aindices, axis=0)
    B_block = math.gather(math.gather(cov, Bindices, axis=1), Bindices, axis=0)
    AB_block = math.gather(math.gather(cov, Bindices, axis=1), Aindices, axis=0)
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
    Bindices = math.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
    Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
    return math.gather(means, Aindices), math.gather(means, Bindices)


def purity(cov: Matrix, hbar: float) -> Scalar:
    r"""
    Returns the purity of the state with the given covariance matrix.
    Arguments:
        cov (Matrix): the covariance matrix
    Returns:
        float: the purity
    """
    return 1 / math.sqrt(math.det((2 / hbar) * cov))


def join_covs(covs: Sequence[Matrix]) -> Tuple[Matrix, Vector]:
    r"""
    Joins the given covariance matrices into a single covariance matrix.
    Arguments:
        covs (Sequence[Matrix]): the covariance matrices
    Returns:
        Matrix: the joined covariance matrix
    """
    modes = list(range(len(covs[0]) // 2))
    cov = XPMatrix.from_xxpp(covs[0], modes=(modes, modes), like_1=True)
    for i, c in enumerate(covs[1:]):
        modes = list(range(cov.num_modes, cov.num_modes + c.shape[-1] // 2))
        cov = cov @ XPMatrix.from_xxpp(c, modes=(modes, modes), like_1=True)
    return cov.to_xxpp()


def join_means(means: Sequence[Vector]) -> Vector:
    r"""
    Joins the given means vectors into a single means vector.
    Arguments:
        means (Sequence[Vector]): the means vectors
    Returns:
        Vector: the joined means vector
    """
    mean = XPVector.from_xxpp(means[0], modes=list(range(len(means[0]) // 2)))
    for i, m in enumerate(means[1:]):
        mean = mean + XPVector.from_xxpp(m, modes=list(range(mean.num_modes, mean.num_modes + len(m) // 2)))
    return mean.to_xxpp()

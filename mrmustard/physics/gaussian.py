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

from thewalrus.quantum import is_pure_cov

from mrmustard.types import *
from mrmustard.utils.xptensor import XPMatrix, XPVector
from mrmustard import settings
from numpy import pi
from mrmustard.math import Math

math = Math()


#  ~~~~~~
#  States
#  ~~~~~~


def vacuum_cov(num_modes: int, hbar: float) -> Matrix:
    r"""Returns the real covariance matrix of the vacuum state.

    Args:
        num_modes (int): number of modes
        hbar (float): value of ``hbar``

    Returns:
        Matrix: vacuum covariance matrix
    """
    return math.eye(num_modes * 2, dtype=math.float64) * hbar / 2


def vacuum_means(num_modes: int, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of the vacuum state.

    Args:
        num_modes (int): number of modes
        hbar (float): value of ``hbar``

    Returns:
        Matrix, Vector: thermal state covariance matrix or means vector
    """
    return displacement(
        math.zeros(num_modes, dtype="float64"), math.zeros(num_modes, dtype="float64"), hbar
    )


def squeezed_vacuum_cov(r: Vector, phi: Vector, hbar: float) -> Matrix:
    r"""Returns the real covariance matrix and real means vector of a squeezed vacuum state.

    The dimension depends on the dimensions of ``r`` and ``phi``.

    Args:
        r (vector): squeezing magnitude
        phi (vector): squeezing angle
        hbar: value of ``hbar``

    Returns:
        Matrix, Vector: thermal state covariance matrix or means vector
    """
    S = squeezing_symplectic(r, phi)
    return math.matmul(S, math.transpose(S)) * hbar / 2


def thermal_cov(nbar: Vector, hbar: float) -> Tuple[Matrix, Vector]:
    r"""Returns the real covariance matrix and real means vector of a thermal state.

    The dimension depends on the dimensions of ``nbar``.

    Args:
        nbar (vector): average number of photons per mode
        hbar: value of ``hbar``

    Returns:
        Matrix, Vector: thermal state covariance matrix or means vector
    """
    g = (2 * math.atleast_1d(nbar) + 1) * hbar / 2
    return math.diag(math.concat([g, g], axis=-1))


def two_mode_squeezed_vacuum_cov(r: Vector, phi: Vector, hbar: float) -> Matrix:
    r"""Returns the real covariance matrix and real means vector of a two-mode squeezed vacuum state.

    The dimension depends on the dimensions of ``r`` and ``phi``.

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
        return math.matmul(
            math.matmul(symplectic, math.diag(math.concat([eigenvalues, eigenvalues], axis=0))),
            math.transpose(symplectic),
        )


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
    return (
        math.diag(math.concat([x, x], axis=0))
        + math.diag(-y, k=num_modes)
        + math.diag(y, k=-num_modes)
    )


def squeezing_symplectic(r: Union[Scalar, Vector], phi: Union[Scalar, Vector]) -> Matrix:
    r"""Symplectic matrix of a squeezing gate.

    The dimension depends on the dimension of ``r`` and ``phi``.

    Args:
        r (scalar or vector): squeezing magnitude
        phi (scalar or vector): rotation parameter

    Returns:
        Tensor: symplectic matrix of a squeezing gate
    """
    r = math.atleast_1d(r)
    phi = math.atleast_1d(phi)
    if r.shape[-1] == 1:
        r = math.tile(r, phi.shape)
    if phi.shape[-1] == 1:
        phi = math.tile(phi, r.shape)
    num_modes = phi.shape[-1]
    cp = math.cos(phi)
    sp = math.sin(phi)
    ch = math.cosh(r)
    sh = math.sinh(r)
    cpsh = cp * sh
    spsh = sp * sh
    return (
        math.diag(math.concat([ch - cpsh, ch + cpsh], axis=0))
        + math.diag(-spsh, k=num_modes)
        + math.diag(-spsh, k=-num_modes)
    )


def displacement(x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float) -> Vector:
    r"""Returns the displacement vector for a displacement by :math:`alpha = x + iy`.
    The dimension depends on the dimensions of ``x`` and ``y``.

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

    The dimension is :math:`4\times 4`.

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

        * if ``internal=True``, both phases act inside the interferometer:
            ``phi_a`` on the upper arm, ``phi_b`` on the lower arm;
        * if `internal = False` (default), both phases act on the upper arm:
            ``phi_a`` before the first BS, ``phi_b`` after the first BS.

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

    The dimension is :math:`4\times 4`.

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


def quadratic_phase(s: Scalar):
    r"""Quadratic phase single mode gate.

    .. math::

        P = \exp(i s q^2 / 2 \hbar)

    Reference: https://strawberryfields.ai/photonics/conventions/gates.html

    Args:
        s (float): interaction strength

    Returns:
        Tensor: the :math:`P(s)` matrix (in ``xxpp`` ordering)
    """

    return math.astensor(
        [
            [1, 0],
            [s, 1],
        ]
    )


def controlled_Z(g: Scalar):
    r"""Controlled PHASE gate of two-gaussian modes.

    .. math::

        C_Z = \exp(ig q_1 \otimes q_2 / \hbar).


    Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 8.
    https://arxiv.org/pdf/1110.3234.pdf, Equation 161.

    Args:
        g (float): interaction strength

    Returns:
        the C_Z(g) matrix (in xxpp ordering)
    """

    return math.astensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, g, 1, 0],
            [g, 0, 0, 1],
        ]
    )


def controlled_X(g: Scalar):
    r"""Controlled NOT gate of two-gaussian modes.

    .. math::

        C_X = \exp(ig q_1 \otimes p_2).

    Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 9.

    Args:
        g (float): interaction strength

    Returns:
        the C_X(g) matrix (in xxpp ordering)
    """

    return math.astensor(
        [
            [1, 0, 0, 0],
            [g, 1, 0, 0],
            [0, 0, 1, -g],
            [0, 0, 0, 1],
        ]
    )


# ~~~~~~~~~~~~~
# CPTP channels
# ~~~~~~~~~~~~~


def CPTP(
    cov: Matrix,
    means: Vector,
    X: Matrix,
    Y: Matrix,
    d: Vector,
    state_modes: Sequence[int],
    transf_modes: Sequence[int],
) -> Tuple[Matrix, Vector]:
    r"""Returns the cov matrix and means vector of a state after undergoing a CPTP channel.

    Computed as ``cov = X \cdot cov \cdot X^T + Y`` and ``d = X \cdot means + d``.

    If the channel is single-mode, ``modes`` can contain ``M`` modes to apply the channel to,
    otherwise it must contain as many modes as the number of modes in the channel.

    Args:
        cov (Matrix): covariance matrix
        means (Vector): means vector
        X (Matrix): the X matrix of the CPTP channel
        Y (Matrix): noise matrix of the CPTP channel
        d (Vector): displacement vector of the CPTP channel
        state_modes (Sequence[int]): modes the state is defined on
        transf_modes (Sequence[int]): modes on which the channel acts
        hbar (float): value of hbar

    Returns:
        Tuple[Matrix, Vector]: the covariance matrix and the means vector of the state after the CPTP channel
    """
    if not set(transf_modes).issubset(state_modes):
        raise ValueError(
            f"The channel should act on a subset of the state modes ({transf_modes} is not a subset of {state_modes})"
        )
    # if single-mode channel, apply to all modes indicated in `modes`
    if X is not None and X.shape[-1] == 2:
        X = math.single_mode_to_multimode_mat(X, len(transf_modes))
    if Y is not None and Y.shape[-1] == 2:
        Y = math.single_mode_to_multimode_mat(Y, len(transf_modes))
    if d is not None and d.shape[-1] == 2:
        d = math.single_mode_to_multimode_vec(d, len(transf_modes))
    indices = [
        state_modes.index(i) for i in transf_modes
    ]  # TODO: do this when calling the method instead of here?
    cov = math.left_matmul_at_modes(X, cov, indices)
    cov = math.right_matmul_at_modes(cov, math.transpose(X), indices)
    cov = math.add_at_modes(cov, Y, indices)
    means = math.matvec_at_modes(X, means, indices)
    means = math.add_at_modes(means, d, indices)
    return cov, means


def loss_XYd(
    transmissivity: Union[Scalar, Vector], nbar: Union[Scalar, Vector], hbar: float
) -> Tuple[Matrix, Matrix, None]:
    r"""Returns the ``X``, ``Y`` matrices and the ``d`` vector for the noisy loss (attenuator) channel.

    .. math::

        X = math.sqrt(amplification)
        Y = (amplification - 1) * (2 * nbar + 1) * hbar / 2

    Reference: Alessio Serafini - Quantum Continuous Variables (5.77, p. 108)

    Args:
        transmissivity (float): value of the transmissivity, must be between 0 and 1
        nbar (float): photon number expectation value in the environment (0 for pure loss channel)

    Returns:
        Tuple[Matrix, Matrix, None]: the ``X``, ``Y`` matrices and the ``d`` vector for the noisy
        loss channel
    """
    if math.any(transmissivity < 0) or math.any(transmissivity > 1):
        raise ValueError("transmissivity must be between 0 and 1")
    x = math.sqrt(transmissivity)
    X = math.diag(math.concat([x, x], axis=0))
    y = (1 - transmissivity) * (2 * nbar + 1) * hbar / 2
    Y = math.diag(math.concat([y, y], axis=0))
    return X, Y, None


def amp_XYd(
    amplification: Union[Scalar, Vector], nbar: Union[Scalar, Vector], hbar: float
) -> Matrix:
    r"""Returns the ``X``, ``Y`` matrices and the d vector for the noisy amplifier channel.

    The quantum limited amplifier channel is recovered for ``nbar = 0.0``.

    Args:
        amplification (float): value of the amplification > 1
        nbar (float): photon number expectation value in the environment (0 for quantum
            limited amplifier)

    Returns:
        Tuple[Matrix, Vector]: the ``X``, ``Y`` matrices and the ``d`` vector for the noisy
        amplifier channel.
    """
    if math.any(amplification < 1):
        raise ValueError("Amplification must be larger than 1")
    x = math.sqrt(amplification)
    X = math.diag(math.concat([x, x], axis=0))
    y = (amplification - 1) * (2 * nbar + 1) * hbar / 2
    Y = math.diag(math.concat([y, y], axis=0))
    return X, Y, None


def noise_Y(noise: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the ``X``, ``Y`` matrices and the d vector for the additive noise channel ``(Y = noise * (\hbar / 2) * I)``

    Args:
        noise (float): number of photons in the thermal state

    Returns:
        Tuple[None, Matrix, None]: the ``X``, ``Y`` matrices and the ``d`` vector of the noise channel.
    """
    return math.diag(math.concat([noise, noise], axis=0)) * hbar / 2


def compose_channels_XYd(
    X1: Matrix, Y1: Matrix, d1: Vector, X2: Matrix, Y2: Matrix, d2: Vector
) -> Tuple[Matrix, Matrix, Vector]:
    r"""Returns the combined ``X``, ``Y``, and ``d`` for two CPTP channels.

    Args:
        X1 (Matrix): the ``X`` matrix of the first CPTP channel
        Y1 (Matrix): the ``Y`` matrix of the first CPTP channel
        d1 (Vector): the displacement vector of the first CPTP channel
        X2 (Matrix): the ``X`` matrix of the second CPTP channel
        Y2 (Matrix): the ``Y`` matrix of the second CPTP channel
        d2 (Vector): the displacement vector of the second CPTP channel

    Returns:
        Tuple[Matrix, Matrix, Vector]: the combined ``X``, ``Y``, and ``d`` matrices
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
    cov: Matrix,
    means: Vector,
    proj_cov: Matrix,
    proj_means: Vector,
    modes: Sequence[int],
    hbar: float,
) -> Tuple[Scalar, Matrix, Vector]:
    r"""Returns the results of a general dyne measurement.

    Args:
        cov (Matrix): covariance matrix of the state being measured
        means (Vector): means vector of the state being measured
        proj_cov (Matrix): covariance matrix of the state being projected onto
        proj_means (Vector): means vector of the state being projected onto (i.e. the measurement outcome)
        modes (Sequence[int]): modes being measured (modes are indexed from 0 to num_modes-1)

    Returns:
        Tuple[Scalar, Matrix, Vector]: the outcome probability, the post-measurement cov and means vector
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
    r"""Returns the photon number means vector given a Wigner covariance matrix and a means vector.

    Args:
        cov: the Wigner covariance matrix
        means: the Wigner means vector
        hbar: the value of the Planck constant

    Returns:
        Vector: the photon number means vector
    """
    N = means.shape[-1] // 2
    return (
        means[:N] ** 2
        + means[N:] ** 2
        + math.diag_part(cov[:N, :N])
        + math.diag_part(cov[N:, N:])
        - hbar
    ) / (2 * hbar)


def number_cov(cov: Matrix, means: Vector, hbar: float) -> Matrix:
    r"""Returns the photon number covariance matrix given a Wigner covariance matrix and a means vector.

    Args:
        cov: the Wigner covariance matrix
        means: the Wigner means vector
        hbar: the value of the Planck constant

    Returns:
        Matrix: the photon number covariance matrix
    """
    N = means.shape[-1] // 2
    mCm = cov * means[:, None] * means[None, :]
    dd = math.diag(math.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])) / (
        2 * hbar ** 2
    )
    CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
    return (
        CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * math.eye(N, dtype=CC.dtype)
    )


def is_mixed_cov(cov: Matrix) -> bool:  # TODO: deprecate
    r"""Returns ``True`` if the covariance matrix is mixed, ``False`` otherwise."""
    return not is_pure_cov(math.asnumpy(cov), hbar=settings.HBAR)


def auto_cutoffs(cov: Matrix, means: Vector, hbar: float) -> List[int]:
    r"""Automatically determines reasonable cutoffs.

    Args:
        cov: the covariance matrix
        means: the means vector
        hbar: the value of the Planck constant

    Returns:
        List[int]: a list of cutoff indices
    """
    cutoffs = (
        number_means(cov, means, hbar)
        + math.sqrt(math.diag(number_cov(cov, means, hbar))) * settings.N_SIGMA_CUTOFF
    )
    return [max(1, int(i)) for i in cutoffs]


def trace(cov: Matrix, means: Vector, Bmodes: Sequence[int]) -> Tuple[Matrix, Vector]:
    r"""Returns the covariances and means after discarding the specified modes.

    Args:
        cov (Matrix): covariance matrix
        means (Vector): means vector
        Bmodes (Sequence[int]): modes to discard

    Returns:
        Tuple[Matrix, Vector]: the covariance matrix and the means vector after discarding the specified modes
    """
    N = len(cov) // 2
    Aindices = math.astensor(
        [i for i in range(N) if i not in Bmodes] + [i + N for i in range(N) if i not in Bmodes]
    )
    A_cov_block = math.gather(math.gather(cov, Aindices, axis=0), Aindices, axis=1)
    A_means_vec = math.gather(means, Aindices)
    return A_cov_block, A_means_vec


def partition_cov(cov: Matrix, Amodes: Sequence[int]) -> Tuple[Matrix, Matrix, Matrix]:
    r"""Partitions the covariance matrix into the ``A`` and ``B`` subsystems and the AB coherence block.

    Args:
        cov (Matrix): the covariance matrix
        Amodes (Sequence[int]): the modes of system ``A``

    Returns:
        Tuple[Matrix, Matrix, Matrix]: the cov of ``A``, the cov of ``B`` and the AB block
    """
    N = cov.shape[-1] // 2
    Bindices = math.cast(
        [i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes],
        "int32",
    )
    Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
    A_block = math.gather(math.gather(cov, Aindices, axis=1), Aindices, axis=0)
    B_block = math.gather(math.gather(cov, Bindices, axis=1), Bindices, axis=0)
    AB_block = math.gather(math.gather(cov, Bindices, axis=1), Aindices, axis=0)
    return A_block, B_block, AB_block


def partition_means(means: Vector, Amodes: Sequence[int]) -> Tuple[Vector, Vector]:
    r"""Partitions the means vector into the ``A`` and ``B`` subsystems.

    Args:
        means (Vector): the means vector
        Amodes (Sequence[int]): the modes of system ``A``

    Returns:
        Tuple[Vector, Vector]: the means of ``A`` and the means of ``B``
    """
    N = len(means) // 2
    Bindices = math.cast(
        [i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes],
        "int32",
    )
    Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
    return math.gather(means, Aindices), math.gather(means, Bindices)


def purity(cov: Matrix, hbar: float) -> Scalar:
    r"""Returns the purity of the state with the given covariance matrix.

    Args:
        cov (Matrix): the covariance matrix

    Returns:
        float: the purity
    """
    return 1 / math.sqrt(math.det((2 / hbar) * cov))


def sympletic_eigenvals(cov: Matrix, hbar: float) -> Any:
    r"""Returns the sympletic eigenspectrum of a covariance matrix.

    For a pure state, we expect the sympletic eigenvalues to be 1.

    Args:
        cov (Matrix): the covariance matrix
        hbar (float): the value of the Planck constant

    Returns:
        List[float]: the sympletic eigenvalues
    """
    J = math.J(cov.shape[-1] // 2)  # create a sympletic form
    M = math.matmul(1j * J, cov * (2 / hbar))
    vals = math.eigvals(M)  # compute the eigenspectrum
    return math.abs(vals[::2])  # return the even eigenvalues  # TODO: sort?


def von_neumann_entropy(cov: Matrix, hbar: float) -> float:
    r"""Returns the Von Neumann entropy.

    For a pure state, we expect the Von Neumann entropy to be 0.

    Reference: (https://arxiv.org/pdf/1110.3234.pdf), Equations 46-47.

    Args:
        cov (Matrix): the covariance matrix

    Returns:
        float: the Von Neumann entropy
    """
    symp_vals = sympletic_eigenvals(cov, hbar)
    g = lambda x: math.xlogy((x + 1) / 2, (x + 1) / 2) - math.xlogy((x - 1) / 2, (x - 1) / 2 + 1e-9)
    entropy = math.sum(g(symp_vals))
    return entropy


def fidelity(
    mu1: Vector, cov1: Matrix, mu2: Vector, cov2: Matrix, hbar=2.0, rtol=1e-05, atol=1e-08
) -> float:
    r"""Returns the fidelity of two gaussian states.

    Reference: `arXiv:2102.05748 <https://arxiv.org/pdf/2102.05748.pdf>`_, equations 95-99.
    Note that we compute the square of equation 98.

    Args:
        mu1 (Vector): the means vector of state 1
        mu2 (Vector): the means vector of state 2
        cov1 (Matrix): the covariance matrix of state 1
        cov1 (Matrix): the covariance matrix of state 2

    Returns:
        float: the fidelity
    """

    cov1 = math.cast(cov1 / hbar, "complex128")  # convert to units where hbar = 1
    cov2 = math.cast(cov2 / hbar, "complex128")  # convert to units where hbar = 1

    mu1 = math.cast(mu1, "complex128")
    mu2 = math.cast(mu2, "complex128")
    deltar = (mu2 - mu1) / math.sqrt(hbar, dtype=mu1.dtype)  # convert to units where hbar = 1
    J = math.J(cov1.shape[0] // 2)
    I = math.eye(cov1.shape[0])
    J = math.cast(J, "complex128")
    I = math.cast(I, "complex128")

    cov12_inv = math.inv(cov1 + cov2)

    V = math.transpose(J) @ cov12_inv @ ((1 / 4) * J + cov2 @ J @ cov1)

    W = -2 * (V @ (1j * J))
    W_inv = math.inv(W)
    matsqrtm = math.sqrtm(
        I - W_inv @ W_inv
    )  # this also handles the case where the input matrix is close to zero
    f0_top = math.det((matsqrtm + I) @ (W @ (1j * J)))
    f0_bot = math.det(cov1 + cov2)

    f0 = (f0_top / f0_bot) ** (1 / 2)  # square of equation 98

    dot = math.sum(
        math.transpose(deltar) * math.matvec(cov12_inv, deltar)
    )  # computing (mu2-mu1)/sqrt(hbar).T @ cov12_inv @ (mu2-mu1)/sqrt(hbar)

    fidelity = f0 * math.exp((-1 / 2) * dot)  # square of equation 95

    return math.cast(fidelity, "float64")


def physical_partial_transpose(cov: Matrix, modes: Sequence[int]) -> Matrix:
    r"""Returns the covariance matrix that corresponds to applying the partial
    transposition on the density matrix of a given set of modes.

    Reference: `https://arxiv.org/abs/quant-ph/9909044 <https://arxiv.org/abs/quant-ph/9909044>`_, Equation 1, 5.

    Args:
        cov (Matrix): the covariance matrix
        modes (Sequence[int]): the modes of system on which transposition is applied

    Returns:
        Matrix: the covariance matrix corresponding to the partially transposed state
    """
    m, _ = cov.shape
    num_modes = m // 2
    mat = [1.0] * m
    for i in modes:
        mat[i + num_modes] = -1.0
    mat = math.astensor(mat, dtype="float64")
    return cov * mat[:, None] * mat[None, :]


def log_negativity(cov: Matrix, hbar: float) -> float:
    r"""Returns the log_negativity of a Gaussian state.

    Reference: `https://arxiv.org/pdf/quant-ph/0102117.pdf <https://arxiv.org/pdf/quant-ph/0102117.pdf>`_ , Equation 57, 61.

    Args:
        cov (Matrix): the covariance matrix

    Returns:
        float: the log-negativity
    """
    vals = sympletic_eigenvals(cov, hbar) / (hbar / 2)
    vals_filtered = math.boolean_mask(
        vals, vals < 1.0
    )  # Get rid of terms that would lead to zero contribution.
    if len(vals_filtered) > 0:
        return -math.sum(
            math.log(vals_filtered) / math.cast(math.log(2.0), dtype=vals_filtered.dtype)
        )
    else:
        return 0


def join_covs(covs: Sequence[Matrix]) -> Tuple[Matrix, Vector]:
    r"""Joins the given covariance matrices into a single covariance matrix.

    Args:
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
    r"""Joins the given means vectors into a single means vector.

    Args:
        means (Sequence[Vector]): the means vectors

    Returns:
        Vector: the joined means vector
    """
    mean = XPVector.from_xxpp(means[0], modes=list(range(len(means[0]) // 2)))
    for i, m in enumerate(means[1:]):
        mean = mean + XPVector.from_xxpp(
            m, modes=list(range(mean.num_modes, mean.num_modes + len(m) // 2))
        )
    return mean.to_xxpp()


def symplectic_inverse(S: Matrix) -> Matrix:
    r"""Returns the inverse of a symplectic matrix.

    Args:
        S (Matrix): the symplectic matrix

    Returns:
        Matrix: the inverse of the symplectic matrix
    """
    S = math.reshape(S, (S.shape[0] // 2, 2, S.shape[1] // 2, 2))
    S = math.transpose(S, (1, 3, 0, 2))
    return math.block(
        [
            [math.transpose(S[1, 1]), -math.transpose(S[0, 1])],
            [-math.transpose(S[1, 0]), math.transpose(S[0, 0])],
        ]
    )


def XYd_dual(X: Matrix, Y: Matrix, d: Vector):
    r"""Returns the dual channel ``(X, Y, d)``.

    Args:
        X (Matrix): the ``X`` matrix
        Y (Matrix): the ``Y`` noise matrix
        d (Vector): the displacement vector

    Returns:
        Tuple[Matrix, Matrix, Vector]: ``(X_dual, Y_dual, d_dual)``
    """
    X_dual = math.inv(X) if X is not None else None
    Y_dual = Y
    d_dual = d
    if Y is not None:
        Y_dual = (
            math.matmul(X_dual, math.matmul(Y, math.transpose(X_dual))) if X_dual is not None else Y
        )
    if d is not None:
        d_dual = math.matvec(X_dual, d) if X_dual is not None else d
    return X_dual, Y_dual, d_dual

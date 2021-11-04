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
import importlib

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Symplectic matrices and displacement vectors
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


def loss_X(transmissivity: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = math.sqrt(transmissivity)
    return math.diag(math.concat([D, D], axis=0))


def loss_Y(transmissivity: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = (1.0 - transmissivity) * hbar / 2
    return math.diag(math.concat([D, D], axis=0))


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


#  ~~~~~~~~~~~~~~~~~~~~~
#  Covariances and means
#  ~~~~~~~~~~~~~~~~~~~~~


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

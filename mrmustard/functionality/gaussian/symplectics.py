from math import pi, sqrt
from thewalrus.quantum import is_pure_cov
from mrmustard._typing import *
from mrmustard import Backend, XPTensor
backend = Backend()


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

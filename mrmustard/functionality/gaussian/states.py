from math import pi, sqrt
from thewalrus.quantum import is_pure_cov
from mrmustard._typing import *
from mrmustard.functionality.gaussian.channels import CPTP
from mrmustard.functionality.gaussian.symplectics import displacement
from mrmustard import Backend, XPTensor
backend = Backend()


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
    d = displacement(x, y, hbar)
    cov, means = vacuum_state(len(d)//2, hbar)
    return CPTP(cov, means, XPTensor(), XPTensor(), d)
    
    
    # num_modes = x.shape[-1]
    # cov = backend.eye(num_modes * 2, dtype=x.dtype) * hbar / 2
    # means = backend.concat([x, y], axis=0) * backend.sqrt(2 * hbar, dtype=x.dtype)
    # return cov, means


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
    cov, means = vacuum_state(len(d)//2, hbar)
    # cov = backend.matmul(S, backend.transpose(S)) * hbar / 2
    # means = backend.zeros(cov.shape[-1], dtype=cov.dtype)
    # return cov, means


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

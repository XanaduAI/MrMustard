from math import pi, sqrt
from thewalrus.quantum import is_pure_cov
from mrmustard._typing import *
from mrmustard.functionality.gaussian.channels import CPTP
from mrmustard.functionality.gaussian.symplectics import displacement, squeezing_symplectic, two_mode_squeezing_symplectic
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
    return XPTensor(cov), XPTensor(means)


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
    return CPTP(X=XPTensor(multiplicative=True), d=displacement(x, y, hbar))


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
    return CPTP(X=squeezing_symplectic(r, phi))


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
    cov = XPTensor(backend.diag(backend.concat([g, g], axis=-1)), multiplicative=True)
    means = XPTensor(backend.zeros(cov.shape[-1], dtype=cov.dtype), additive=True)
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
    return CPTP(X=squeezing_symplectic(r, phi), d=displacement(x, y, hbar))


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
    return CPTP(X=two_mode_squeezing_symplectic(r, phi))

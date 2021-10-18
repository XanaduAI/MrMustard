from mrmustard import Backend
from mrmustard._typing import *
import numpy as np

r"""
A plugin that interfaces the phase space representation with the Fock representation.

It implements:
- fock_representation and its gradient
- classical stochastic channels
"""

backend = Backend()


def fock_representation(cov: Matrix, means: Vector, cutoffs: Sequence[int], mixed: bool, hbar: float) -> Tensor:
    r"""
    Returns the Fock representation of the phase space representation
    given a Wigner covariance matrix and a means vector. If the state is pure
    it returns the ket, if it is mixed it returns the density matrix.
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        cutoffs: The shape of the tensor.
        mixed: Whether the state vector is mixed or not.
        hbar: The value of the Planck constant.
    Returns:
        The Fock representation of the phase space representation.
    """
    assert len(cutoffs) == means.shape[-1] // 2 == cov.shape[-1] // 2
    A, B, C = hermite_parameters(cov, means, mixed, hbar)
    return backend.hermite_renormalized(backend.conj(-A), backend.conj(B), backend.conj(C), shape=cutoffs + cutoffs * mixed)


def ket_to_dm(ket: Tensor) -> Tensor:
    r"""
    Maps a ket to a density matrix.
    Args:
        ket: The ket.
    Returns:
        The density matrix.
    """
    return backend.outer(ket, backend.conj(ket))


def ket_to_probs(ket: Tensor) -> Tensor:
    r"""
    Maps a ket to probabilities.
    Args:
        ket: The ket.
    Returns:
        The probabilities vector.
    """
    return backend.abs(ket) ** 2


def dm_to_probs(dm: Tensor) -> Tensor:
    r"""
    Extracts the diagonals of a density matrix.
    Args:
        dm: The density matrix.
    Returns:
        The probabilities vector.
    """
    return backend.all_diagonals(dm, real=True)


def hermite_parameters(cov: Matrix, means: Vector, mixed: bool, hbar: float) -> Tuple[Matrix, Vector, Scalar]:
    r"""
    Returns the A matrix, B vector and C scalar given a Wigner covariance matrix and a means vector of an N-mode state.
    The A, B, C triple is needed to compute the Fock representation of the state.
    If the state is pure, then A has shape (N, N), B has shape (N) and C has shape ().
    If the state is mixed, then A has shape (2N, 2N), B has shape (2N) and C has shape ().
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        mixed: Whether the state vector is mixed or not.
        hbar: The value of the Planck constant.
    Returns:
        The A matrix, B vector and C scalar.
    """
    num_indices = means.shape[-1]
    num_modes = num_indices // 2

    # cov and means in the amplitude basis
    R = backend.rotmat(num_indices // 2)
    sigma = backend.matmul(backend.matmul(R, cov / hbar), backend.dagger(R))
    beta = backend.matvec(R, means / backend.sqrt(hbar, dtype=means.dtype))

    sQ = sigma + 0.5 * backend.eye(num_indices, dtype=sigma.dtype)
    sQinv = backend.inv(sQ)
    X = backend.Xmat(num_modes)
    A = backend.matmul(X, backend.eye(num_indices, dtype=sQinv.dtype) - sQinv)
    B = backend.matvec(backend.transpose(sQinv), backend.conj(beta))
    exponent = -0.5 * backend.sum(backend.conj(beta)[:, None] * sQinv * beta[None, :])
    T = backend.exp(exponent) / backend.sqrt(backend.det(sQ))
    N = 2 * num_modes if mixed else num_modes
    return (
        A[:N, :N],
        B[:N],
        T ** (1.0 if mixed else 0.5),
    )  # will be off by global phase because T is real even for pure states


def fidelity(state_a, state_b, a_pure: bool = True, b_pure: bool = True) -> Scalar:
    r"""computes the fidelity between two states in Fock representation"""
    if a_pure and b_pure:
        return backend.sum(backend.abs(state_a * state_b) ** 2)
    elif a_pure:
        a = state_a.reshape(-1)
        return backend.real(backend.sum(backend.conj(a) * backend.matvec(backend.reshape(state_b, (len(a), len(a))), a)))
    elif b_pure:
        b = state_b.reshape(-1)
        return backend.real(backend.sum(backend.conj(b) * backend.matvec(state_a, b)))
    else:
        return backend.sum(backend.abs(state_a * state_b))

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

import numpy as np

from mrmustard.utils.types import *
from mrmustard import settings
from mrmustard.math import Math

math = Math()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ static functions ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def fock_state(n: Sequence[int]) -> Tensor:
    r"""
    Returns a pure or mixed Fock state.
    Args:
        n: a list of photon numbers.
    Returns:
        The Fock state up to cutoffs n+1
    """
    psi = np.zeros(np.array(n) + np.ones_like(n), dtype=np.complex128)
    psi[tuple(np.atleast_1d(n))] = 1
    return psi


def autocutoffs(
    number_variances: Matrix, number_means: Vector, max_cutoff: int = settings.AUTOCUTOFF_MAX_CUTOFF
) -> Tuple[int, ...]:
    r"""
    Returns the autocutoffs of a Wigner state.
    Arguments:
        number_vars: The number variances in each mode (i.e. the diagonal of the covariance matrix)
        number_means: The number means vector.
        max_cutoff: The maximum cutoff.
    Returns:
        The suggested cutoffs.
    """
    autocutoffs = math.cast(number_means + math.sqrt(number_variances) * settings.AUTOCUTOFF_SIGMA_FACTOR, "int32")
    return [int(n) for n in math.clip(autocutoffs, 1, max_cutoff)]


def fock_representation(
    cov: Matrix,
    means: Vector,
    shape: Sequence[int],
    is_mixed: bool = None,
    is_unitary: bool = None,
    choi_r: float = None,
) -> Tensor:
    r"""
    Returns the Fock representation of a state or Choi state.
    If the state is pure it returns the state vector (ket).
    If the state is mixed it returns the density matrix.
    If the transformation is unitary it returns the unitary transformation matrix.
    If the transformation is not unitary it returns the Choi matrix.
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        shape: The shape of the tensor.
        is_mixed: Whether the state vector is mixed or not.
        is_unitary: Whether the transformation is unitary or not.
        choi_r: The TMSV squeezing magnitude.
    Returns:
        The Fock representation.
    """
    if is_mixed is not None and is_unitary is not None:
        raise ValueError("Cannot specify both mixed and unitary.")
    if is_mixed is None and is_unitary is None:
        raise ValueError("Must specify either mixed or unitary.")
    if is_unitary is not None and choi_r is None:
        raise ValueError("Must specify the choi_r value.")
    if is_mixed is not None:  # i.e. it's a state
        A, B, C = ABC(cov, means, full=is_mixed)
    elif is_unitary is not None and choi_r is not None:  # i.e. it's a transformation
        A, B, C = ABC(cov, means, full=not is_unitary, choi_r=choi_r)
    return math.hermite_renormalized(
        math.conj(-A), math.conj(B), math.conj(C), shape=shape
    )  # NOTE: remove conj when TW is updated


def ket_to_dm(ket: Tensor) -> Tensor:
    r"""
    Maps a ket to a density matrix.
    Args:
        ket: The ket.
    Returns:
        The density matrix.
    """
    return math.outer(ket, math.conj(ket))


def ket_to_probs(ket: Tensor) -> Tensor:
    r"""
    Maps a ket to probabilities.
    Args:
        ket: The ket.
    Returns:
        The probabilities vector.
    """
    return math.abs(ket) ** 2


def dm_to_probs(dm: Tensor) -> Tensor:
    r"""
    Extracts the diagonals of a density matrix.
    Args:
        dm: The density matrix.
    Returns:
        The probabilities vector.
    """
    return math.all_diagonals(dm, real=True)


def U_to_choi(U: Tensor) -> Tensor:
    r"""
    Converts a unitary transformation to a Choi tensor.
    Args:
        U: The unitary transformation.
    Returns:
        The Choi tensor.
    """
    cutoffs = U.shape[: len(U.shape) // 2]
    N = len(cutoffs)
    outer = math.outer(U, math.conj(U))
    choi = math.transpose(
        outer,
        list(range(0, N))
        + list(range(2 * N, 3 * N))
        + list(range(N, 2 * N))
        + list(range(3 * N, 4 * N)),
    )
    return choi


def ABC(cov, means, full: bool, choi_r: float = None) -> Tuple[Matrix, Vector, Scalar]:
    r"""
    Returns the full-size A matrix, B vector and C scalar.
    Arguments:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        full: Whether to return the full-size A, B and C or the half-size A, B and C.
        choi_r: The TMSV squeezing magnitude if not None we consider ABC of a Choi state.
    """
    is_state = choi_r is None
    N = cov.shape[-1] // 2
    R = math.rotmat(N)
    sigma = math.matmul(math.matmul(R, cov / settings.HBAR), math.dagger(R))
    beta = math.matvec(R, means / math.sqrt(settings.HBAR, dtype=means.dtype))
    Q = sigma + 0.5 * math.eye(2 * N, dtype=sigma.dtype)  # Husimi covariance matrix
    Qinv = math.inv(Q)
    A = math.matmul(math.Xmat(N), math.eye(2 * N, dtype=Qinv.dtype) - Qinv)
    denom = math.sqrt(math.det(Q)) if is_state else math.sqrt(math.det(Q / np.cosh(choi_r)))
    if full:
        B = math.matvec(math.transpose(Qinv), math.conj(beta))
        exponent = -0.5 * math.sum(math.conj(beta)[:, None] * Qinv * beta[None, :])
        C = math.exp(exponent) / denom
    else:
        A = A[
            :N, :N
        ]  # TODO: find a way to compute the half-size A without computing the full-size A first
        B = beta[N:] - math.matvec(A, beta[:N])
        exponent = -0.5 * math.sum(beta[:N] * B)
        C = math.exp(exponent) / math.sqrt(denom)
    if choi_r is not None:
        ones = math.ones(
            N // 2, dtype=A.dtype
        )  # N//2 is the actual number of modes because of the choi trick
        factor = 1.0 / np.tanh(choi_r)
        if full:
            rescaling = math.concat([ones, factor * ones, ones, factor * ones], axis=0)
        else:
            rescaling = math.concat([ones, factor * ones], axis=0)
        A = rescaling[:, None] * rescaling[None, :] * A
        B = rescaling * B
    return A, B, C


def fidelity(state_a, state_b, a_pure: bool = True, b_pure: bool = True) -> Scalar:
    r"""computes the fidelity between two states in Fock representation"""
    if a_pure and b_pure:
        min_cutoffs = tuple([slice(min(a, b)) for a, b in zip(state_a.shape, state_b.shape)])
        state_a = state_a[min_cutoffs]
        state_b = state_b[min_cutoffs]
        return math.abs(math.sum(math.conj(state_a) * state_b)) ** 2
    elif a_pure:
        min_cutoffs = tuple(
            [
                slice(min(a, b))
                for a, b in zip(state_a.shape, state_b.shape[: len(state_b.shape) // 2])
            ]
        )
        state_a = state_a[min_cutoffs]
        state_b = state_b[min_cutoffs]
        a = math.reshape(state_a, -1)
        return math.real(
            math.sum(math.conj(a) * math.matvec(math.reshape(state_b, (len(a), len(a))), a))
        )
    elif b_pure:
        min_cutoffs = tuple(
            [
                slice(min(a, b))
                for a, b in zip(state_a.shape[: len(state_a.shape) // 2], state_b.shape)
            ]
        )
        state_a = state_a[min_cutoffs]
        state_b = state_b[min_cutoffs]
        b = math.reshape(state_b, -1)
        return math.real(
            math.sum(math.conj(b) * math.matvec(math.reshape(state_a, (len(b), len(b))), b))
        )
    else:
        raise NotImplementedError("Fidelity between mixed states is not implemented yet.")


def number_means(tensor, is_dm: bool):
    r"""
    returns the first moment of the number operator
    """
    probs = math.all_diagonals(tensor) if is_dm else math.abs(tensor) ** 2
    modes = [m for m in range(len(probs.shape))]
    marginals = [math.sum(probs, axes=modes[:k] + modes[k+1:]) for k in range(len(modes))]
    return math.astensor(
        [
            math.sum(marginal * math.arange(len(marginal), dtype=marginal.dtype))
            for marginal in marginals
        ]
    )

def number_variances(tensor, is_dm: bool):
    r"""
    returns the second moment of the number operator in each mode
    """
    probs = math.all_diagonals(tensor) if is_dm else math.abs(tensor) ** 2
    modes = [m for m in range(len(probs.shape))]
    marginals = [math.sum(probs, axes=modes[:k] + modes[k+1:]) for k in range(len(modes))]
    return math.astensor(
            [
                (math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype)**2) - 
                 math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype))**2)
                for marginal in marginals
            ]
    )


def purity(dm: Tensor) -> Scalar:
    r"""Computes the purity of a state in Fock representation."""
    cutoffs = dm.shape[: len(dm.shape) // 2]
    d = int(np.prod(cutoffs))  # combined cutoffs in all modes
    dm = math.reshape(dm, (d, d))
    return math.abs(math.sum(math.transpose(dm) * dm))  # tr(rho^2)


def CPTP(
    transformation, fock_state, transformation_is_unitary: bool, state_is_mixed: bool
) -> Tensor:
    r"""computes the CPTP (# NOTE: CP, really) channel given by a transformation (unitary matrix or choi operator) on a state.
    It assumes that the cutoffs of the transformation matche the cutoffs of the relevant axes of the state.
    Arguments:
        transformation: The transformation tensor.
        fock_state: The state to transform.
        transformation_is_unitary: Whether the transformation is a unitary matrix or a Choi operator.
        state_is_mixed: Whether the state is mixed or not.
    Returns:
        The transformed state.
    """
    num_modes = len(fock_state.shape) // 2 if state_is_mixed else len(fock_state.shape)
    indices = list(range(num_modes))
    if transformation_is_unitary:
        U = transformation
        Us = math.tensordot(U, fock_state, axes=([num_modes + s for s in indices], indices))
        if state_is_mixed:
            UsU = math.tensordot(
                Us, math.dagger(U), axes=([num_modes + s for s in indices], indices)
            )
            return UsU
        else:
            return Us
    else:
        C = transformation
        Cs = math.tensordot(C, fock_state, axes=([-s for s in reversed(indices)], indices))
        if state_is_mixed:
            return Cs
        else:
            Css = math.tensordot(
                Cs, math.conj(fock_state), axes=([-s for s in reversed(indices)], indices)
            )
            return Css


def POVM(
    fock_state: Tensor,
    is_state_dm: bool,
    POVM_effect: Sequence[Optional[int]],
    modes: Sequence[int],
) -> Union[Tensor, Scalar]:
    r"""
    Computes <POVM_effect|fock_state> if fock_state is a ket or <POVM_effect|fock_state|POVM_effect> if it is a dm.
    Arguments:
        fock_state: The state to project.
        is_state_dm: Whether fock_state is a density matrix or not.
        POVM_effect: The POVM effect to apply.
        modes: The modes of the state on which to apply the POVM effect.
    Returns:
        The unnormalized projected state or the projection probability if there is no leftover state.
    """
    POVM_modes = list(range(len(POVM_effect.shape)))
    if not is_state_dm:
        return math.tensordot(math.conj(POVM_effect), fock_state, axes=(POVM_modes, modes))
    else:
        return math.tensordot(
            math.conj(POVM_effect),
            math.tensordot(
                fock_state,
                POVM_effect,
                axes=([m + len(fock_state.shape) // 2 for m in modes], POVM_modes),
            ),
            axes=(POVM_modes, modes),
        )


def normalize(fock: Tensor, is_mixed: bool):
    if is_mixed:
        return fock / math.sum(math.all_diagonals(fock, real=False))
    else:
        return fock / math.sum(math.norm(fock))

def is_mixed_dm(dm):
    cutoffs = dm.shape[: len(dm.shape) // 2]
    square = math.reshape(dm, (int(np.prod(cutoffs)), -1))
    return not np.isclose(math.sum(square * math.transpose(square)), 1.0)

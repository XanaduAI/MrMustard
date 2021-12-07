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

from mrmustard.types import *
from mrmustard import settings
from mrmustard.math import Math

math = Math()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ static functions ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def fock_state(n: Sequence[int]) -> Tensor:
    r"""Returns a pure or mixed Fock state.

    Args:
        n: a list of photon numbers

    Returns:
        the Fock state up to cutoffs ``n+1``
    """
    psi = np.zeros(np.array(n) + np.ones_like(n), dtype=np.complex128)
    psi[tuple(np.atleast_1d(n))] = 1
    return psi


def autocutoffs(
    number_stdev: Matrix, number_means: Vector, max_cutoff: int = None, min_cutoff: int = None
) -> Tuple[int, ...]:
    r"""Returns the autocutoffs of a Wigner state.

    Args:
        number_stdev: the photon number standard deviation in each mode
            (i.e. the square root of the diagonal of the covariance matrix)
        number_means: the photon number means vector
        max_cutoff: the maximum cutoff

    Returns:
        Tuple[int, ...]: the suggested cutoffs
    """
    if max_cutoff is None:
        max_cutoff = settings.AUTOCUTOFF_MAX_CUTOFF
    if min_cutoff is None:
        min_cutoff = settings.AUTOCUTOFF_MIN_CUTOFF
    autocutoffs = settings.AUTOCUTOFF_MIN_CUTOFF + math.cast(
        number_means + number_stdev * settings.AUTOCUTOFF_STDEV_FACTOR, "int32"
    )
    return [int(n) for n in math.clip(autocutoffs, min_cutoff, max_cutoff)]


def fock_representation(
    cov: Matrix,
    means: Vector,
    shape: Sequence[int],
    return_dm: bool = None,
    return_unitary: bool = None,
    choi_r: float = None,
) -> Tensor:
    r"""Returns the Fock representation of a state or Choi state.

    * If the state is pure it returns the state vector (ket).
    * If the state is mixed it returns the density matrix.
    * If the transformation is unitary it returns the unitary transformation matrix.
    * If the transformation is not unitary it returns the Choi matrix.

    Args:
        cov: the Wigner covariance matrix
        means: the Wigner means vector
        shape: the shape of the tensor
        return_dm: whether the state vector is mixed or not
        return_unitary: whether the transformation is unitary or not
        choi_r: the TMSV squeezing magnitude

    Returns:
        Tensor: the fock representation
    """
    if return_dm is not None and return_unitary is not None:
        raise ValueError("Cannot specify both mixed and unitary.")
    if return_dm is None and return_unitary is None:
        raise ValueError("Must specify either mixed or unitary.")
    if return_unitary is not None and choi_r is None:
        raise ValueError("Must specify the choi_r value.")
    if return_dm is not None:  # i.e. it's a state
        A, B, C = ABC(cov, means, full=return_dm)
    elif return_unitary is not None and choi_r is not None:  # i.e. it's a transformation
        A, B, C = ABC(cov, means, full=not return_unitary, choi_r=choi_r)
    return math.hermite_renormalized(
        math.conj(-A), math.conj(B), math.conj(C), shape=shape
    )  # NOTE: remove conj when TW is updated


def ket_to_dm(ket: Tensor) -> Tensor:
    r"""Maps a ket to a density matrix.

    Args:
        ket: the ket

    Returns:
        Tensor: the density matrix
    """
    return math.outer(ket, math.conj(ket))


def ket_to_probs(ket: Tensor) -> Tensor:
    r"""Maps a ket to probabilities.

    Args:
        ket: the ket

    Returns:
        Tensor: the probabilities vector
    """
    return math.abs(ket) ** 2


def dm_to_probs(dm: Tensor) -> Tensor:
    r"""Extracts the diagonals of a density matrix.

    Args:
        dm: the density matrix

    Returns:
        Tensor: the probabilities vector
    """
    return math.all_diagonals(dm, real=True)


def U_to_choi(U: Tensor) -> Tensor:
    r"""Converts a unitary transformation to a Choi tensor.

    Args:
        U: the unitary transformation

    Returns:
        Tensor: the Choi tensor
    """
    cutoffs = U.shape[: len(U.shape) // 2]
    N = len(cutoffs)
    outer = math.outer(U, math.conj(U))
    return math.transpose(
        outer,
        list(range(0, N))
        + list(range(2 * N, 3 * N))
        + list(range(N, 2 * N))
        + list(range(3 * N, 4 * N)),
    )  # NOTE: mode blocks 1 and 3 are at the end so we can tensordot dm with them


def ABC(cov, means, full: bool, choi_r: float = None) -> Tuple[Matrix, Vector, Scalar]:
    r"""Returns the full-size ``A`` matrix, ``B`` vector and ``C`` scalar.

    Args:
        cov: the Wigner covariance matrix
        means: the Wigner means vector
        full: whether to return the full-size ``A``, ``B`` and ``C`` or the half-size ``A``, ``B``
            and ``C``
        choi_r: the TMSV squeezing magnitude if not None we consider ABC of a Choi state

    Returns:
        Tuple[Matrix, Vector, Scalar]: full-size ``A`` matrix, ``B`` vector and ``C`` scalar
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


def fidelity(state_a, state_b, a_ket: bool, b_ket: bool) -> Scalar:
    r"""Computes the fidelity between two states in Fock representation."""
    if a_ket and b_ket:
        min_cutoffs = tuple([slice(min(a, b)) for a, b in zip(state_a.shape, state_b.shape)])
        state_a = state_a[min_cutoffs]
        state_b = state_b[min_cutoffs]
        return math.abs(math.sum(math.conj(state_a) * state_b)) ** 2
    elif a_ket:
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
    elif b_ket:
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
    r"""Returns the mean of the number operator in each mode."""
    probs = math.all_diagonals(tensor, real=True) if is_dm else math.abs(tensor) ** 2
    modes = [m for m in range(len(probs.shape))]
    marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
    return math.astensor(
        [
            math.sum(marginal * math.arange(len(marginal), dtype=marginal.dtype))
            for marginal in marginals
        ]
    )


def number_variances(tensor, is_dm: bool):
    r"""Returns the variance of the number operator in each mode."""
    probs = math.all_diagonals(tensor, real=True) if is_dm else math.abs(tensor) ** 2
    modes = [m for m in range(len(probs.shape))]
    marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
    return math.astensor(
        [
            (
                math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype) ** 2)
                - math.sum(marginal * math.arange(marginal.shape[0], dtype=marginal.dtype)) ** 2
            )
            for marginal in marginals
        ]
    )


def purity(dm: Tensor) -> Scalar:
    r"""Returns the purity of a density matrix."""
    cutoffs = dm.shape[: len(dm.shape) // 2]
    d = int(np.prod(cutoffs))  # combined cutoffs in all modes
    dm = math.reshape(dm, (d, d))
    dm = normalize(dm, is_dm=True)
    return math.abs(math.sum(math.transpose(dm) * dm))  # tr(rho^2)


def CPTP(transformation, fock_state, transformation_is_unitary: bool, state_is_dm: bool) -> Tensor:
    r"""Computes the CPTP (note: CP, really) channel given by a transformation (unitary matrix or choi operator) on a state.

    It assumes that the cutoffs of the transformation matches the cutoffs of the relevant axes of the state.

    Args:
        transformation: the transformation tensor
        fock_state: the state to transform
        transformation_is_unitary: whether the transformation is a unitary matrix or a Choi operator
        state_is_dm: whether the state is a density matrix or a ket

    Returns:
        Tensor: the transformed state
    """
    num_modes = len(fock_state.shape) // 2 if state_is_dm else len(fock_state.shape)
    N0 = list(range(0, num_modes))
    N1 = list(range(num_modes, 2 * num_modes))
    N2 = list(range(2 * num_modes, 3 * num_modes))
    N3 = list(range(3 * num_modes, 4 * num_modes))
    if transformation_is_unitary:
        U = transformation
        Us = math.tensordot(U, fock_state, axes=(N1, N0))
        if not state_is_dm:
            return Us
        else:  # is state is dm, the input indices of dm are still at the end of Us
            return math.tensordot(Us, math.dagger(U), axes=(N1, N0))
    else:
        C = transformation  # choi operator
        if state_is_dm:
            return math.tensordot(C, fock_state, axes=(N1 + N3, N0 + N1))
        else:
            Cs = math.tensordot(C, fock_state, axes=(N1, N0))
            return math.tensordot(
                Cs, math.conj(fock_state), axes=(N2, N0)
            )  # N2 is the last set of indices now


def contract_states(
    stateA, stateB, a_is_mixed: bool, b_is_mixed: bool, modes: List[int], normalize: bool
):
    r"""Contracts two states in the specified modes, it assumes that the modes spanned by ``B`` are a subset of the modes spanned by ``A``.

    Args:
        stateA: the first state
        stateB: the second state (assumed to be on a subset of the modes of stateA)
        a_is_mixed: whether the first state is mixed or not.
        b_is_mixed: whether the second state is mixed or not.
        modes: the modes on which to contract the states.
        normalize: whether to normalize the result

    Returns:
        State: the contracted state (subsystem of ``A``)
    """
    indices = list(range(len(modes)))
    if not a_is_mixed and not b_is_mixed:
        out = math.tensordot(math.conj(stateB), stateA, axes=(indices, modes))
        if normalize:
            out = out / math.norm(out)
        return out
    elif a_is_mixed and not b_is_mixed:
        Ab = math.tensordot(
            stateA, stateB, axes=([m + len(stateA.shape) // 2 for m in modes], indices)
        )
        out = math.tensordot(math.conj(stateB), Ab, axes=(indices, modes))
    elif not a_is_mixed and b_is_mixed:
        Ba = math.tensordot(stateB, stateA, axes=(indices, modes))  # now B indices are all first
        out = math.tensordot(math.conj(stateA), Ba, axes=(modes, indices))
    elif a_is_mixed and b_is_mixed:
        out = math.tensordot(
            stateA,
            math.conj(stateB),
            axes=(
                modes + [m + len(stateA.shape) // 2 for m in modes],
                indices + [i + len(stateB.shape) // 2 for i in indices],
            ),
        )
    if normalize:
        out = out / math.sum(math.all_diagonals(out, real=False))
    return out


def normalize(fock: Tensor, is_dm: bool):
    if is_dm:
        return fock / math.sum(math.all_diagonals(fock, real=False))
    else:
        return fock / math.sum(math.norm(fock))


def is_mixed_dm(dm):
    cutoffs = dm.shape[: len(dm.shape) // 2]
    square = math.reshape(dm, (int(np.prod(cutoffs)), -1))
    return not np.isclose(math.sum(square * math.transpose(square)), 1.0)


def trace(dm, keep: List[int]):
    r"""Computes the partial trace of a density matrix.

    Args:
        dm: the density matrix
        keep: the modes to keep
    """
    N = len(dm.shape) // 2
    trace = [m for m in range(N) if m not in keep]
    # put at the end all of the indices to trace over
    dm = math.transpose(
        dm, [i for pair in [(k, k + N) for k in keep] + [(t, t + N) for t in trace] for i in pair]
    )
    d = int(np.prod(dm.shape[-len(trace) :]))
    # make it square on those indices
    dm = math.reshape(dm, dm.shape[: 2 * len(keep)] + (d, d))
    return math.trace(dm)

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

# pylint: disable=redefined-outer-name

"""
This module contains functions for performing calculations on Fock states.
"""

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numba import jit

from mrmustard import settings
from mrmustard.math import Math
from mrmustard.math.caching import tensor_int_cache
from mrmustard.math.mmtensor import MMTensor
from mrmustard.math.numba.compactFock_diagonal_amps import fock_representation_diagonal_amps
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
)
from mrmustard.typing import Matrix, Scalar, Tensor, Vector

math = Math()
SQRT = np.sqrt(np.arange(1e6))

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


def autocutoffs(cov: Matrix, means: Vector, probability: float):
    r"""Returns the cutoffs of a Gaussian state by computing the 1-mode marginals until
    the probability of the marginal is less than ``probability``.

    Args:
        cov: the covariance matrix
        means: the means vector
        probability: the cutoff probability

    Returns:
        Tuple[int, ...]: the suggested cutoffs
    """
    M = len(means) // 2
    cutoffs = []
    for i in range(M):
        cov_i = np.array([[cov[i, i], cov[i, i + M]], [cov[i + M, i], cov[i + M, i + M]]])
        means_i = np.array([means[i], means[i + M]])
        # apply 1-d recursion until probability is less than 0.99
        A, B, C = [math.asnumpy(x) for x in wigner_to_bargmann_rho(cov_i, means_i)]
        diag = fock_representation_diagonal_amps(A, B, C, 1, cutoffs=[100])[0]
        # find at what index in the cumsum the probability is more than 0.99
        for i, val in enumerate(np.cumsum(diag)):
            if val > probability:
                cutoffs.append(max(i + 1, settings.AUTOCUTOFF_MIN_CUTOFF))
                break
        else:
            cutoffs.append(settings.AUTOCUTOFF_MAX_CUTOFF)
    return cutoffs


def wigner_to_fock_state(
    cov: Matrix,
    means: Vector,
    shape: Sequence[int],
    return_dm: bool = True,
) -> Tensor:
    r"""Returns the Fock representation of a Gaussian state.
    Use with caution: if the cov matrix is that of a mixed state,
    setting return_dm to False will produce nonsense.

    * If the state is pure it can return the state vector (ket) or the density matrix.
        The index order is going to be ket_i where i is the only multimode index.
    * If the state is mixed it can return the density matrix.
        The index order is going to be dm_ij where j is the right multimode index and i is the left one.

    Args:
        cov: the Wigner covariance matrix
        means: the Wigner means vector
        shape: the shape of the tensor
        return_dm: whether to return the density matrix (otherwise it returns the ket)

    Returns:
        Tensor: the fock representation
    """
    if return_dm:
        A, B, C = wigner_to_bargmann_rho(cov, means)
        return math.hermite_renormalized(-A, B, C, shape=shape)
    else:
        A, B, C = wigner_to_bargmann_psi(cov, means)
        return math.hermite_renormalized(-A, B, C, shape=shape)


def wigner_to_fock_U(X, d, shape):
    r"""Returns the Fock representation of a Gaussian unitary transformation.
    The index order is out_l, in_l, where in_l is to be contracted with the indices of a ket,
    or with the left indices of a density matrix.

    Arguments:
        X: the X matrix
        d: the d vector
        shape: the shape of the tensor

    Returns:
        Tensor: the fock representation of the unitary transformation
    """
    A, B, C = wigner_to_bargmann_U(X, d)
    return math.hermite_renormalized(-A, B, C, shape=shape)


def wigner_to_fock_Choi(X, Y, d, shape):
    r"""Returns the Fock representation of a Gaussian Choi matrix.
    The order of choi indices is :math:`[\mathrm{out}_l, \mathrm{in}_l, \mathrm{out}_r, \mathrm{in}_r]`
    where :math:`\mathrm{in}_l` and :math:`\mathrm{in}_r` are to be contracted with the left and right indices of a density matrix.

    Arguments:
        X: the X matrix
        Y: the Y matrix
        d: the d vector
        shape: the shape of the tensor

    Returns:
        Tensor: the fock representation of the Choi matrix
    """
    A, B, C = wigner_to_bargmann_Choi(X, Y, d)
    return math.hermite_renormalized(-A, B, C, shape=shape)


def ket_to_dm(ket: Tensor) -> Tensor:
    r"""Maps a ket to a density matrix.

    Args:
        ket: the ket

    Returns:
        Tensor: the density matrix
    """
    return math.outer(ket, math.conj(ket))


def dm_to_ket(dm: Tensor) -> Tensor:
    r"""Maps a density matrix to a ket if the state is pure.

    If the state is pure :math:`\hat \rho= |\psi\rangle\langle \psi|` then the
    ket is the eigenvector of :math:`\rho` corresponding to the eigenvalue 1.

    Args:
        dm (Tensor): the density matrix

    Returns:
        Tensor: the ket

    Raises:
        ValueError: if ket for mixed states cannot be calculated
    """

    is_pure_dm = np.isclose(purity(dm), 1.0, atol=1e-6)
    if not is_pure_dm:
        raise ValueError("Cannot calculate ket for mixed states.")

    cutoffs = dm.shape[: len(dm.shape) // 2]
    d = int(np.prod(cutoffs))
    dm = math.reshape(dm, (d, d))

    eigvals, eigvecs = math.eigh(dm)
    # eigenvalues and related eigenvectors are sorted in non-decreasing order,
    # meaning the associated eigvec to largest eigval is stored last.
    ket = eigvecs[:, -1] * math.sqrt(eigvals[-1])
    ket = math.reshape(ket, cutoffs)

    return ket


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
        Tensor: the Choi tensor. The index order is going to be :math:`[\mathrm{out}_l, \mathrm{in}_l, \mathrm{out}_r, \mathrm{in}_r]`
        where :math:`\mathrm{in}_l` and :math:`\mathrm{in}_r` are to be contracted with the left and right indices of the density matrix.
    """
    return math.outer(U, math.conj(U))


def fidelity(state_a, state_b, a_ket: bool, b_ket: bool) -> Scalar:
    r"""Computes the fidelity between two states in Fock representation."""
    if a_ket and b_ket:
        min_cutoffs = [slice(min(a, b)) for a, b in zip(state_a.shape, state_b.shape)]
        state_a = state_a[tuple(min_cutoffs)]
        state_b = state_b[tuple(min_cutoffs)]
        return math.abs(math.sum(math.conj(state_a) * state_b)) ** 2

    if a_ket:
        min_cutoffs = [
            slice(min(a, b))
            for a, b in zip(state_a.shape, state_b.shape[: len(state_b.shape) // 2])
        ]
        state_a = state_a[tuple(min_cutoffs)]
        state_b = state_b[tuple(min_cutoffs * 2)]
        a = math.reshape(state_a, -1)
        return math.real(
            math.sum(math.conj(a) * math.matvec(math.reshape(state_b, (len(a), len(a))), a))
        )

    if b_ket:
        min_cutoffs = [
            slice(min(a, b))
            for a, b in zip(state_a.shape[: len(state_a.shape) // 2], state_b.shape)
        ]
        state_a = state_a[tuple(min_cutoffs * 2)]
        state_b = state_b[tuple(min_cutoffs)]
        b = math.reshape(state_b, -1)
        return math.real(
            math.sum(math.conj(b) * math.matvec(math.reshape(state_a, (len(b), len(b))), b))
        )

    # mixed state
    # Richard Jozsa (1994) Fidelity for Mixed Quantum States, Journal of Modern Optics, 41:12, 2315-2323, DOI: 10.1080/09500349414552171

    # trim states to have same cutoff
    min_cutoffs = [
        slice(min(a, b))
        for a, b in zip(
            state_a.shape[: len(state_a.shape) // 2], state_b.shape[: len(state_b.shape) // 2]
        )
    ]
    state_a = state_a[tuple(min_cutoffs * 2)]
    state_b = state_b[tuple(min_cutoffs * 2)]
    return math.abs(
        (
            math.trace(
                math.sqrtm(
                    math.matmul(math.matmul(math.sqrtm(state_a), state_b), math.sqrtm(state_a))
                )
            )
            ** 2
        )
    )


def number_means(tensor, is_dm: bool):
    r"""Returns the mean of the number operator in each mode."""
    probs = math.all_diagonals(tensor, real=True) if is_dm else math.abs(tensor) ** 2
    modes = list(range(len(probs.shape)))
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
    modes = list(range(len(probs.shape)))
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
    dm = dm / math.trace(dm)  # assumes all nonzero values are included in the density matrix
    return math.abs(math.sum(math.transpose(dm) * dm))  # tr(rho^2)


def validate_contraction_indices(in_idx, out_idx, M, name):
    r"""Validates the indices used for the contraction of a tensor."""
    if len(set(in_idx)) != len(in_idx):
        raise ValueError(f"{name}_in_idx should not contain repeated indices.")
    if len(set(out_idx)) != len(out_idx):
        raise ValueError(f"{name}_out_idx should not contain repeated indices.")
    if not set(range(M)).intersection(out_idx).issubset(set(in_idx)):
        wrong_indices = set(range(M)).intersection(out_idx) - set(in_idx)
        raise ValueError(
            f"Indices {wrong_indices} in {name}_out_idx are trying to replace uncontracted indices."
        )


def apply_kraus_to_ket(kraus, ket, kraus_in_idx, kraus_out_idx=None):
    r"""Applies a kraus operator to a ket.
    It assumes that the ket is indexed as left_1, ..., left_n.

    The kraus op has indices that contract with the ket (kraus_in_idx) and indices that are left over (kraus_out_idx).
    The final index order will be sorted (note that an index appearing in both kraus_in_idx and kraus_out_idx will replace the original index).

    Args:
        kraus (array): the kraus operator to be applied
        ket (array): the ket to which the operator is applied
        kraus_in_idx (list of ints): the indices (counting from 0) of the kraus operator that contract with the ket
        kraus_out_idx (list of ints): the indices (counting from 0) of the kraus operator that are leftover

    Returns:
        array: the resulting ket with indices as kraus_out_idx + uncontracted ket indices
    """
    if kraus_out_idx is None:
        kraus_out_idx = kraus_in_idx

    if not set(kraus_in_idx).issubset(range(ket.ndim)):
        raise ValueError("kraus_in_idx should be a subset of the ket indices.")

    # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
    validate_contraction_indices(kraus_in_idx, kraus_out_idx, ket.ndim, "kraus")

    ket = MMTensor(ket, axis_labels=[f"left_{i}" for i in range(ket.ndim)])
    kraus = MMTensor(
        kraus,
        axis_labels=[f"out_left_{i}" for i in kraus_out_idx] + [f"left_{i}" for i in kraus_in_idx],
    )

    # contract the operator with the ket.
    # now the leftover indices are in the order kraus_out_idx + uncontracted ket indices
    kraus_ket = kraus @ ket

    # sort kraus_ket.axis_labels by the int at the end of each label.
    # Each label is guaranteed to have a unique int at the end.
    new_axis_labels = sorted(kraus_ket.axis_labels, key=lambda x: int(x.split("_")[-1]))

    return kraus_ket.transpose(new_axis_labels).tensor


def apply_kraus_to_dm(kraus, dm, kraus_in_idx, kraus_out_idx=None):
    r"""Applies a kraus operator to a density matrix.
    It assumes that the density matrix is indexed as left_1, ..., left_n, right_1, ..., right_n.

    The kraus operator has indices that contract with the density matrix (kraus_in_idx) and indices that are leftover (kraus_out_idx).
    `kraus` will contract from the left and from the right with the density matrix. For right contraction the kraus op is conjugated.

    Args:
        kraus (array): the operator to be applied
        dm (array): the density matrix to which the operator is applied
        kraus_in_idx (list of ints): the indices (counting from 0) of the kraus operator that contract with the density matrix
        kraus_out_idx (list of ints): the indices (counting from 0) of the kraus operator that are leftover (default None, in which case kraus_out_idx = kraus_in_idx)

    Returns:
        array: the resulting density matrix
    """
    if kraus_out_idx is None:
        kraus_out_idx = kraus_in_idx

    if not set(kraus_in_idx).issubset(range(dm.ndim // 2)):
        raise ValueError("kraus_in_idx should be a subset of the density matrix indices.")

    # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
    validate_contraction_indices(kraus_in_idx, kraus_out_idx, dm.ndim // 2, "kraus")

    dm = MMTensor(
        dm,
        axis_labels=[f"left_{i}" for i in range(dm.ndim // 2)]
        + [f"right_{i}" for i in range(dm.ndim // 2)],
    )
    kraus = MMTensor(
        kraus,
        axis_labels=[f"out_left_{i}" for i in kraus_out_idx] + [f"left_{i}" for i in kraus_in_idx],
    )
    kraus_conj = MMTensor(
        math.conj(kraus.tensor),
        axis_labels=[f"out_right_{i}" for i in kraus_out_idx]
        + [f"right_{i}" for i in kraus_in_idx],
    )

    # contract the kraus operator with the density matrix from the left and from the right.
    k_dm_k = kraus @ dm @ kraus_conj
    # now the leftover indices are in the order:
    # out_left_idx + uncontracted left indices + uncontracted right indices + out_right_idx

    # sort k_dm_k.axis_labels by the int at the end of each label, first left, then right
    N = k_dm_k.tensor.ndim // 2
    left = sorted(k_dm_k.axis_labels[:N], key=lambda x: int(x.split("_")[-1]))
    right = sorted(k_dm_k.axis_labels[N:], key=lambda x: int(x.split("_")[-1]))

    return k_dm_k.transpose(left + right).tensor


def apply_choi_to_dm(choi, dm, choi_in_idx, choi_out_idx=None):
    r"""Applies a choi operator to a density matrix.
    It assumes that the density matrix is indexed as left_1, ..., left_n, right_1, ..., right_n.

    The choi operator has indices that contract with the density matrix (choi_in_idx) and indices that are left over (choi_out_idx).
    `choi` will contract choi_in_idx from the left and from the right with the density matrix.

    Args:
        choi (array): the choi operator to be applied
        dm (array): the density matrix to which the choi operator is applied
        choi_in_idx (list of ints): the indices of the choi operator that contract with the density matrix
        choi_out_idx (list of ints): the indices of the choi operator that re leftover

    Returns:
        array: the resulting density matrix
    """
    if choi_out_idx is None:
        choi_out_idx = choi_in_idx

    if not set(choi_in_idx).issubset(range(dm.ndim // 2)):
        raise ValueError("choi_in_idx should be a subset of the density matrix indices.")

    # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
    validate_contraction_indices(choi_in_idx, choi_out_idx, dm.ndim // 2, "choi")

    dm = MMTensor(
        dm,
        axis_labels=[f"left_{i}" for i in range(dm.ndim // 2)]
        + [f"right_{i}" for i in range(dm.ndim // 2)],
    )
    choi = MMTensor(
        choi,
        axis_labels=[f"out_left_{i}" for i in choi_out_idx]
        + [f"left_{i}" for i in choi_in_idx]
        + [f"out_right_{i}" for i in choi_out_idx]
        + [f"right_{i}" for i in choi_in_idx],
    )

    # contract the choi matrix with the density matrix.
    # now the leftover indices are in the order out_left_idx + out_right_idx + uncontracted left indices + uncontracted right indices
    choi_dm = choi @ dm

    # sort choi_dm.axis_labels by the int at the end of each label, first left, then right
    left_labels = [label for label in choi_dm.axis_labels if "left" in label]
    left = sorted(left_labels, key=lambda x: int(x.split("_")[-1]))
    right_labels = [label for label in choi_dm.axis_labels if "right" in label]
    right = sorted(right_labels, key=lambda x: int(x.split("_")[-1]))

    return choi_dm.transpose(left + right).tensor


def apply_choi_to_ket(choi, ket, choi_in_idx, choi_out_idx=None):
    r"""Applies a choi operator to a ket.
    It assumes that the ket is indexed as left_1, ..., left_n.

    The choi operator has indices that contract with the ket (choi_in_idx) and indices that are left over (choi_out_idx).
    `choi` will contract choi_in_idx from the left and from the right with the ket.

    Args:
        choi (array): the choi operator to be applied
        ket (array): the ket to which the choi operator is applied
        choi_in_idx (list of ints): the indices of the choi operator that contract with the ket
        choi_out_idx (list of ints): the indices of the choi operator that re leftover

    Returns:
        array: the resulting ket
    """
    if choi_out_idx is None:
        choi_out_idx = choi_in_idx

    if not set(choi_in_idx).issubset(range(ket.ndim)):
        raise ValueError("choi_in_idx should be a subset of the ket indices.")

    # check that there are no repeated indices in kraus_in_idx and kraus_out_idx (separately)
    validate_contraction_indices(choi_in_idx, choi_out_idx, ket.ndim, "choi")

    ket = MMTensor(ket, axis_labels=[f"left_{i}" for i in range(ket.ndim)])
    ket_dual = MMTensor(math.conj(ket.tensor), axis_labels=[f"right_{i}" for i in range(ket.ndim)])
    choi = MMTensor(
        choi,
        axis_labels=[f"out_left_{i}" for i in choi_out_idx]
        + [f"left_{i}" for i in choi_in_idx]
        + [f"out_right_{i}" for i in choi_out_idx]
        + [f"right_{i}" for i in choi_in_idx],
    )

    # contract the choi matrix with the ket and its dual, like choi @ |ket><ket|
    # now the leftover indices are in the order out_left_idx + out_right_idx + uncontracted left indices + uncontracted right indices
    choi_ket = choi @ ket @ ket_dual

    # sort choi_ket.axis_labels by the int at the end of each label, first left, then right
    left_labels = [label for label in choi_ket.axis_labels if "left" in label]
    left = sorted(left_labels, key=lambda x: int(x.split("_")[-1]))
    right_labels = [label for label in choi_ket.axis_labels if "right" in label]
    right = sorted(right_labels, key=lambda x: int(x.split("_")[-1]))

    return choi_ket.transpose(left + right).tensor


def contract_states(
    stateA, stateB, a_is_dm: bool, b_is_dm: bool, modes: List[int], normalize: bool
):
    r"""Contracts two states in the specified modes.
    Assumes that the modes of B are a subset of the modes of A.

    Args:
        stateA: the first state
        stateB: the second state
        a_is_dm: whether the first state is a density matrix.
        b_is_dm: whether the second state is a density matrix.
        modes: the modes on which to contract the states.
        normalize: whether to normalize the result

    Returns:
        Tensor: the contracted state tensor (subsystem of ``A``). Either ket or dm.
    """

    if a_is_dm:
        if b_is_dm:  # a DM, b DM
            dm = apply_choi_to_dm(choi=stateB, dm=stateA, choi_in_idx=modes, choi_out_idx=[])
        else:  # a DM, b ket
            dm = apply_kraus_to_dm(
                kraus=math.conj(stateB), dm=stateA, kraus_in_idx=modes, kraus_out_idx=[]
            )
    else:
        if b_is_dm:  # a ket, b DM
            dm = apply_kraus_to_dm(
                kraus=math.conj(stateA), dm=stateB, kraus_in_idx=modes, kraus_out_idx=[]
            )
        else:  # a ket, b ket
            ket = apply_kraus_to_ket(
                kraus=math.conj(stateB), ket=stateA, kraus_in_idx=modes, kraus_out_idx=[]
            )

    try:
        return dm / math.sum(math.all_diagonals(dm, real=False)) if normalize else dm
    except NameError:
        return ket / math.norm(ket) if normalize else ket


def normalize(fock: Tensor, is_dm: bool):
    r"""Returns the normalized ket state.

    Args:
        fock (Tensor): the state to be normalized
        is_dm (optioanl bool): whether the input tensor is a density matrix

    Returns:
        Tensor: the normalized state
    """
    if is_dm:
        return fock / math.sum(math.all_diagonals(fock, real=False))

    return fock / math.sum(math.norm(fock))


def norm(state: Tensor, is_dm: bool):
    r"""
    Returns the norm of a ket or the trace of the density matrix.
    Note that the "norm" is intended as the float number that is used to normalize the state,
    and depends on the representation. Hence different numbers for different representations
    of the same state (:math:`|amp|` for ``ket`` and :math:`|amp|^2` for ``dm``).
    """
    if is_dm:
        return math.sum(math.all_diagonals(state, real=True))

    return math.abs(math.norm(state))


def is_mixed_dm(dm):
    r"""Evaluates if a density matrix represents a mixed state."""
    cutoffs = dm.shape[: len(dm.shape) // 2]
    square = math.reshape(dm, (int(np.prod(cutoffs)), -1))
    return not np.isclose(math.sum(square * math.transpose(square)), 1.0)


def trace(dm, keep: List[int]):
    r"""Computes the partial trace of a density matrix.
    The indices of the density matrix are in the order (out0, ..., outN-1, in0, ..., inN-1).
    The indices to keep are a subset of the N 'out' indices
    (they count for the 'in' indices as well).

    Args:
        dm: the density matrix
        keep: the modes to keep (0-based)
    """
    dm = MMTensor(
        dm,
        axis_labels=[
            f"out_{i}" if i in keep else f"contract_{i}" for i in range(len(dm.shape) // 2)
        ]
        + [f"in_{i}" if i in keep else f"contract_{i}" for i in range(len(dm.shape) // 2)],
    )
    return dm.contract().tensor


@tensor_int_cache
def oscillator_eigenstate(q: Vector, cutoff: int) -> Tensor:
    r"""Harmonic oscillator eigenstate wavefunction `\psi_n(q) = <n|q>`.

    Args:
        q (Vector): a vector containing the q points at which the function is evaluated (units of \sqrt{\hbar})
        cutoff (int): maximum number of photons
        hbar (optional): value of `\hbar`, defaults to Mr Mustard's internal value

    Returns:
        Tensor: a tensor of size ``len(q)*cutoff``. Each entry with index ``[i, j]`` represents the eigenstate evaluated
            with number of photons ``i`` evaluated at position ``q[j]``, i.e., `\psi_i(q_j)`.

    .. details::

        .. admonition:: Definition
            :class: defn

        The q-quadrature eigenstates are defined as

        .. math::

            \psi_n(x) = 1/sqrt[2^n n!](\frac{\omega}{\pi \hbar})^{1/4}
                \exp{-\frac{\omega}{2\hbar} x^2} H_n(\sqrt{\frac{\omega}{\pi}} x)

        where :math:`H_n(x)` is the (physicists) `n`-th Hermite polynomial.
    """
    omega_over_hbar = math.cast(1 / settings.HBAR, "float64")
    x_tensor = math.sqrt(omega_over_hbar) * math.cast(q, "float64")  # unit-less vector

    # prefactor term (\Omega/\hbar \pi)**(1/4) * 1 / sqrt(2**n)
    prefactor = (omega_over_hbar / np.pi) ** (1 / 4) * math.sqrt(2 ** (-math.arange(0, cutoff)))

    # Renormalized physicist hermite polys: Hn / sqrt(n!)
    R = np.array([[2 + 0j]])  # to get the physicist polys

    def f_hermite_polys(xi):
        poly = math.hermite_renormalized(R, 2 * math.astensor([xi], "complex128"), 1 + 0j, cutoff)
        return math.cast(poly, "float64")

    hermite_polys = math.map_fn(f_hermite_polys, x_tensor)

    # (real) wavefunction
    psi = math.exp(-(x_tensor**2 / 2)) * math.transpose(prefactor * hermite_polys)
    return psi


@lru_cache
def estimate_dx(cutoff, period_resolution=20):
    r"""Estimates a suitable quadrature discretization interval `dx`. Uses the fact
    that Fock state `n` oscillates with angular frequency :math:`\sqrt{2(n + 1)}`,
    which follows from the relation

    .. math::

            \psi^{[n]}'(q) = q - sqrt(2*(n + 1))*\psi^{[n+1]}(q)

    by setting q = 0, and approximating the oscillation amplitude by `\psi^{[n+1]}(0)

    Ref: https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions

    Args
        cutoff (int): Fock cutoff
        period_resolution (int): Number of points used to sample one Fock
            wavefunction oscillation. Larger values yields better approximations
            and thus smaller `dx`.

    Returns
        (float): discretization value of quadrature
    """
    fock_cutoff_frequency = np.sqrt(2 * (cutoff + 1))
    fock_cutoff_period = 2 * np.pi / fock_cutoff_frequency
    dx_estimate = fock_cutoff_period / period_resolution
    return dx_estimate


@lru_cache
def estimate_xmax(cutoff, minimum=5):
    r"""Estimates a suitable quadrature axis length

    Args
        cutoff (int): Fock cutoff
        minimum (float): Minimum value of the returned xmax

    Returns
        (float): maximum quadrature value
    """
    if cutoff == 0:
        xmax_estimate = 3
    else:
        # maximum q for a classical particle with energy n=cutoff
        classical_endpoint = np.sqrt(2 * cutoff)
        # approximate probability of finding particle outside classical region
        excess_probability = 1 / (7.464 * cutoff ** (1 / 3))
        # Emperical factor that yields reasonable results
        A = 5
        xmax_estimate = classical_endpoint * (1 + A * excess_probability)
    return max(minimum, xmax_estimate)


@lru_cache
def estimate_quadrature_axis(cutoff, minimum=5, period_resolution=20):
    """Generates a suitable quadrature axis.

    Args
        cutoff (int): Fock cutoff
        minimum (float): Minimum value of the returned xmax
        period_resolution (int): Number of points used to sample one Fock
            wavefunction oscillation. Larger values yields better approximations
            and thus smaller dx.

    Returns
        (array): quadrature axis
    """
    xmax = estimate_xmax(cutoff, minimum=minimum)
    dx = estimate_dx(cutoff, period_resolution=period_resolution)
    xaxis = np.arange(-xmax, xmax, dx)
    xaxis = np.append(xaxis, xaxis[-1] + dx)
    xaxis = xaxis - np.mean(xaxis)  # center around 0
    return xaxis


def quadrature_distribution(
    state: Tensor, quadrature_angle: float = 0.0, x: Vector = None, hbar: Optional[float] = None
):
    r"""Given the ket or density matrix of a single-mode state, it generates the probability
    density distribution :math:`\tr [ \rho |x_\phi><x_\phi| ]`  where `\rho` is the
    density matrix of the state and |x_\phi> the quadrature eigenvector with angle `\phi`
    equal to ``quadrature_angle``.

    Args:
        state (Tensor): single mode state ket or density matrix
        quadrature_angle (float): angle of the quadrature basis vector
        x (Vector): points at which the quadrature distribution is evaluated

    Returns:
        tuple(Vector, Vector): coordinates at which the pdf is evaluated and the probability distribution
    """
    dims = len(state.shape)
    if dims > 2:
        raise ValueError(
            "Input state has dimension {state.shape}. Make sure is either a single-mode ket or dm."
        )

    is_dm = dims == 2
    cutoff = state.shape[0]

    if not np.isclose(quadrature_angle, 0.0):
        # rotate mode to the homodyne basis
        theta = -math.arange(cutoff) * quadrature_angle
        Ur = math.diag(math.make_complex(math.cos(theta), math.sin(theta)))
        state = (
            math.einsum("ij,jk,kl->il", Ur, state, math.dagger(Ur))
            if is_dm
            else math.matvec(Ur, state)
        )

    if x is None:
        hbar = hbar or settings.HBAR
        x = np.sqrt(hbar) * math.new_constant(estimate_quadrature_axis(cutoff), "q_tensor")

    psi_x = math.cast(oscillator_eigenstate(x, cutoff), "complex128")
    pdf = (
        math.einsum("nm,nj,mj->j", state, psi_x, psi_x)
        if is_dm
        else math.abs(math.einsum("n,nj->j", state, psi_x)) ** 2
    )

    return x, math.cast(pdf, "float64")


def sample_homodyne(
    state: Tensor, quadrature_angle: float = 0.0, hbar: Optional[float] = None
) -> Tuple[float, float]:
    r"""Given a single-mode state, it generates the pdf of :math:`\tr [ \rho |x_\phi><x_\phi| ]`
    where `\rho` is the reduced density matrix of the state.

    Args:
        state (Tensor): ket or density matrix of the state being measured
        quadrature_angle (float): angle of the quadrature distribution

    Returns:
        tuple(float, float): outcome and probability of the outcome
    """
    dims = len(state.shape)
    if dims > 2:
        raise ValueError(
            "Input state has dimension {state.shape}. Make sure is either a single-mode ket or dm."
        )

    x, pdf = quadrature_distribution(state, quadrature_angle, hbar=hbar or settings.HBAR)
    probs = pdf * (x[1] - x[0])

    # draw a sample from the distribution
    pdf = math.Categorical(probs=probs, name="homodyne_dist")
    sample_idx = pdf.sample()
    homodyne_sample = math.gather(x, sample_idx)
    probability_sample = math.gather(probs, sample_idx)

    return homodyne_sample, probability_sample


@jit(nopython=True)
def _displacement(r, phi, cutoff, dtype=np.complex128):  # pragma: no cover
    r"""Calculates the matrix elements of the displacement gate using a recurrence relation.
    Uses the log of the matrix elements to avoid numerical issues and then takes the exponential.

    Args:
        r (float): displacement magnitude
        phi (float): displacement angle
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation

    Returns:
        array[complex]: matrix representing the displacement operation.
    """
    D = np.zeros((cutoff, cutoff), dtype=dtype)
    rng = np.arange(cutoff)
    rng[0] = 1
    log_k_fac = np.cumsum(np.log(rng))
    for n_minus_m in range(cutoff):
        m_max = cutoff - n_minus_m
        logL = np.log(_laguerre(r**2.0, m_max, n_minus_m))
        for m in range(m_max):
            n = n_minus_m + m
            D[n, m] = np.exp(
                0.5 * (log_k_fac[m] - log_k_fac[n])
                + n_minus_m * np.log(r)
                - (r**2.0) / 2.0
                + 1j * phi * n_minus_m
                + logL[m]
            )
            D[m, n] = (-1.0) ** (n_minus_m) * np.conj(D[n, m])
    return D


@jit(nopython=True, cache=True)
def _laguerre(x, N, alpha, dtype=np.complex128):  # pragma: no cover
    r"""Returns the N first generalized Laguerre polynomials evaluated at x.

    Args:
        x (float): point at which to evaluate the polynomials
        N (int): maximum Laguerre polynomial to calculate
        alpha (float): continuous parameter for the generalized Laguerre polynomials
    """
    L = np.zeros(N, dtype=dtype)
    L[0] = 1.0
    if N > 1:
        for m in range(0, N - 1):
            L[m + 1] = ((2 * m + 1 + alpha - x) * L[m] - (m + alpha) * L[m - 1]) / (m + 1)
    return L


@jit(nopython=True)
def _grad_displacement(T, r, phi):  # pragma: no cover
    r"""Calculates the gradients of the displacement gate with respect to the displacement magnitude and angle.

    Args:
        T (array[complex]): array representing the gate
        r (float): displacement magnitude
        phi (float): displacement angle

    Returns:
        tuple[array[complex], array[complex]]: The gradient of the displacement gate with respect to r and phi
    """
    cutoff = T.shape[0]
    dtype = T.dtype
    ei = np.exp(1j * phi)
    eic = np.exp(-1j * phi)
    alpha = r * ei
    alphac = r * eic
    sqrt = np.sqrt(np.arange(cutoff, dtype=dtype))
    grad_r = np.zeros((cutoff, cutoff), dtype=dtype)
    grad_phi = np.zeros((cutoff, cutoff), dtype=dtype)

    for m in range(cutoff):
        for n in range(cutoff):
            grad_r[m, n] = -r * T[m, n] + sqrt[m] * ei * T[m - 1, n] - sqrt[n] * eic * T[m, n - 1]
            grad_phi[m, n] = (
                sqrt[m] * 1j * alpha * T[m - 1, n] + sqrt[n] * 1j * alphac * T[m, n - 1]
            )

    return grad_r, grad_phi


@math.custom_gradient
def displacement(r, phi, cutoff, tol=1e-15):
    """creates a single mode displacement matrix"""
    if r > tol:
        gate = _displacement(math.asnumpy(r), math.asnumpy(phi), cutoff)
    else:
        gate = math.eye(cutoff, dtype="complex128")

    def grad(dy):  # pragma: no cover
        Dr, Dphi = _grad_displacement(math.asnumpy(gate), math.asnumpy(r), math.asnumpy(phi))
        grad_r = math.real(math.sum(dy * math.conj(Dr)))
        grad_phi = math.real(math.sum(dy * math.conj(Dphi)))
        return grad_r, grad_phi, None

    return gate, grad

# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional

import numpy as np
from numba import njit, typed, types

from mrmustard.math.lattice import paths, steps
from mrmustard.typing import ComplexMatrix, ComplexTensor, ComplexVector, IntVector


@njit
def vanilla(shape: tuple[int, ...], A, b, c) -> ComplexTensor:
    r"""Vanilla Fock-Bargmann strategy. Fills the tensor by iterating over all indices
    in ndindex order.

    Args:
        shape (tuple[int, ...]): shape of the output tensor
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """

    # init output tensor
    Z = np.zeros(shape, dtype=np.complex128)

    # initialize path iterator
    path = np.ndindex(shape)

    # write vacuum amplitude
    Z[next(path)] = c

    # iterate over the rest of the indices
    for index in path:
        Z[index] = steps.vanilla_step(Z, A, b, index)
    return Z


@njit
def vanilla_jacobian(G, A, b, c) -> tuple[ComplexTensor, ComplexTensor, ComplexTensor]:
    r"""Vanilla Fock-Bargmann strategy gradient. Returns dG/dA, dG/db, dG/dc.
    Notice that G is a holomorphic function of A, b, c. This means that there is only
    one gradient to care about for each parameter (i.e. not dG/dA.conj() etc).
    """

    # init output tensors
    dGdA = np.zeros(G.shape + A.shape, dtype=np.complex128)
    dGdb = np.zeros(G.shape + b.shape, dtype=np.complex128)
    dGdc = G / c

    # initialize path iterator
    path = paths.ndindex_path(G.shape)

    # skip first index
    next(path)

    # iterate over the rest of the indices
    for index in path:
        dGdA, dGdb = steps.vanilla_step_jacobian(G, A, b, index, dGdA, dGdb)

    return dGdA, dGdb, dGdc


@njit
def vanilla_grad(G, D, c, dLdG) -> tuple[ComplexMatrix, ComplexVector, complex]:
    r"""Vanilla Fock-Bargmann strategy gradient. Returns dL/dA, dL/db, dL/dc.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        D (int): Dimension of the A/b tensors
        c (complex): vacuum amplitude
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor

    Returns:
        tuple[np.ndarray, np.ndarray, complex]: dL/dA, dL/db, dL/dc
    """
    # init gradients
    dA = np.zeros((D, D), dtype=np.complex128)  # component of dL/dA
    db = np.zeros(D, dtype=np.complex128)  # component of dL/db
    dLdA = np.zeros_like(dA)
    dLdb = np.zeros_like(db)

    # initialize path iterator
    path = np.ndindex(G.shape)

    # skip first index
    next(path)

    # iterate over the rest of the indices
    for index in path:
        dA, db = steps.vanilla_step_grad(G, index, dA, db)
        dLdA += dA * dLdG[index]
        dLdb += db * dLdG[index]

    dLdc = np.sum(G * dLdG) / c

    return dLdA, dLdb, dLdc


def binomial(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    max_prob: Optional[float] = None,
    global_cutoff: Optional[int] = None,
) -> ComplexTensor:
    r"""Binomial strategy (fill ket by weight), python version with numba function/loop.

    Args:
        local_cutoffs (tuple[int, ...]): local cutoffs of the tensor (used at least as shape)
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude
        max_prob (float): max L2 norm. If reached, the computation is stopped early.
        global_cutoff (Optional[int]): global cutoff (max total photon number considered).
            If not given it is calculated from the local cutoffs.

    Returns:
        G, prob (np.ndarray, float): Fock representation of the Gaussian tensor with shape ``shape`` and L2 norm
    """
    # sort out global cutoff
    if global_cutoff is None:
        global_cutoff = sum(local_cutoffs) - len(local_cutoffs)

    # init output tensor
    G = np.zeros(local_cutoffs, dtype=np.complex128)

    # write vacuum amplitude
    G.flat[0] = c
    prob = np.abs(c) ** 2

    # iterate over subspaces by weight and stop if norm is large enough. Caches indices.
    for photons in range(1, global_cutoff):
        try:
            indices = paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)]
        except KeyError:
            indices = paths.binomial_subspace(local_cutoffs, photons)
            paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)] = indices
        G, prob_subspace = steps.binomial_step(G, A, b, indices)  # numba parallelized function
        prob += prob_subspace
        try:
            if prob > max_prob:
                break
        except TypeError:  # max_prob is None
            pass
    return G, prob


def binomial_dict(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    max_prob: Optional[float] = None,
    global_cutoff: Optional[int] = None,
) -> dict[tuple[int, ...], complex]:
    r"""Factorial speedup strategy (fill ket by weight), python version with numba function/loop.
    Uses a dictionary to store the output.

    Args:
        local_cutoffs (tuple[int, ...]): local cutoffs of the tensor (used at least as shape)
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude
        max_prob (float): max L2 norm. If reached, the computation is stopped early.
        global_cutoff (Optional[int]): global cutoff (max total photon number considered).
            If not given it is calculated from the local cutoffs.

    Returns:
        dict[tuple[int, ...], complex]: Fock representation of the Gaussian tensor.
    """
    if global_cutoff is None:
        global_cutoff = sum(local_cutoffs) - len(local_cutoffs)

    # init numba output dict
    Z = typed.Dict.empty(
        key_type=types.UniTuple(types.int64, len(local_cutoffs)),
        value_type=types.complex128,
    )

    # write vacuum amplitude
    Z[(0,) * len(local_cutoffs)] = c
    prob = np.abs(c) ** 2

    # iterate over subspaces by weight and stop if norm is large enough. Caches indices.
    for photons in range(1, global_cutoff):
        try:
            indices = paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)]
        except KeyError:
            indices = paths.binomial_subspace(local_cutoffs, photons)
            paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)] = indices
        Z, prob_subspace = steps.binomial_step_dict(Z, A, b, indices)  # numba parallelized function
        prob += prob_subspace
        try:
            if prob > max_prob:
                break
        except TypeError:
            pass
    return Z


@njit
def binomial_numba(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    FP: dict[tuple[tuple[int, ...], int], list[tuple[int, ...]]],
    max_prob: float = 0.999,
    global_cutoff: Optional[int] = None,
) -> ComplexTensor:
    r"""Binomial strategy (fill by weight), fully numba version."""
    if global_cutoff is None:
        global_cutoff = sum(local_cutoffs) - len(local_cutoffs)

    # init output tensor
    Z = np.zeros(local_cutoffs, dtype=np.complex128)

    # write vacuum amplitude
    Z.flat[0] = c
    prob = np.abs(c) ** 2

    # iterate over all other indices in parallel and stop if norm is large enough
    for photons in range(1, global_cutoff):
        try:
            indices = FP[(local_cutoffs, photons)]
        except Exception:  # pylint: disable=broad-except
            indices = paths.binomial_subspace(local_cutoffs, photons)
            FP[(local_cutoffs, photons)] = indices
        Z, prob_subspace = steps.binomial_step(Z, A, b, indices)
        prob += prob_subspace
        if prob > max_prob:
            break
    return Z


@njit
def wormhole(shape: IntVector) -> IntVector:
    r"wormhole strategy, not implemented yet"
    raise NotImplementedError("Wormhole strategy not implemented yet")


@njit
def diagonal(shape: IntVector) -> IntVector:
    r"diagonal strategy, not implemented yet"
    raise NotImplementedError("Diagonal strategy not implemented yet")


@njit
def dynamic_U(shape: IntVector) -> IntVector:
    r"dynamic U strategy, not implemented yet"
    raise NotImplementedError("Diagonal strategy not implemented yet")

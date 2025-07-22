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

"This module contains binomial strategies"

from __future__ import annotations

import numpy as np
from numba import njit, typed, types

from mrmustard.math.lattice import paths, steps
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector

SQRT = np.sqrt(np.arange(100000))

__all__ = ["binomial", "binomial_dict", "binomial_numba"]


def binomial(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    max_l2: float,
    global_cutoff: int,
) -> ComplexTensor:
    r"""Binomial strategy (fill ket by weight), python version with numba function/loop.

    Args:
        local_cutoffs (tuple[int, ...]): local cutoffs of the tensor (used at least as shape)
        A (np.ndarray): A matrix of the Fock-Bargmann representation
        b (np.ndarray): B vector of the Fock-Bargmann representation
        c (complex): vacuum amplitude
        max_l2 (float): max L2 norm. If reached, the computation is stopped early.
        global_cutoff (Optional[int]): global cutoff (max total photon number considered + 1).

    Returns:
        G, prob (np.ndarray, float): Fock representation of the Gaussian tensor with shape ``shape`` and L2 norm
    """
    # init output tensor
    G = np.zeros(local_cutoffs, dtype=np.complex128)

    # write vacuum amplitude
    G.flat[0] = c
    norm = np.abs(c) ** 2

    # iterate over subspaces by weight and stop if norm is large enough. Caches indices.
    for photons in range(1, global_cutoff):
        try:
            indices = paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)]
        except KeyError:
            indices = paths.binomial_subspace_basis(local_cutoffs, photons)
            paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)] = indices
        G, subspace_norm = steps.binomial_step(G, A, b, indices)  # numba parallelized function
        norm += subspace_norm
        try:
            if norm > max_l2:
                break
        except TypeError:  # max_l2 is None
            pass
    return G, norm


def binomial_dict(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    max_prob: float | None = None,
    global_cutoff: int | None = None,
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
    G = typed.Dict.empty(
        key_type=types.UniTuple(types.int64, len(local_cutoffs)),
        value_type=types.complex128,
    )

    # write vacuum amplitude
    G[(0,) * len(local_cutoffs)] = c
    prob = np.abs(c) ** 2

    # iterate over subspaces by weight and stop if norm is large enough. Caches indices.
    for photons in range(1, global_cutoff):
        try:
            indices = paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)]
        except KeyError:
            indices = paths.binomial_subspace_basis(local_cutoffs, photons)
            paths.BINOMIAL_PATHS_PYTHON[(local_cutoffs, photons)] = indices
        G, prob_subspace = steps.binomial_step_dict(G, A, b, indices)  # numba parallelized function
        prob += prob_subspace
        try:
            if prob > max_prob:
                break
        except TypeError:
            pass
    return G


@njit(cache=True)
def binomial_numba(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    FP: dict[tuple[tuple[int, ...], int], list[tuple[int, ...]]],
    max_prob: float = 0.999,
    global_cutoff: int | None = None,
) -> ComplexTensor:  # pragma: no cover
    r"""Binomial strategy (fill by weight), fully numba version."""
    if global_cutoff is None:
        global_cutoff = sum(local_cutoffs) - len(local_cutoffs)

    # init output tensor
    G = np.zeros(local_cutoffs, dtype=np.complex128)

    # write vacuum amplitude
    G.flat[0] = c
    prob = np.abs(c) ** 2

    # iterate over all other indices in parallel and stop if norm is large enough
    for photons in range(1, global_cutoff):
        try:
            indices = FP[(local_cutoffs, photons)]
        except KeyError:
            indices = paths.binomial_subspace_basis(local_cutoffs, photons)
            FP[(local_cutoffs, photons)] = indices
        G, prob_subspace = steps.binomial_step(G, A, b, indices)
        prob += prob_subspace
        if prob > max_prob:
            break
    return G

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

import numpy as np
import numba
from numba import njit, prange, typed, types
from typing import Optional, Union

from mrmustard.math.lattice import paths, steps, utils
from mrmustard.math.lattice.paths import FACTORIAL_PATHS_DICT
from mrmustard.typing import ComplexTensor, ComplexMatrix, ComplexVector


@njit
def vanilla(shape: tuple[int, ...], A, b, c) -> ComplexTensor:
    r"""Vanilla Fock-Bargmann strategy."""

    # init output tensor
    Z = np.zeros(shape, dtype=np.complex128)

    # initialize path iterator
    path = paths.ndindex_path(shape)

    # write vacuum amplitude
    Z[next(path)] = c

    # iterate over all other indices
    for index in path:
        Z[index] = steps.vanilla_step(Z, A, b, index)
    return Z


FACTORIAL_PATHS_PYTHON = {}


def factorial_python(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    max_prob: float = 0.999,
    global_cutoff: Optional[int] = None,
) -> ComplexTensor:
    r"""Factorial speedup strategy."""
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
            indices = FACTORIAL_PATHS_PYTHON[(local_cutoffs, photons)]
        except KeyError:
            indices = paths.binomial_subspace(local_cutoffs, photons)
            FACTORIAL_PATHS_PYTHON[(local_cutoffs, photons)] = indices
        Z, prob_subspace = factorial_step(Z, A, b, indices)  # numba parallelized function
        prob += prob_subspace
        if prob > max_prob:
            break
    return Z


def factorial_dict(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    max_prob: Optional[float] = None,
    global_cutoff: Optional[int] = None,
) -> dict[tuple[int, ...], complex]:
    r"""Factorial speedup strategy."""
    if global_cutoff is None:
        global_cutoff = sum(local_cutoffs) - len(local_cutoffs)

    # init output dict
    Z = typed.Dict.empty(
        key_type=types.UniTuple(types.int64, len(local_cutoffs)),
        value_type=types.complex128,
    )

    # write vacuum amplitude
    Z[(0,) * len(local_cutoffs)] = c
    prob = np.abs(c) ** 2

    # iterate over all other indices in parallel and stop if norm is large enough
    for photons in range(1, global_cutoff):
        try:
            indices = FACTORIAL_PATHS_PYTHON[(local_cutoffs, photons)]
        except KeyError:
            indices = paths.binomial_subspace(local_cutoffs, photons)
            FACTORIAL_PATHS_PYTHON[(local_cutoffs, photons)] = indices
        Z, prob_subspace = factorial_step_dict(Z, A, b, indices)  # numba parallelized function
        prob += prob_subspace
        try:
            if prob > max_prob:
                break
        except TypeError:
            pass
    return Z


@njit
def factorial_step(
    Z: ComplexTensor,
    A: ComplexMatrix,
    b: ComplexVector,
    subspace_indices: list[tuple[int, ...]],
) -> tuple[ComplexTensor, float]:
    prob = 0.0
    for i in range(len(subspace_indices)):
        value = steps.vanilla_step(Z, A, b, subspace_indices[i])
        Z[subspace_indices[i]] = value
        prob = prob + np.abs(value) ** 2
    return Z, prob


@njit
def factorial_step_dict(
    Z: types.DictType,
    A: ComplexMatrix,
    b: ComplexVector,
    subspace_indices: list[tuple[int, ...]],
) -> tuple[ComplexTensor, float]:
    prob = 0.0
    for i in range(len(subspace_indices)):
        value = steps.vanilla_step_dict(Z, A, b, subspace_indices[i])
        Z[subspace_indices[i]] = value
        prob = prob + np.abs(value) ** 2
    return Z, prob


@njit
def factorial_numba(
    local_cutoffs: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    FP: dict[tuple[tuple[int, ...], int], list[tuple[int, ...]]],
    max_prob: float = 0.999,
    global_cutoff: Optional[int] = None,
) -> ComplexTensor:
    r"""Factorial speedup strategy."""
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
        Z, prob_subspace = factorial_step(Z, A, b, indices)
        prob += prob_subspace
        if prob > max_prob:
            break
    return Z

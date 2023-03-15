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
from numba import njit, prange, typeof
from typing import Optional

from mrmustard.math.lattice import paths, steps, utils
from mrmustard.typing import ComplexTensor, ComplexMatrix, ComplexVector

from numba.typed import Dict, List


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


# COMBINATIONS_DICT: dict[int, dict[tuple[tuple[int,...],int],list[tuple[int,...]]]] = Dict.empty(


@njit(parallel=True)
def factorial(
    shape: tuple[int, ...],
    A: ComplexMatrix,
    b: ComplexVector,
    c: complex,
    max_prob: float = 0.999,
    max_photons: Optional[int] = None,
) -> ComplexTensor:
    r"""Factorial speedup strategy."""

    # init output tensor
    Z = np.zeros(shape, dtype=np.complex128)

    # initialize path iterator
    path = paths.equal_weight_path(shape, max_photons)

    # write vacuum amplitude
    Z[next(path)[0]] = c

    # iterate over all other indices in parallel and stop if norm is large enough
    for indices in path:
        for i in prange(len(indices)):
            Z[indices[i]] = steps.vanilla_step(Z, A, b, indices[i])
        if np.linalg.norm(Z.reshape(-1)) ** 2 > max_prob:
            break

    return Z

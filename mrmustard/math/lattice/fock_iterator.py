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

from typing import Optional, Tuple

import numpy as np
from numba import njit

from mrmustard.math.lattice import paths, steps, utils
from mrmustard.typing import Tensor


@njit
def vanilla(shape: Tuple[int], A, b, c) -> Tensor:
    Z = np.zeros(shape, dtype=np.complex128)
    Z.flat[0] = c
    path = paths.ndindex_iter(np.asarray(shape))
    next(path)  # skip the zero index
    pivot_idx = np.zeros(len(shape), dtype=np.int32)
    neighbors_idx = np.zeros((len(shape), len(shape)), dtype=np.int32)
    for index in path:
        val_at_index = steps.vanilla_step(Z, A, b, index, pivot_idx, neighbors_idx)
        # Z[tuple(index)] = val_at_index
        utils.tensor_set(Z, index, val_at_index)
    return Z


def adaptive_U(input_shape: Tuple[int], output_shape: Optional[Tuple[int]], A, b, c) -> Tensor:
    r"""Computes the Fock amplitudes of a unitary transformation. If the output shape is not
    specified, the output shape is determined by the cutoff that is necessary to achieve
    prob = settings.AUTOCUTOFF_PROBABILITY."""

    if output_shape is None:
        output_shape = input_shape
    path = paths.ndindex_iter(np.asarray(input_shape))
    next(path)  # skip the zero index
    for index in path:
        val_at_index = steps.adaptive_U_step(Z, A, b, c, index)
        Z[tuple(index)] = val_at_index

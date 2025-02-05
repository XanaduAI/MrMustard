# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the fast diagonal strategy for computing the conditional density matrices.
"""

from itertools import product
import numpy as np
from functools import lru_cache
from mrmustard.math.lattice.strategies.vanilla import vanilla_stable

__all__ = ["fast_diagonal"]

SQRT = np.sqrt(np.arange(10000))


def fast_diagonal(A, b, c, output_cutoff: int, pnr_cutoffs: tuple[int, ...]):
    r"""
    Computes an array of conditional density matrices using the fast diagonal strategy.
    """
    output_shape = (output_cutoff, output_cutoff)
    L = len(pnr_cutoffs) + 1  # total number of modes
    perm = [i for m in range(L) for i in (m, m + L)]
    A = A[perm, :][:, perm]
    b = b[perm]
    output = np.zeros(pnr_cutoffs + output_shape, dtype=np.complex128)
    output[(0,) * (L - 1)] = vanilla_stable(output_shape, A[:2, :2], b[:2], c)
    buffer_1 = {}
    buffer_0 = {(0, 0) * (L - 1): output[(0,) * (L - 1)]}
    for weight in range(1, 2 * output_cutoff + 2 * np.sum(pnr_cutoffs) - L):
        buffer_2 = buffer_1
        buffer_1 = buffer_0
        buffer_0 = {}
        for w in enumerate_diagonal_coords(weight, pnr_cutoffs):
            i, pivot = get_pivot(w)
            buffer_0[w] = single_step(A, b, buffer_1, buffer_2, i, pivot)
        if weight % 2 == 0:
            for partition in generate_partitions(weight // 2, tuple(p - 1 for p in pnr_cutoffs)):
                output[partition] = buffer_0[tuple(i for p in partition for i in (p, p))]
    return output


def get_pivot(k: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    r"""
    Find the index and pivot for the next recursion step.

    Args:
        k: Tuple of integers representing current coordinates

    Returns:
        A list containing the index and pivot
    """
    modes = len(k) // 2
    pairs = [(k[2 * m], k[2 * m + 1]) for m in range(modes)]
    deltas = [abs(a - b) for (a, b) in pairs]
    m = np.argmax(deltas)
    if deltas[m] == 0:
        i = np.argmax(k)
        return i, k[:i] + (k[i] - 1,) + k[i + 1 :]
    if k[2 * m] > k[2 * m + 1]:
        return 2 * m, k[: 2 * m] + (k[2 * m] - 1,) + k[2 * m + 1 :]
    return 2 * m + 1, k[: 2 * m + 1] + (k[2 * m + 1] - 1,) + k[2 * m + 2 :]


def single_step(A, b, buffer_1, buffer_2, i, pivot):
    r"""
    Perform a single step in the fast diagonal algorithm.

    Args:
        A: Matrix of parameters
        b: Vector of parameters
        buffer_1: Previous buffer
        buffer_2: Buffer from two steps ago
        i: lowered index
        pivot: Pivot coordinates

    Returns:
        Updated values for the current step
    """
    val = b[i + 2] * buffer_1[pivot]
    mat = buffer_1[pivot]
    val[1:, :] += A[i + 2, 0] * SQRT[1 : mat.shape[0]][:, None] * mat[:-1]
    val[:, 1:] += A[i + 2, 1] * SQRT[1 : mat.shape[1]][None, :] * mat[:, :-1]
    for j, p in enumerate(pivot):
        if p > 0:
            val += A[i + 2, j + 2] * SQRT[p] * buffer_2[lower(pivot, j)]
    return val / SQRT[pivot[i] + 1]


def lower(pivot, j):
    r"""
    Decrease the j-th coordinate of the pivot by 1.

    Args:
        pivot: Tuple of coordinates
        j: Index to decrease

    Returns:
        New tuple with j-th coordinate decreased by 1
    """
    return pivot[:j] + (pivot[j] - 1,) + pivot[j + 1 :]


def generate_partitions(w: int, m: tuple[int]):
    r"""Generate partitions of weight w where each element i cannot exceed m[i]"""
    if len(m) == 1:
        if w <= m[0]:
            yield (w,)
        return
    max_first = min(w, m[0])
    for first in range(max_first, -1, -1):
        for rest in generate_partitions(w - first, m[1:]):
            yield (first,) + rest


@lru_cache(maxsize=None)
def enumerate_diagonal_coords(weight: int, m: tuple[int]) -> list[tuple[int, ...]]:
    r"""
    Enumerate all coordinates of a given weight (sum(coordinate) == weight) and length 2*n_modes
    that satisfy the following conditions:
      - For each pair (a,b) must satisfy |a - b| ≤ 2.
      - At most one pair can have |a - b| == 2.

    Examples:
      enumerate_coords(2, (1,))
          -> [(2,0), (1,1), (0,2)]
      enumerate_coords(3, (1,))
          -> [(2,1), (1,2)]
    """
    valid = []
    n_modes = len(m)

    for partition in generate_partitions(weight, tuple(2 * k for k in m)):
        delta2_options = []
        non_delta2_options = []
        for i in range(n_modes):
            s = partition[i]
            max_val = m[i]
            l = s // 2 - 1
            u = s // 2 + 1
            d = s // 2
            if s % 2 == 0:
                delta2_options.append([(l, u), (u, l)] if 0 <= l and u <= max_val else [])
                non_delta2_options.append([(d, d)] if d <= max_val else [])
            else:
                delta2_options.append([])
                non_delta2_options.append([(d, u), (u, d)] if u <= max_val else [])
        # first include combinations which have exactly one pair with delta==2
        for i in range(n_modes):
            product_options = (
                non_delta2_options[:i] + [delta2_options[i]] + non_delta2_options[i + 1 :]
            )
            for combo in product(*product_options):
                coord = tuple(num for pair in combo for num in pair)
                valid.append(coord)
        # Now include combinations which have zero pairs with delta==2
        for combo in product(*non_delta2_options):
            coord = tuple(num for pair in combo for num in pair)
            valid.append(coord)

    return valid

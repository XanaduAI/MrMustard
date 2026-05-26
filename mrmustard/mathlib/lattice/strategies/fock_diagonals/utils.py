# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for the ``fock_diagonals`` restricted diagonal lattice.

The restricted diagonal lattice tracks bra/ket coordinate pairs ``(a, b)``
for each measured mode, subject to two constraints:

1. Each pair satisfies ``|a - b| ≤ 2``.
2. At most one pair across all modes may have ``|a - b| = 2``.

Coordinates are grouped by *weight* ``w = Σ(a_i + b_i)`` and traversed in
increasing weight order, mirroring the dependency structure of the recurrence
(each coordinate depends only on weights ``w-1`` and ``w-2``).

Inside Numba-compiled kernels, coordinate tuples are encoded as single
integers via a mixed-radix positional system, enabling ``O(log n)`` neighbor
lookups by binary search on sorted integer arrays.
"""

from collections.abc import Iterator
from functools import cache
from itertools import product
from typing import NamedTuple

import numba
import numpy as np

type DiagonalCoords = tuple[int, ...]


class DiagonalLatticeArrays(NamedTuple):
    r"""Precomputed lattice data consumed by Numba kernels in ``conditional_dm`` and ``diagonal_amplitudes``.

    Field layout matches ``precompute_lattice``; all array members use dtype
    ``numpy.int64``.
    """

    coords_array: np.ndarray
    keys_array: np.ndarray
    weight_starts: np.ndarray
    sorted_keys: np.ndarray
    sorted_to_global: np.ndarray
    strides: np.ndarray


def generate_partitions(
    total_weight: int, upper_bounds: tuple[int, ...]
) -> Iterator[tuple[int, ...]]:
    r"""Yield tuples summing to ``total_weight`` with bounded entries.

    Args:
        total_weight: Target sum of each yielded tuple.
        upper_bounds: Per-component inclusive upper limits.

    Yields:
        Tuples ``t`` with ``sum(t) == total_weight`` and
        ``t[i] <= upper_bounds[i]``.
    """
    if len(upper_bounds) == 1:
        if total_weight <= upper_bounds[0]:
            yield (total_weight,)
        return
    max_first = min(total_weight, upper_bounds[0])
    for first in range(max_first, -1, -1):
        for rest in generate_partitions(total_weight - first, upper_bounds[1:]):
            yield (first, *rest)


def _bra_ket_pairs_for_mode(
    mode_weight: int, max_photons: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r"""Return the valid (bra, ket) pairs for one mode given its total weight.

    Args:
        mode_weight: Sum ``a + b`` for this mode's bra/ket pair.
        max_photons: Maximum photon number allowed on this mode (inclusive).

    Each mode's bra/ket pair ``(a, b)`` satisfies ``a + b = mode_weight`` and
    ``|a - b| ≤ 2``.  Pairs are split into two groups:

    * **close pairs**: ``|a - b| ≤ 1`` (the "safe" choices that don't consume
      the single distance-2 budget).
    * **distance-two pairs**: ``|a - b| = 2`` (only one mode across all modes
      may use such a pair in any given coordinate).

    Returns:
        ``(close_pairs, distance_two_pairs)``
    """
    mid = mode_weight // 2

    if mode_weight % 2 == 0:
        close_pairs = [(mid, mid)] if mid <= max_photons else []
        low, high = mid - 1, mid + 1
        distance_two_pairs = [(low, high), (high, low)] if low >= 0 and high <= max_photons else []
    else:
        high = mid + 1
        close_pairs = [(mid, high), (high, mid)] if high <= max_photons else []
        distance_two_pairs = []

    return close_pairs, distance_two_pairs


@cache
def enumerate_diagonal_coords(
    weight: int, max_photons_per_mode: tuple[int, ...]
) -> list[tuple[int, ...]]:
    r"""Enumerate diagonal coordinates of given weight on the restricted lattice.

    A coordinate is a flat tuple ``(a₁, b₁, a₂, b₂, …)`` where ``(aₖ, bₖ)``
    is the bra/ket pair for measured mode ``k``.  The constraints are:

    * ``aₖ + bₖ`` sums to the mode's share of the total weight,
    * ``|aₖ - bₖ| ≤ 2`` for every mode,
    * at most one mode has ``|aₖ - bₖ| = 2``.

    Args:
        weight: Total lattice weight for this slice.
        max_photons_per_mode: Per measured mode, the maximum photon number
            (inclusive).  Fock arrays along that mode have length
            ``max_photons + 1``.

    Returns:
        All coordinate tuples at this weight, each a flat tuple
        ``(a0, b0, a1, b1, ...)`` of nonnegative integers obeying the
        restricted-diagonal constraints.

    The function first partitions the total weight among modes (each mode can
    receive up to ``2 * max_photons_k``), then for each partition enumerates
    all valid bra/ket assignments respecting the distance-2 budget.
    """
    n_modes = len(max_photons_per_mode)
    max_mode_weights = tuple(2 * m for m in max_photons_per_mode)
    valid = []

    for mode_weights in generate_partitions(weight, max_mode_weights):
        per_mode_close = []
        per_mode_dist2 = []
        for mode_idx in range(n_modes):
            close, dist2 = _bra_ket_pairs_for_mode(
                mode_weights[mode_idx], max_photons_per_mode[mode_idx]
            )
            per_mode_close.append(close)
            per_mode_dist2.append(dist2)

        # Coordinates where exactly one mode uses a distance-2 pair.
        for dist2_mode in range(n_modes):
            choices_per_mode = [
                per_mode_dist2[dist2_mode] if mode_idx == dist2_mode else per_mode_close[mode_idx]
                for mode_idx in range(n_modes)
            ]
            valid.extend(
                tuple(val for pair in combo for val in pair) for combo in product(*choices_per_mode)
            )

        # Coordinates where all modes use close pairs (no distance-2 pair used).
        valid.extend(
            tuple(val for pair in combo for val in pair) for combo in product(*per_mode_close)
        )

    return valid


def precompute_lattice(
    pnr_cutoffs: tuple[int, ...],
    coord_rank: int,
    max_weight: int,
    max_photon_number: int,
) -> DiagonalLatticeArrays:
    r"""Enumerate all diagonal-lattice coordinates and build lookup arrays for Numba kernels.

    Both ``conditional_dm.py`` and ``diagonal_amplitudes.py`` need to look up lattice neighbors
    inside Numba-compiled kernels where Python dicts are unavailable.
    This function encodes each coordinate tuple as a single integer using a
    mixed-radix positional system with base ``max_photon_number + 1`` (digits
    run from ``0`` through ``max_photon_number`` inclusive), then sorts keys
    within each weight level so that neighbor lookups become ``O(log n)``
    binary searches on plain ``int64`` arrays.

    Args:
        pnr_cutoffs: Per measured mode, maximum photon number (inclusive).
        coord_rank: Length of each coordinate tuple (``2 * n_modes``).
        max_weight: Maximum lattice weight to enumerate.
        max_photon_number: ``max(pnr_cutoffs)`` — largest digit in the encoding.

    The key property: decrementing coordinate dimension ``d`` corresponds to
    subtracting ``strides[d]`` from the integer key, so the neighbor key is
    computed in ``O(1)`` without reconstructing the full coordinate tuple.

    Returns:
        Named tuple with:

        coords_array
            ``(n_total, coord_rank)`` — raw coordinate tuples, ordered by
            weight then by enumeration order within each weight.
        keys_array
            ``(n_total,)`` — integer key for each coordinate, same order as in ``coords_array``.
        weight_starts
            ``(max_weight + 2,)`` — ``weight_starts[w]`` starts weight ``w``;
            ``weight_starts[w + 1]`` ends it.
        sorted_keys
            ``(n_total,)`` — keys sorted within each weight level (binary-search haystack).
        sorted_to_global
            ``(n_total,)`` — sorted position to global index in ``coords_array`` / ``keys_array``.
        strides
            ``(coord_rank,)`` — mixed-radix weights: ``key = sum(coords[d] * strides[d])``.

        All of the above arrays use dtype ``numpy.int64``.
    """
    base = max_photon_number + 1
    strides = np.ones(coord_rank, dtype=np.int64)
    for dim in range(1, coord_rank):
        strides[dim] = strides[dim - 1] * base

    origin = (0, 0) * (coord_rank // 2)
    all_coords: list[tuple[int, ...]] = [origin]
    all_keys: list[int] = [0]
    weight_boundaries: list[int] = [0, 1]

    for weight in range(1, max_weight + 1):
        for coord in enumerate_diagonal_coords(weight, pnr_cutoffs):
            key = sum(coord[dim] * int(strides[dim]) for dim in range(coord_rank))
            all_coords.append(coord)
            all_keys.append(key)
        weight_boundaries.append(len(all_coords))

    n_total = len(all_coords)
    coords_array = (
        np.array(all_coords, dtype=np.int64)
        if n_total
        else np.empty((0, coord_rank), dtype=np.int64)
    )
    keys_array = np.array(all_keys, dtype=np.int64)
    weight_starts = np.array(weight_boundaries, dtype=np.int64)

    sorted_keys = np.empty(n_total, dtype=np.int64)
    sorted_to_global = np.empty(n_total, dtype=np.int64)
    for weight in range(max_weight + 1):
        start = weight_starts[weight]
        end = weight_starts[weight + 1]
        if end == start:
            continue
        weight_keys = keys_array[start:end]
        sort_order = np.argsort(weight_keys)
        sorted_keys[start:end] = weight_keys[sort_order]
        sorted_to_global[start:end] = start + sort_order

    return DiagonalLatticeArrays(
        coords_array, keys_array, weight_starts, sorted_keys, sorted_to_global, strides
    )


@numba.njit(cache=True)
def binary_search(sorted_arr, start, end, target):  # pragma: no cover
    r"""Return the index of ``target`` in ``sorted_arr[start:end]``, or ``-1``.

    Args:
        sorted_arr: Sorted 1-D array of keys (typically ``int64``).
        start: Inclusive slice start index.
        end: Exclusive slice end index.
        target: Value to locate.

    Returns:
        Index ``i`` with ``start <= i < end`` and ``sorted_arr[i] == target``,
        or ``-1`` if none exists.
    """
    lo = start
    hi = end - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        if sorted_arr[mid] == target:
            return mid
        if sorted_arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

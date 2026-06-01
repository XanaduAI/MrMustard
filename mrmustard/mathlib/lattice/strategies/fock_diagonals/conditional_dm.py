# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conditional density matrices via the ``fock_diagonals`` diagonal lattice.

Traverses a reduced measured-mode lattice and applies the same entrywise
pivot averaging used by the stable full ``vanilla`` recurrence.

For each conditional-density-matrix entry ``(m, n)`` every valid recurrence
pivot inside the restricted lattice is averaged:

* leftover bra pivot,
* leftover ket pivot,
* every measured bra/ket pivot with all required second-order neighbors.

Averaging over all available valid pivots preserves exactness while
substantially improving numerical stability.

The entire weight sweep runs inside a single Numba-compiled kernel that
stores blocks in a three-slot sliding window and resolves dependencies via
binary search on mixed-radix integer keys — no Python-level dict lookups,
block copies, or per-entry allocation in the hot path.
"""

from __future__ import annotations

import numba
import numpy as np

from mrmustard.mathlib.cython_lattice import SQRT
from mrmustard.utils.typing import ComplexMatrix, ComplexScalar, ComplexTensor, ComplexVector

from .utils import binary_search, precompute_lattice

# ---------------------------------------------------------------------------
# Index convention for A and b after permutation
# ---------------------------------------------------------------------------
# After the bra/ket interleaving permutation applied in fock_diagonals_1leftover,
# the A matrix and b vector are indexed as:
#
#   0          → leftover bra
#   1          → leftover ket
#   2 + 2*k    → measured mode k, bra    (k = 0, 1, …, n_pnr - 1)
#   2 + 2*k+1  → measured mode k, ket
#
# The diagonal-lattice coordinate tuple (a₀, b₀, a₁, b₁, …) uses the
# same ordering for the measured modes (without the leftover), so
# ``coords[d]`` maps to A/b index ``d + 2``.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _fill_block(  # noqa: C901
    block,
    coords,
    seeded_origin,
    A,
    b,
    sqrt_values,
    prev_weight_blocks,
    lower_neighbor_idx,
    older_weight_blocks,
    older_neighbor_idx,
):  # pragma: no cover
    r"""Fill one conditional-density-matrix block via pivot averaging.

    For each entry ``(m, n)`` of the block, averages all valid recurrence
    pivots (leftover bra, leftover ket, and measured-mode pivots), matching
    the stable full ``vanilla`` strategy on the restricted lattice.

    Performance: everything that depends only on the *coordinate* (not on
    the inner-loop indices ``m, n``) is hoisted out of the entry loop.
    Neighbor blocks are read directly from the sliding-window arrays via
    pre-resolved integer indices (``-1`` = neighbor absent from lattice).

    Args:
        block: Leftover-mode Fock block with shape ``(n_max + 1, n_max + 1)``
            where ``n_max`` is the maximum leftover photon number (inclusive).
        coords: ``(coord_rank,)`` measured-mode coordinate of this block.
        seeded_origin: if True, ``block[0, 0]`` is already seeded (skip it).
        A: Bargmann matrix after bra/ket interleaving permutation.
        b: Bargmann vector after bra/ket interleaving permutation.
        sqrt_values: Precomputed ``sqrt(0), sqrt(1), …`` up to the largest
            photon number appearing along any lattice dimension.
        prev_weight_blocks: ``(n_prev, cutoff_m, cutoff_n)`` — all blocks at
            weight ``w-1`` (the "previous" sliding-window slot).
        lower_neighbor_idx: ``(coord_rank,)`` — for each dimension ``d``,
            the local index of the weight-(w-1) neighbor in
            ``prev_weight_blocks``, or ``-1`` if absent.
        older_weight_blocks: ``(n_older, cutoff_m, cutoff_n)`` — all blocks
            at weight ``w-2`` (the "older" sliding-window slot).
        older_neighbor_idx: ``(coord_rank, coord_rank)`` — for dimensions
            ``(d, j)``, the local index of the weight-(w-2) neighbor obtained
            by decrementing both ``d`` and ``j``, or ``-1`` if absent.
    """
    coord_rank = coords.shape[0]
    cutoff_m = block.shape[0]
    cutoff_n = block.shape[1]

    # ── Identify nonzero dimensions ──
    # Only dimensions with nonzero photon counts contribute measured-mode
    # terms and require lower neighbors for the leftover pivots.
    nonzero_dims = np.empty(coord_rank, dtype=np.int64)
    n_nonzero = 0
    for dim in range(coord_rank):
        if coords[dim] > 0:
            nonzero_dims[n_nonzero] = dim
            n_nonzero += 1

    # ── Leftover pivot validity ──
    # Both leftover bra and ket pivots need the same set of
    # lower neighbors (one per nonzero measured dimension), so their
    # validity is identical and checked once.
    leftover_pivots_valid = True
    for k in range(n_nonzero):
        if lower_neighbor_idx[nonzero_dims[k]] < 0:
            leftover_pivots_valid = False
            break

    # ── Pre-compute leftover pivot coefficients ──
    # The measured-mode sums use A[0, d+2]*sqrt(a_d) for
    # bra and A[1, d+2]*sqrt(a_d) for ket.  These depend on the coordinate but
    # NOT on the inner-loop variables (m, n).
    leftover_bra_meas_coeff = np.empty(n_nonzero, dtype=np.complex128)
    leftover_ket_meas_coeff = np.empty(n_nonzero, dtype=np.complex128)
    leftover_lower_local_idx = np.empty(n_nonzero, dtype=np.int64)
    for k in range(n_nonzero):
        dim = nonzero_dims[k]
        leftover_bra_meas_coeff[k] = A[0, dim + 2] * sqrt_values[coords[dim]]
        leftover_ket_meas_coeff[k] = A[1, dim + 2] * sqrt_values[coords[dim]]
        leftover_lower_local_idx[k] = lower_neighbor_idx[dim]

    # ── Pre-compute measured pivot data ──
    # For each valid measured pivot (dimension d with a usable lower
    # neighbor AND all required second-order neighbors), collect the
    # pre-computable coefficients into parallel arrays ("struct of arrays"
    # layout required by Numba).
    meas_lower_local_idx = np.empty(coord_rank, dtype=np.int64)
    meas_b_coeff = np.empty(coord_rank, dtype=np.complex128)
    meas_A_bra_coeff = np.empty(coord_rank, dtype=np.complex128)
    meas_A_ket_coeff = np.empty(coord_rank, dtype=np.complex128)
    meas_sqrt_divisor = np.empty(coord_rank, dtype=np.float64)
    meas_num_older = np.empty(coord_rank, dtype=np.int64)
    meas_older_local_idx = np.empty((coord_rank, coord_rank), dtype=np.int64)
    meas_older_coeff = np.empty((coord_rank, coord_rank), dtype=np.complex128)
    n_valid_measured_pivots = 0

    for k in range(n_nonzero):
        dim = nonzero_dims[k]
        lower_local = lower_neighbor_idx[dim]
        if lower_local < 0:
            continue

        # Check that ALL second-order neighbors of this pivot exist.
        all_older_present = True
        num_older = 0
        for j in range(coord_rank):
            # Photon count at dimension j in the pivot coordinate
            # (current coord with dimension `dim` decremented by 1).
            pivot_photon_count = coords[j] - (1 if j == dim else 0)
            if pivot_photon_count <= 0:
                continue
            older_local = older_neighbor_idx[dim, j]
            if older_local < 0:
                all_older_present = False
                break
            meas_older_local_idx[n_valid_measured_pivots, num_older] = older_local
            meas_older_coeff[n_valid_measured_pivots, num_older] = (
                A[dim + 2, j + 2] * sqrt_values[pivot_photon_count]
            )
            num_older += 1

        if not all_older_present:
            continue

        meas_lower_local_idx[n_valid_measured_pivots] = lower_local
        meas_b_coeff[n_valid_measured_pivots] = b[dim + 2]
        meas_A_bra_coeff[n_valid_measured_pivots] = A[dim + 2, 0]
        meas_A_ket_coeff[n_valid_measured_pivots] = A[dim + 2, 1]
        meas_sqrt_divisor[n_valid_measured_pivots] = sqrt_values[coords[dim]]
        meas_num_older[n_valid_measured_pivots] = num_older
        n_valid_measured_pivots += 1

    # ── Pivot counts per (m, n) region ──
    # The number of valid pivots (= averaging denominator) depends on
    # whether the leftover bra/ket pivots are available at (m, n):
    #   interior (m>0, n>0): bra + ket + measured
    #   top row  (m>0, n=0): bra + measured
    #   left col (m=0, n>0): ket + measured
    #   origin   (m=0, n=0): measured only
    leftover_count = 1 if leftover_pivots_valid else 0
    num_pivots_interior = 2 * leftover_count + n_valid_measured_pivots
    num_pivots_bra_only = leftover_count + n_valid_measured_pivots
    num_pivots_ket_only = leftover_count + n_valid_measured_pivots
    num_pivots_origin = n_valid_measured_pivots

    A_bra_bra = A[0, 0]
    A_bra_ket = A[0, 1]
    A_ket_bra = A[1, 0]
    A_ket_ket = A[1, 1]
    b_bra = b[0]
    b_ket = b[1]

    # ── Entry loop: evaluate and average all valid pivots at each (m, n) ──
    for m in range(cutoff_m):
        for n in range(cutoff_n):
            if seeded_origin and m == 0 and n == 0:
                continue

            total = 0.0 + 0.0j

            # Leftover bra pivot: step down one photon in leftover bra.
            if m > 0 and leftover_pivots_valid:
                value = b_bra * block[m - 1, n]
                if m > 1:
                    value += A_bra_bra * sqrt_values[m - 1] * block[m - 2, n]
                if n > 0:
                    value += A_bra_ket * sqrt_values[n] * block[m - 1, n - 1]
                for k in range(n_nonzero):
                    value += (
                        leftover_bra_meas_coeff[k]
                        * prev_weight_blocks[leftover_lower_local_idx[k], m - 1, n]
                    )
                total += value / sqrt_values[m]

            # Leftover ket pivot: step down one photon in leftover ket.
            if n > 0 and leftover_pivots_valid:
                value = b_ket * block[m, n - 1]
                if m > 0:
                    value += A_ket_bra * sqrt_values[m] * block[m - 1, n - 1]
                if n > 1:
                    value += A_ket_ket * sqrt_values[n - 1] * block[m, n - 2]
                for k in range(n_nonzero):
                    value += (
                        leftover_ket_meas_coeff[k]
                        * prev_weight_blocks[leftover_lower_local_idx[k], m, n - 1]
                    )
                total += value / sqrt_values[n]

            # Measured-mode pivots: step down one photon in each
            # valid measured dimension.
            for p in range(n_valid_measured_pivots):
                lower_local = meas_lower_local_idx[p]
                value = meas_b_coeff[p] * prev_weight_blocks[lower_local, m, n]
                if m > 0:
                    value += (
                        meas_A_bra_coeff[p]
                        * sqrt_values[m]
                        * prev_weight_blocks[lower_local, m - 1, n]
                    )
                if n > 0:
                    value += (
                        meas_A_ket_coeff[p]
                        * sqrt_values[n]
                        * prev_weight_blocks[lower_local, m, n - 1]
                    )
                for older_k in range(meas_num_older[p]):
                    value += (
                        meas_older_coeff[p, older_k]
                        * older_weight_blocks[meas_older_local_idx[p, older_k], m, n]
                    )
                total += value / meas_sqrt_divisor[p]

            # Average over all valid pivots.
            if m > 0 and n > 0:
                num_pivots = num_pivots_interior
            elif m > 0:
                num_pivots = num_pivots_bra_only
            elif n > 0:
                num_pivots = num_pivots_ket_only
            else:
                num_pivots = num_pivots_origin

            if num_pivots > 0:
                block[m, n] = total / num_pivots


@numba.njit(cache=True)
def _run_1leftover_kernel(  # noqa: C901
    blocks,
    output_flat,
    coords_array,
    keys_array,
    weight_starts,
    sorted_keys,
    sorted_to_global,
    strides,
    pnr_output_strides,
    A,
    b,
    sqrt_values,
    c,
    max_weight,
    n_pnr_modes,
):  # pragma: no cover
    r"""Process the entire weight sweep for conditional density-matrix blocks.

    Uses a three-slot sliding window over weight levels so that only
    weights ``w``, ``w-1``, and ``w-2`` coexist in memory.  All neighbor
    lookups are integer binary searches inside Numba — the Python
    interpreter is never re-entered.

    For each coordinate at the current weight ``w``, the kernel:

    1. Resolves first-order neighbors (weight ``w-1``) by computing
       ``key - strides[d]`` and binary-searching the sorted key array.
    2. For each found neighbor, resolves second-order neighbors
       (weight ``w-2``) the same way.
    3. Passes the resolved neighbor indices to ``_fill_block``.
    4. On even weights, copies diagonal blocks (bra == ket for every
       measured-mode pair) to the output array.
    """
    coord_rank = coords_array.shape[1]
    cutoff_m = blocks.shape[2]
    cutoff_n = blocks.shape[3]

    curr_slot = 0
    prev_slot = 1
    older_slot = 2

    # Scratch arrays reused across all coordinates.
    # lower_neighbor_idx[d]:    local index in the prev-weight slot for
    #                           the neighbor with dimension d decremented.
    # older_neighbor_idx[d, j]: local index in the older-weight slot for
    #                           the neighbor with both d and j decremented.
    # A value of -1 means "neighbor not in the lattice".
    lower_neighbor_idx = np.empty(coord_rank, dtype=np.int64)
    older_neighbor_idx = np.empty((coord_rank, coord_rank), dtype=np.int64)

    for weight in range(max_weight + 1):
        weight_start = weight_starts[weight]
        weight_end = weight_starts[weight + 1]
        num_coords = weight_end - weight_start

        # Zero only the blocks we'll actually fill.
        for local_idx in range(num_coords):
            for mi in range(cutoff_m):
                for ni in range(cutoff_n):
                    blocks[curr_slot, local_idx, mi, ni] = 0.0

        if weight == 0:
            # Seed the origin block and fill via leftover-only recurrence.
            blocks[curr_slot, 0, 0, 0] = c
            _fill_block(
                blocks[curr_slot, 0],
                coords_array[weight_start],
                True,
                A,
                b,
                sqrt_values,
                blocks[prev_slot],
                lower_neighbor_idx,
                blocks[older_slot],
                older_neighbor_idx,
            )
        else:
            prev_start = weight_starts[weight - 1]
            prev_end = weight_starts[weight]
            has_older = weight >= 2
            older_start = weight_starts[weight - 2] if has_older else np.int64(0)
            older_end = prev_start if has_older else np.int64(0)

            for local_idx in range(num_coords):
                global_idx = weight_start + local_idx
                coord_key = keys_array[global_idx]

                lower_neighbor_idx[:] = -1
                older_neighbor_idx[:, :] = -1

                for dim in range(coord_rank):
                    if coords_array[global_idx, dim] == 0:
                        continue

                    # First-order neighbor: decrement dimension `dim`.
                    neighbor_key = coord_key - strides[dim]
                    found_pos = binary_search(sorted_keys, prev_start, prev_end, neighbor_key)
                    if found_pos == -1:
                        continue
                    neighbor_global_idx = sorted_to_global[found_pos]
                    lower_neighbor_idx[dim] = neighbor_global_idx - prev_start

                    if not has_older:
                        continue

                    # Second-order neighbors: for each nonzero dimension j
                    # of the pivot, decrement again to reach weight w-2.
                    for j in range(coord_rank):
                        if coords_array[neighbor_global_idx, j] == 0:
                            continue
                        older_key = neighbor_key - strides[j]
                        found_pos = binary_search(sorted_keys, older_start, older_end, older_key)
                        if found_pos == -1:
                            continue
                        older_global_idx = sorted_to_global[found_pos]
                        older_neighbor_idx[dim, j] = older_global_idx - older_start

                _fill_block(
                    blocks[curr_slot, local_idx],
                    coords_array[global_idx],
                    False,
                    A,
                    b,
                    sqrt_values,
                    blocks[prev_slot],
                    lower_neighbor_idx,
                    blocks[older_slot],
                    older_neighbor_idx,
                )

        # Diagonal blocks (bra == ket for every measured-mode pair) exist
        # only at even weights and correspond to physical PNR outcomes.
        if weight % 2 == 0:
            for local_idx in range(num_coords):
                global_idx = weight_start + local_idx
                is_diagonal = True
                for mode in range(n_pnr_modes):
                    if coords_array[global_idx, 2 * mode] != coords_array[global_idx, 2 * mode + 1]:
                        is_diagonal = False
                        break
                if is_diagonal:
                    flat_pnr_idx = 0
                    for mode in range(n_pnr_modes):
                        flat_pnr_idx += (
                            coords_array[global_idx, 2 * mode] * pnr_output_strides[mode]
                        )
                    for mi in range(cutoff_m):
                        for ni in range(cutoff_n):
                            output_flat[flat_pnr_idx, mi, ni] = blocks[curr_slot, local_idx, mi, ni]

        # Rotate the three-slot sliding window.
        tmp = older_slot
        older_slot = prev_slot
        prev_slot = curr_slot
        curr_slot = tmp


def fock_diagonals_1leftover(
    A: ComplexMatrix,
    b: ComplexVector,
    c: ComplexScalar,
    output_cutoff: int,
    pnr_cutoffs: tuple[int, ...],
) -> ComplexTensor:
    r"""Density matrices on mode 0 conditioned on photon numbers on the other modes.

    .. math::
        \mathrm{out}_{ij;mn\dots} = \langle i, m, n, \dots | \rho | j, m, n, \dots \rangle

    Mode ``0`` is the leftover (conditional) mode; modes ``1, \ldots, n_{\mathrm{pnr}}``
    are the PNR sectors with cutoffs ``pnr_cutoffs``.

    Note:
        Not numerically stable when the Bargmann vector ``b`` has large
        magnitudes (large displacements). Always check your outputs.

    Args:
        A: Bargmann matrix, shape ``(2 * n_modes, 2 * n_modes)``, dtype
            ``complex128``, with ``n_modes = 1 + len(pnr_cutoffs)``.
        b: Bargmann vector, shape ``(2 * n_modes,)``, dtype ``complex128``.
        c: Bargmann scalar; Python ``complex`` or NumPy complex scalar
            coercible with ``numpy.asarray``, stored as ``complex128``.
        output_cutoff: Maximum photon number on the leftover mode (inclusive);
            leftover Fock axis length is ``output_cutoff + 1``.
        pnr_cutoffs: Length ``n_pnr``; ``pnr_cutoffs[k]`` is the maximum photon
            number on measured mode ``k + 1``. Fock axis length for
            that outcome index is ``pnr_cutoffs[k] + 1``.

    Returns:
        ``numpy.ndarray`` of ``complex128``. Shape
        ``(output_cutoff + 1, output_cutoff + 1) + tuple(p + 1 for p in pnr_cutoffs)``
        with the two leftover indices leading the tensor.

    Raises:
        ValueError: If any entry of ``pnr_cutoffs`` is negative.
    """
    if any(cutoff < 0 for cutoff in pnr_cutoffs):
        raise ValueError("All cutoffs must be non-negative.")

    # Interleave bra/ket indices so that each mode's bra and ket are adjacent:
    # [mode0_bra, mode0_ket, mode1_bra, mode1_ket, ...].  This aligns the
    # coordinate dimensions with the A/b index convention documented at the
    # top of this module.
    n_modes = len(pnr_cutoffs) + 1
    permutation = [index for mode in range(n_modes) for index in (mode, mode + n_modes)]
    A = np.asarray(A, dtype=np.complex128)[permutation, :][:, permutation]
    b = np.asarray(b, dtype=np.complex128)[permutation]
    c = np.asarray(c, dtype=np.complex128)

    # Fock axis lengths: photon indices run 0..cutoff inclusive ⇒ shape = cutoff + 1.
    leftover_fock_shape = output_cutoff + 1
    n_pnr = len(pnr_cutoffs)
    coord_rank = 2 * n_pnr
    pnr_fock_shape = tuple(cutoff + 1 for cutoff in pnr_cutoffs)
    output = np.zeros(
        (*pnr_fock_shape, leftover_fock_shape, leftover_fock_shape), dtype=np.complex128
    )

    max_cutoff = max((output_cutoff, *pnr_cutoffs), default=output_cutoff)
    sqrt_values = SQRT[: max_cutoff + 1]

    # No measured modes: just fill the single leftover block directly.
    if n_pnr == 0:
        output[0, 0] = c
        _fill_block(
            output,
            np.empty(0, dtype=np.int64),
            True,
            A,
            b,
            sqrt_values,
            np.empty((0, leftover_fock_shape, leftover_fock_shape), dtype=np.complex128),
            np.empty(0, dtype=np.int64),
            np.empty((0, leftover_fock_shape, leftover_fock_shape), dtype=np.complex128),
            np.empty((0, 0), dtype=np.int64),
        )
        return output

    max_weight = 2 * sum(pnr_cutoffs)
    max_photon_number = max(pnr_cutoffs)
    (
        coords_array,
        keys_array,
        weight_starts,
        sorted_keys,
        sorted_to_global,
        strides,
    ) = precompute_lattice(pnr_cutoffs, coord_rank, max_weight, max_photon_number)

    max_coords_per_weight = max(
        int(weight_starts[w + 1] - weight_starts[w]) for w in range(max_weight + 1)
    )

    # Three-slot sliding window: only weights w, w-1, w-2 coexist at once.
    blocks = np.zeros(
        (3, max_coords_per_weight, leftover_fock_shape, leftover_fock_shape), dtype=np.complex128
    )

    # C-order strides for flat indexing into the PNR output dimensions.
    pnr_output_strides = np.ones(n_pnr, dtype=np.int64)
    for i in range(n_pnr - 2, -1, -1):
        pnr_output_strides[i] = pnr_output_strides[i + 1] * (pnr_cutoffs[i + 1] + 1)

    n_pnr_entries = 1
    for cutoff in pnr_cutoffs:
        n_pnr_entries *= cutoff + 1
    output_flat = output.reshape(n_pnr_entries, leftover_fock_shape, leftover_fock_shape)

    _run_1leftover_kernel(
        blocks,
        output_flat,
        coords_array,
        keys_array,
        weight_starts,
        sorted_keys,
        sorted_to_global,
        strides,
        pnr_output_strides,
        A,
        b,
        sqrt_values,
        c,
        max_weight,
        n_pnr,
    )

    return np.moveaxis(output, [-2, -1], [0, 1])

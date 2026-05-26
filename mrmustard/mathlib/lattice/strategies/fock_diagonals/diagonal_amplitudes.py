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

"""Multimode Fock diagonal amplitudes via the ``fock_diagonals`` lattice.

Traverses the same reduced diagonal lattice used by ``conditional_dm.py``, but
computes scalar amplitudes rather than density-matrix blocks.  Each lattice
coordinate is a pair ``(a, b)`` per mode satisfying ``|a - b| ≤ 2`` (at most
one pair at distance two); on the even-weight diagonal slice ``a == b`` these
amplitudes are the Fock diagonal entries ``⟨n|ρ|n⟩``.

At every coordinate all valid recurrence pivots are averaged, matching the
stable ``vanilla`` strategy and giving numerically robust probabilities.
Batching over the Bargmann vector ``b`` and scalar
``c`` is supported via a trailing batch axis.

The recurrence kernel is Numba-compiled: all coordinates across every weight
level are precomputed into flat integer arrays with mixed-radix encoded keys,
and a single ``@njit`` kernel processes the entire weight sweep using binary
search for neighbor lookups — no Python-level dict lookups or function-call
overhead in the hot path.
"""

from __future__ import annotations

import numba
import numpy as np

from mrmustard.mathlib.cython_lattice import SQRT
from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexScalar,
    ComplexTensor,
    ComplexVector,
)

from .utils import (
    DiagonalCoords,
    binary_search,
    generate_partitions,
    precompute_lattice,
)


def _resolve_fock_diagonals_batch_shape(
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray]:
    r"""Infer batch dimensions shared by ``b`` and ``c``.

    Args:
        b: Bargmann vector, shape ``(2 * n_modes,)`` or ``(2 * n_modes, batch)``.
        c: Scalar or batch with shape matching ``b``'s batch axis when present.

    Returns:
        ``(batch_shape, c_arr)`` where ``c_arr`` is broadcast to ``batch_shape``
        when ``c`` was scalar and ``b`` is batched.
    """
    b_batch_shape = () if b.ndim == 1 else (b.shape[1],)
    c_batch_shape = () if c.ndim == 0 else c.shape

    if b_batch_shape and c_batch_shape and b_batch_shape != c_batch_shape:
        raise ValueError("The batch dimensions of b and c must match.")

    batch_shape = b_batch_shape or c_batch_shape
    if batch_shape and c.ndim == 0:
        return batch_shape, np.full(batch_shape, c, dtype=np.complex128)
    return batch_shape, c


def _prepare_fock_diagonals_inputs(
    A: ComplexMatrix,
    b: ComplexVector,
    c: ComplexScalar,
    pnr_cutoffs: DiagonalCoords,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    r"""Validate shapes, align batching of ``b``/``c``, and apply the bra/ket permutation.

    Args:
        A: Bargmann matrix, shape ``(2 * n_modes, 2 * n_modes)``, dtype ``complex128``.
        b: Bargmann vector, shape ``(2 * n_modes,)`` or batched
            ``(2 * n_modes, batch)``, dtype ``complex128``.
        c: Scalar or 1-D batch ``(batch,)``, dtype ``complex128``.
        pnr_cutoffs: Per measured mode, maximum photon number (inclusive).

    Returns:
        ``(A_perm, b_perm, c_aligned, batch_shape)`` with adjacent bra/ket
        ordering on ``A_perm`` / ``b_perm`` and ``c_aligned`` broadcast when
        only ``b`` is batched.

    Raises:
        ValueError: Negative cutoffs; non-square ``A``; wrong ranks on ``b`` or
            ``c``; mode count mismatch; or mismatched batch shapes on ``b`` and ``c``.
    """
    if any(cutoff < 0 for cutoff in pnr_cutoffs):
        raise ValueError("All cutoffs must be non-negative.")

    A = np.asarray(A, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("The matrix A must be square.")
    if b.ndim not in (1, 2):
        raise ValueError("The vector b must be one-dimensional or two-dimensional.")
    if c.ndim > 1:
        raise ValueError("The scalar c must be scalar or one-dimensional.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("The matrix A and vector b have incompatible dimensions.")

    n_modes = len(pnr_cutoffs)
    if A.shape[0] % 2 != 0 or A.shape[0] // 2 != n_modes:
        raise ValueError(
            "The Bargmann triple and cutoffs have incompatible dimensions: "
            f"received {A.shape[0] // 2} modes in (A, b) but expected {n_modes}."
        )

    batch_shape, c = _resolve_fock_diagonals_batch_shape(b, c)
    permutation = [index for mode in range(n_modes) for index in (mode, mode + n_modes)]
    A = A[permutation, :][:, permutation]
    b = b[permutation] if b.ndim == 1 else b[permutation, :]
    return A, b, c, batch_shape


@numba.njit(cache=True)
def _run_probs_kernel(  # noqa: C901
    values,
    coords_array,
    keys_array,
    weight_starts,
    sorted_keys,
    sorted_to_global,
    strides,
    A,
    b,
    sqrt_values,
    max_weight,
):  # pragma: no cover
    r"""Process the entire weight sweep for scalar diagonal amplitudes.

    Unlike ``conditional_dm.py`` which stores full density-matrix blocks per
    coordinate, here each coordinate holds a scalar amplitude (or a batch
    of them).  The ``values`` array has shape ``(n_total_coords, batch_size)``
    and persists across all weights — no sliding window needed since
    scalars are cheap.

    The recurrence for each coordinate averages over all valid pivots, matching
    the stable full ``vanilla`` strategy restricted to the diagonal lattice.
    For each nonzero dimension ``d``, the pivot is the neighbor at
    weight ``w-1`` obtained by decrementing ``d``.  A pivot is valid only
    if ALL its second-order (weight ``w-2``) neighbors exist in the lattice.
    """
    coord_rank = coords_array.shape[1]
    batch_size = values.shape[1]

    # Scratch arrays for collecting second-order neighbor info within
    # a single pivot evaluation.
    older_global_indices = np.empty(coord_rank, dtype=np.int64)
    older_dimensions = np.empty(coord_rank, dtype=np.int64)
    older_photon_counts = np.empty(coord_rank, dtype=np.int64)

    for weight in range(1, max_weight + 1):
        prev_start = weight_starts[weight - 1]
        prev_end = weight_starts[weight]
        has_older = weight >= 2
        older_start = weight_starts[weight - 2] if has_older else np.int64(0)
        older_end = prev_start if has_older else np.int64(0)
        curr_start = weight_starts[weight]
        curr_end = weight_starts[weight + 1]

        for coord_idx in range(curr_start, curr_end):
            num_valid_pivots = 0
            coord_key = keys_array[coord_idx]

            for dim in range(coord_rank):
                photon_count = coords_array[coord_idx, dim]
                if photon_count == 0:
                    continue

                # First-order neighbor at weight w-1.
                neighbor_key = coord_key - strides[dim]
                found_pos = binary_search(sorted_keys, prev_start, prev_end, neighbor_key)
                if found_pos == -1:
                    continue
                pivot_global_idx = sorted_to_global[found_pos]

                # Validate ALL second-order neighbors of this pivot.
                all_older_present = True
                num_older = 0
                for j in range(coord_rank):
                    pivot_photon_count = coords_array[pivot_global_idx, j]
                    if pivot_photon_count == 0:
                        continue
                    if not has_older:
                        all_older_present = False
                        break
                    older_key = neighbor_key - strides[j]
                    found_pos_older = binary_search(sorted_keys, older_start, older_end, older_key)
                    if found_pos_older == -1:
                        all_older_present = False
                        break
                    older_global_indices[num_older] = sorted_to_global[found_pos_older]
                    older_dimensions[num_older] = j
                    older_photon_counts[num_older] = pivot_photon_count
                    num_older += 1

                if not all_older_present:
                    continue

                # Accumulate this pivot's contribution across the batch.
                num_valid_pivots += 1
                divisor = sqrt_values[photon_count]
                for batch_idx in range(batch_size):
                    val = b[dim, batch_idx] * values[pivot_global_idx, batch_idx]
                    for older_k in range(num_older):
                        val += (
                            A[dim, older_dimensions[older_k]]
                            * sqrt_values[older_photon_counts[older_k]]
                            * values[older_global_indices[older_k], batch_idx]
                        )
                    values[coord_idx, batch_idx] += val / divisor

            # Average over all valid pivots (the stability trick).
            if num_valid_pivots > 0:
                inv_count = 1.0 / num_valid_pivots
                for batch_idx in range(batch_size):
                    values[coord_idx, batch_idx] *= inv_count


def fock_diagonals(
    A: ComplexMatrix,
    b: ComplexVector,
    c: ComplexScalar,
    pnr_cutoffs: DiagonalCoords,
) -> ComplexTensor:
    r"""Fock diagonal amplitudes of a multimode Gaussian operator.

    .. math::

        \mathrm{out}_{ij\dots} = \langle i, j, \ldots | G | i, j, \ldots \rangle

    If ``G`` is a density matrix, diagonal entries are multimode photon-number
    probabilities (real and non-negative up to numerical error).

    Args:
        A: Bargmann matrix, shape ``(2 * n_modes, 2 * n_modes)``, dtype ``complex128``.
        b: Bargmann vector, shape ``(2 * n_modes,)`` or batched
            ``(2 * n_modes, batch)``, dtype ``complex128``.
        c: Bargmann scalar or batch ``(batch,)``, dtype ``complex128``. When
            both ``b`` and ``c`` are batched, their batch lengths must agree.
        pnr_cutoffs: Tuple of length ``n_modes``; ``pnr_cutoffs[k]`` is the
            maximum photon number on measured mode ``k``.

    Returns:
        ``numpy.ndarray`` of ``complex128`` with shape
        ``tuple(p + 1 for p in pnr_cutoffs)``, or that shape plus a trailing
        batch dimension when ``b`` or ``c`` is batched.

    Raises:
        ValueError: Negative cutoffs; non-square ``A``; invalid ranks on ``b``
            or ``c``; ``A``/``b`` mode count not equal to ``len(pnr_cutoffs)``;
            or mismatched batch dimensions between ``b`` and ``c``.
    """
    A, b, c, batch_shape = _prepare_fock_diagonals_inputs(A, b, c, pnr_cutoffs)

    n_modes = len(pnr_cutoffs)
    coord_rank = 2 * n_modes
    fock_shape_per_mode = tuple(p + 1 for p in pnr_cutoffs)
    output = np.zeros(fock_shape_per_mode + batch_shape, dtype=np.complex128)
    output[(0,) * n_modes] = c

    max_weight = 2 * sum(pnr_cutoffs)
    if max_weight == 0:
        return output

    max_photon_number = max(pnr_cutoffs)
    sqrt_values = SQRT[: max_photon_number + 1]

    batch_size = batch_shape[0] if batch_shape else 1
    b_2d = np.broadcast_to(b[:, np.newaxis], (b.shape[0], batch_size)).copy() if b.ndim == 1 else b
    c_1d = np.full(batch_size, c, dtype=np.complex128) if c.ndim == 0 else c

    (
        coords_array,
        keys_array,
        weight_starts,
        sorted_keys,
        sorted_to_global,
        strides,
    ) = precompute_lattice(pnr_cutoffs, coord_rank, max_weight, max_photon_number)

    # One scalar per lattice coordinate per batch element.
    # All weights stay alive (scalars are small, unlike full blocks).
    n_total = coords_array.shape[0]
    values = np.zeros((n_total, batch_size), dtype=np.complex128)
    values[0, :] = c_1d

    _run_probs_kernel(
        values,
        coords_array,
        keys_array,
        weight_starts,
        sorted_keys,
        sorted_to_global,
        strides,
        A,
        b_2d,
        sqrt_values,
        max_weight,
    )

    # ── Collect physical probabilities from diagonal coordinates ──
    # Only even-weight coords where bra == ket for every mode pair (i.e.
    # the coordinate is (n,n) per mode) correspond to physical <n|rho|n>.
    # Their integer key is n*strides[2i] + n*strides[2i+1].
    for weight in range(0, max_weight + 1, 2):
        start = weight_starts[weight]
        end = weight_starts[weight + 1]

        key_to_global_idx: dict[int, int] = {}
        for idx in range(start, end):
            key_to_global_idx[int(keys_array[idx])] = idx

        for photon_numbers in generate_partitions(weight // 2, pnr_cutoffs):
            diagonal_key = sum(
                n_k * (int(strides[2 * mode]) + int(strides[2 * mode + 1]))
                for mode, n_k in enumerate(photon_numbers)
            )
            global_idx = key_to_global_idx[diagonal_key]
            if batch_shape:
                output[photon_numbers] = values[global_idx, :]
            else:
                output[photon_numbers] = values[global_idx, 0]

    return output

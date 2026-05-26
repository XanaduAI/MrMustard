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

r"""Wormhole algorithm for conditional states with one leftover mode.

This module implements wormhole entry points for quantum optics applications.
Given a multi-mode Gaussian state and a set of PNR measurement outcomes on all but one
mode, it computes the conditional state of the remaining "leftover" mode.

Two variants are provided:
- wormhole_1leftover_dm: For density matrices (DM), returns dict mapping PNR outcomes to conditional DMs
- wormhole_1leftover_ket: For kets, returns dict mapping PNR outcomes to conditional ket amplitudes (faster)

The wormhole algorithm efficiently traverses the Fock lattice in a snake-like pattern,
computing only the amplitudes needed along the path to each target index. This dramatically
reduces memory and computation for high-index targets compared to computing the full tensor.

For DMs:
    Memory complexity: O(2^(2M) × cutoff²) where M is the number of measured modes
    Time complexity: O(N × 2^(2M) × cutoff²) where N is total photon count

For Kets:
    Memory complexity: O(2^M × cutoff) where M is the number of measured modes
    Time complexity: O(N × 2^M × cutoff) where N is total photon count
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba.cpython.unsafe.tuple import tuple_setitem

from mrmustard.mathlib.cython_lattice import vanilla, vanilla_batched

from .core import Ab_at_pivot_directional
from .tree import create_visiting_tree

# Pre-computed square roots for efficient Fock recurrence computations
# Index i contains sqrt(i). Size chosen to support high photon counts.
SQRT = np.sqrt(np.arange(100000))

# Type aliases
PNR = tuple[int, ...]


def _reorder_for_wormhole(
    A: np.ndarray,
    b: np.ndarray,
    leftover_mode: int,
    is_dm: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reorder Bargmann (A, b) to put leftover mode last.

    For DMs: uses paired bra/ket ordering [bra_0, ket_0, bra_1, ket_1, ...]
    For Kets: uses simple mode ordering [mode_0, mode_1, ...]

    Supports both single and batched inputs (arbitrary leading batch dimensions).

    Args:
        A: Bargmann matrix with shape (..., n, n).
        b: Bargmann vector with shape (..., n).
        leftover_mode: Which mode to keep unmeasured (-1 for last mode).
        is_dm: True for density matrices, False for kets.

    Returns:
        (A_reordered, b_reordered, num_modes)
    """
    A = np.asarray(A)
    b = np.asarray(b)

    n = A.shape[-1]
    num_modes = n // 2 if is_dm else n
    L = leftover_mode % num_modes
    modes = list(range(L)) + list(range(L + 1, num_modes)) + [L]

    if is_dm:
        perm = [d for m in modes for d in (m, m + num_modes)]
        return A[..., perm, :][..., :, perm], b[..., perm], num_modes
    return A[..., modes, :][..., :, modes], b[..., modes], num_modes


def _validate_wormhole_inputs(
    pnr_outcomes: list[PNR],
    first_hypercube_pnr: PNR | None,
    num_measured_modes: int,
) -> PNR:
    """Validate wormhole inputs and return normalized first_hypercube_pnr.

    Args:
        pnr_outcomes: List of target PNR outcomes.
        first_hypercube_pnr: Starting PNR point (or None for origin).
        num_measured_modes: Number of measured modes.

    Returns:
        Normalized first_hypercube_pnr (defaults to zeros if None).

    Raises:
        ValueError: If first_hypercube_pnr exceeds any pnr_outcomes component.
    """
    first_hypercube_pnr = first_hypercube_pnr or (0,) * num_measured_modes

    if pnr_outcomes:
        min_outcomes = tuple(np.array(pnr_outcomes).min(axis=0))
        if any(p > q for p, q in zip(first_hypercube_pnr, min_outcomes)):
            raise ValueError(
                f"first_hypercube_pnr {first_hypercube_pnr} cannot exceed "
                f"minimum pnr_outcomes {min_outcomes}"
            )

    return first_hypercube_pnr


def wormhole_1leftover_dm(
    A: np.ndarray,
    b: np.ndarray,
    c: complex | np.ndarray,
    output_cutoff: int,
    pnr_outcomes: list[PNR],
    leftover_mode: int = -1,
    first_hypercube_pnr: PNR | None = None,
    stable: bool = True,
) -> dict[PNR, np.ndarray]:
    r"""Compute conditional density matrices for one leftover mode given PNR measurements.

    This is the primary wormhole entry point for quantum optics applications.
    Given a multi-mode Gaussian state and a set of PNR measurement outcomes
    on all but one mode, computes the conditional density matrix of the
    remaining "leftover" mode.

    Automatically dispatches to batched or non-batched implementation based on
    input dimensions:
    - If A has shape (n, n): single computation
    - If A has shape (batch, n, n): parallel computation over batch

    Args:
        A: Bargmann matrix. Shape (2*n_modes, 2*n_modes) for single or
            (batch, 2*n_modes, 2*n_modes) for batched computation.
            Uses MrMustard DM ordering: [bra_0, bra_1, ..., ket_0, ket_1, ...].
        b: Bargmann vector. Shape (2*n_modes,) or (batch, 2*n_modes).
        c: Bargmann scalar normalization. Scalar or array of shape (batch,).
        output_cutoff: Maximum photon number for output density matrix.
            The output shape will be (output_cutoff+1, output_cutoff+1).
        pnr_outcomes: List of PNR measurement outcomes to condition on.
            Each tuple has length = num_modes - 1 (measured modes only).
            E.g., for a 3-mode system with leftover_mode=0, pnr_outcomes=[(1, 4)]
            means measuring 1 photon on mode 1 and 4 photons on mode 2.
        leftover_mode: Which mode to keep unmeasured (default: last mode, -1).
            The returned density matrices will be for this mode.
        first_hypercube_pnr: Starting PNR point for the wormhole traversal.
            Must not exceed any pnr_outcomes component-wise.
            Default is (0, ..., 0), meaning start from vacuum on all measured modes.
        stable: Use numerically stable algorithm with renormalized Hermite polynomials.
            Recommended for high photon numbers.

    Returns:
        Dict mapping PNR outcomes to density matrices.
        - Single: Each value has shape (output_cutoff+1, output_cutoff+1).
        - Batched: Each value has shape (batch, output_cutoff+1, output_cutoff+1).

    Raises:
        ValueError: If pnr_outcomes is empty.
        ValueError: If first_hypercube_pnr exceeds any component of pnr_outcomes.

    Example:
        >>> # Single computation
        >>> A, b, c = state.bargmann_triple()
        >>> results = wormhole_1leftover_dm(
        ...     A, b, c, output_cutoff=70, pnr_outcomes=[(1, 4)], leftover_mode=0
        ... )
        >>> # results[(1, 4)] has shape (71, 71)
        >>>
        >>> # Batched computation (100 states in parallel)
        >>> A_batch = np.stack([A + noise for _ in range(100)])
        >>> b_batch = np.stack([b + noise for _ in range(100)])
        >>> c_batch = np.ones(100, dtype=complex)
        >>> results = wormhole_1leftover_dm(
        ...     A_batch, b_batch, c_batch, output_cutoff=70, pnr_outcomes=[(1, 4)]
        ... )
        >>> # results[(1, 4)] has shape (100, 71, 71)
    """
    A = np.asarray(A)

    # Dispatch based on input dimensions
    if A.ndim == 2:
        return _wormhole_1leftover_dm_single(
            A, b, c, output_cutoff, pnr_outcomes, leftover_mode, first_hypercube_pnr, stable
        )
    if A.ndim == 3:
        return _wormhole_1leftover_dm_batched(
            A, b, c, output_cutoff, pnr_outcomes, leftover_mode, first_hypercube_pnr, stable
        )
    raise ValueError(f"A must have 2 or 3 dimensions, got {A.ndim}")


def _wormhole_1leftover_dm_single(
    A: np.ndarray,
    b: np.ndarray,
    c: complex,
    output_cutoff: int,
    pnr_outcomes: list[PNR],
    leftover_mode: int = -1,
    first_hypercube_pnr: PNR | None = None,
    stable: bool = True,
) -> dict[PNR, np.ndarray]:
    """Single (non-batched) implementation of wormhole_1leftover_dm."""
    # Validate pnr_outcomes is not empty
    if not pnr_outcomes:
        raise ValueError("pnr_outcomes cannot be empty")

    # Reorder A and b to put leftover mode last in paired bra/ket ordering
    # MrMustard ordering: [bra_0, bra_1, ..., ket_0, ket_1, ...]
    # Wormhole ordering:  [bra_0, ket_0, bra_1, ket_1, ...] with leftover LAST
    A, b, num_modes = _reorder_for_wormhole(A, b, leftover_mode, is_dm=True)

    # Validate inputs
    first_hypercube_pnr = _validate_wormhole_inputs(
        pnr_outcomes, first_hypercube_pnr, num_modes - 1
    )

    # Create the visiting tree and initialize the first hypercube
    tree = create_visiting_tree(pnr_outcomes, first_hypercube_pnr)

    # Shape for hermite computation: origin + 2 in each measured dimension,
    # full cutoff for leftover mode
    hypercube_shape = (
        *(q + 2 for p in first_hypercube_pnr for q in (p, p)),
        output_cutoff + 1,
        output_cutoff + 1,
    )

    first_hypercube = vanilla(hypercube_shape, A, b, c, stable, None)

    # Extract just the last 2x2x...x2 hypercube (slicing off the initial computation)
    first_hypercube = first_hypercube[(slice(-2, None),) * 2 * (num_modes - 1)]
    first_hypercube = np.asarray(first_hypercube)

    # Store the first result: conditional DM at the first hypercube origin
    results: dict[PNR, np.ndarray] = {
        first_hypercube_pnr: first_hypercube[(0, 0) * (num_modes - 1)].copy()
    }

    # 4. Recursively compute all requested outcomes using branching wormhole
    _branching_wormhole(A, b, c, first_hypercube, tree, results)

    return results


def wormhole_1leftover_ket(
    A: np.ndarray,
    b: np.ndarray,
    c: complex | np.ndarray,
    output_cutoff: int,
    pnr_outcomes: list[PNR],
    leftover_mode: int = -1,
    first_hypercube_pnr: PNR | None = None,
    stable: bool = True,
) -> dict[PNR, np.ndarray]:
    r"""Compute conditional ket amplitudes for one leftover mode given PNR measurements.

    This is the Ket variant of the wormhole algorithm, which is significantly faster
    than the DM version because:
    - The Bargmann matrix A is M×M (vs 2M×2M for DM)
    - The hypercube has M-1 binary dimensions (vs 2(M-1) for DM)
    - Output is 1D vectors (vs 2D matrices for DM)

    Automatically dispatches to batched or non-batched implementation based on
    input dimensions:
    - If A has shape (n, n): single computation
    - If A has shape (batch, n, n): parallel computation over batch

    Args:
        A: Bargmann matrix. Shape (n_modes, n_modes) for single or
            (batch, n_modes, n_modes) for batched computation.
        b: Bargmann vector. Shape (n_modes,) or (batch, n_modes).
        c: Bargmann scalar normalization. Scalar or array of shape (batch,).
        output_cutoff: Maximum photon number for output ket amplitudes.
            The output shape will be (output_cutoff+1,).
        pnr_outcomes: List of PNR measurement outcomes to condition on.
            Each tuple has length = num_modes - 1 (measured modes only).
        leftover_mode: Which mode to keep unmeasured (default: last mode, -1).
        first_hypercube_pnr: Starting PNR point for the wormhole traversal.
            Default is (0, ..., 0).
        stable: Use numerically stable algorithm with renormalized Hermite polynomials.

    Returns:
        Dict mapping PNR outcomes to ket amplitude arrays.
        - Single: Each value has shape (output_cutoff+1,).
        - Batched: Each value has shape (batch, output_cutoff+1).

    Raises:
        ValueError: If pnr_outcomes is empty.
        ValueError: If first_hypercube_pnr exceeds any component of pnr_outcomes.

    Example:
        >>> # Single computation
        >>> A, b, c = ket.bargmann_triple()
        >>> results = wormhole_1leftover_ket(
        ...     A, b, c, output_cutoff=70, pnr_outcomes=[(1, 4)], leftover_mode=0
        ... )
        >>> # results[(1, 4)] has shape (71,)
        >>>
        >>> # Batched computation (100 kets in parallel)
        >>> A_batch = np.stack([A + noise for _ in range(100)])
        >>> results = wormhole_1leftover_ket(
        ...     A_batch, b_batch, c_batch, output_cutoff=70, pnr_outcomes=[(1, 4)]
        ... )
        >>> # results[(1, 4)] has shape (100, 71)
    """
    A = np.asarray(A)

    # Dispatch based on input dimensions
    if A.ndim == 2:
        return _wormhole_1leftover_ket_single(
            A, b, c, output_cutoff, pnr_outcomes, leftover_mode, first_hypercube_pnr, stable
        )
    if A.ndim == 3:
        return _wormhole_1leftover_ket_batched(
            A, b, c, output_cutoff, pnr_outcomes, leftover_mode, first_hypercube_pnr, stable
        )
    raise ValueError(f"A must have 2 or 3 dimensions, got {A.ndim}")


def _wormhole_1leftover_ket_single(
    A: np.ndarray,
    b: np.ndarray,
    c: complex,
    output_cutoff: int,
    pnr_outcomes: list[PNR],
    leftover_mode: int = -1,
    first_hypercube_pnr: PNR | None = None,
    stable: bool = True,
) -> dict[PNR, np.ndarray]:
    """Single (non-batched) implementation of wormhole_1leftover_ket."""
    # Validate pnr_outcomes is not empty
    if not pnr_outcomes:
        raise ValueError("pnr_outcomes cannot be empty")

    # Reorder A and b to put leftover mode last
    A, b, num_modes = _reorder_for_wormhole(A, b, leftover_mode, is_dm=False)

    # Validate inputs
    first_hypercube_pnr = _validate_wormhole_inputs(
        pnr_outcomes, first_hypercube_pnr, num_modes - 1
    )

    # Create the visiting tree and initialize the first hypercube
    tree = create_visiting_tree(pnr_outcomes, first_hypercube_pnr)

    # Shape: origin + 2 in each measured dimension, full cutoff for leftover
    hypercube_shape = (*(p + 2 for p in first_hypercube_pnr), output_cutoff + 1)

    first_hypercube = vanilla(hypercube_shape, A, b, c, stable, None)

    # Extract just the last 2x2x...x2 hypercube
    first_hypercube = first_hypercube[(slice(-2, None),) * (num_modes - 1)]
    first_hypercube = np.asarray(first_hypercube)

    # Store the first result
    results: dict[PNR, np.ndarray] = {
        first_hypercube_pnr: first_hypercube[(0,) * (num_modes - 1)].copy()
    }

    # Recursively compute all requested outcomes
    _branching_wormhole_ket(A, b, c, first_hypercube, tree, results)

    return results


def _branching_wormhole_ket(
    A: np.ndarray,
    b: np.ndarray,
    c: complex,
    hypercube: np.ndarray,
    tree: dict[tuple[PNR, int], dict],
    results: dict[PNR, np.ndarray],
) -> None:
    r"""Recursive branching wormhole traversal for Kets.

    Performs depth-first traversal of the visiting tree, advancing the hypercube
    along each branch and extracting conditional ket amplitudes at target PNR
    positions. Simpler than the DM version since each measured mode has only
    one Fock index (no bra/ket pairing required).

    The traversal works as follows:
    1. For each branch (origin_pnr, dimension) in the current tree level:
       a. Advance the hypercube by 1 step in the specified dimension
       b. Extract and store the conditional ket at the new PNR position
       c. Recursively process any subtree branches from this position

    Args:
        A: Bargmann matrix (num_modes × num_modes) in reordered form
            (measured modes first, leftover mode last)
        b: Bargmann vector (num_modes,) in reordered form
        c: Bargmann scalar (unused but kept for signature consistency with DM version)
        hypercube: Current hypercube array with shape (2, 2, ..., 2, cutoff+1).
            Binary dimensions track neighboring amplitudes for measured modes;
            last dimension holds the conditional ket for the leftover mode.
        tree: Visiting tree encoding branches to traverse. Structure is
            {(parent_pnr, dim): subtree, ...} where subtree has the same format.
        results: Dictionary to store computed kets, modified in place.
            Keys are PNR tuples, values are 1D amplitude arrays.
    """
    for (origin_pnr, k), next_tree in tree.items():
        # Convert PNR origin to full lattice coordinates (append 0 for leftover)
        origin = np.array([*origin_pnr, 0])

        # Advance along dimension k
        hypercube_out = _next_hypercube_1leftover_ket(A, b, hypercube, tuple(origin), k)

        # Extract and store the conditional ket at this PNR outcome
        new_pnr = tuple(p + 1 if i == k else p for i, p in enumerate(origin_pnr))
        results[new_pnr] = hypercube_out[(0,) * len(origin_pnr)].copy()

        # Recursively process subtree
        _branching_wormhole_ket(A, b, c, hypercube_out, next_tree, results)


@njit(cache=True)
def _next_hypercube_1leftover_ket(
    A: np.ndarray,
    b: np.ndarray,
    current_hypercube: np.ndarray,
    origin: tuple[int, ...],
    direction: int,
) -> np.ndarray:  # pragma: no cover
    r"""Advance hypercube by one step along given direction for Kets.

    This is the core lattice propagation for the Ket wormhole. It shifts the
    hypercube forward by one Fock index in the specified dimension, computing
    new amplitudes using the renormalized Hermite polynomial recurrence relation.

    The hypercube is a (2, 2, ..., 2, cutoff+1) array where:
    - Binary dimensions (2×2×...×2) track neighboring Fock amplitudes for
      measured modes (needed for the recurrence relation)
    - Last dimension (cutoff+1) holds the full conditional ket for the leftover mode

    Algorithm for each position on the front face (where hypercube_pivot[direction] == 1):
    1. Copy current front face → new back face (sliding window)
    2. Compute new front face using recurrence:
       val = b_eff * current + A_eff @ neighbors
       where neighbors include both measured mode neighbors (from hypercube)
       and leftover mode neighbors (from within the ket vector)

    Args:
        A: Bargmann matrix (num_modes × num_modes) in reordered form
        b: Bargmann vector (num_modes,) in reordered form
        current_hypercube: Current hypercube with shape (2, 2, ..., 2, cutoff+1)
        origin: Lattice coordinates of hypercube origin. Last element is the
            leftover mode coordinate (should be 0 since we track full ket).
        direction: Measured mode index to advance (0 to num_measured_modes-1)

    Returns:
        New hypercube with same shape, representing the lattice region shifted
        by +1 in the specified direction.
    """
    next_hypercube = np.zeros_like(current_hypercube)
    num_measured_dims = len(current_hypercube.shape) - 1  # Exclude leftover dim

    for hypercube_pivot in np.ndindex(current_hypercube.shape[:-1]):
        if hypercube_pivot[direction] == 1:
            # Front face: compute new values
            vec = current_hypercube[hypercube_pivot].copy()

            # Copy to back face of next hypercube
            next_hypercube[tuple_setitem(hypercube_pivot, direction, 0)] = vec

            # Compute recurrence coefficients at this pivot
            # Append 1 for the leftover mode position
            A_, b_ = Ab_at_pivot_directional(A, b, (*hypercube_pivot, 1), origin, direction)

            # Start with b contribution
            val = b_[direction] * vec

            # Leftover mode neighbor contribution (last dimension)
            val[1:] += A_[direction, -1] * SQRT[1 : vec.shape[0]] * vec[:-1]

            # Measured mode neighbor contributions
            for m in range(num_measured_dims):
                val += (
                    A_[direction, m]
                    * current_hypercube[tuple_setitem(hypercube_pivot, m, hypercube_pivot[m] - 1)]
                )

            next_hypercube[hypercube_pivot] = val

    return next_hypercube


def _branching_wormhole(
    A: np.ndarray,
    b: np.ndarray,
    c: complex,
    hypercube: np.ndarray,
    tree: dict[tuple[PNR, int], dict],
    results: dict[PNR, np.ndarray],
) -> None:
    r"""Recursive branching wormhole traversal for density matrices.

    Performs depth-first traversal of the visiting tree, advancing the hypercube
    along each branch and extracting conditional density matrices at target PNR
    positions. For DMs, each PNR step requires TWO lattice advances (bra and ket
    indices are paired), making this more complex than the Ket version.

    The traversal works as follows:
    1. For each branch (origin_pnr, k) in the current tree level:
       a. Advance hypercube in ket direction (index 2k+1)
       b. Advance hypercube in bra direction (index 2k)
       c. Extract and store the conditional DM at the new PNR position
       d. Recursively process any subtree branches from this position

    The hypercube uses paired bra/ket ordering: [bra_0, ket_0, bra_1, ket_1, ...].
    This means mode k's PNR value corresponds to indices 2k (bra) and 2k+1 (ket).

    Args:
        A: Bargmann matrix (2*num_modes × 2*num_modes) in wormhole ordering
            [bra_0, ket_0, bra_1, ket_1, ..., leftover_bra, leftover_ket]
        b: Bargmann vector (2*num_modes,) in wormhole ordering
        c: Bargmann scalar (unused but kept for signature consistency)
        hypercube: Current enriched hypercube with shape (2, 2, ..., 2, cutoff, cutoff).
            Binary dimensions track neighboring amplitudes for measured mode pairs;
            last two dimensions hold the conditional DM for the leftover mode.
        tree: Visiting tree encoding branches to traverse. Structure is
            {(parent_pnr, dim): subtree, ...} where subtree has the same format.
        results: Dictionary to store computed DMs, modified in place.
            Keys are PNR tuples, values are 2D density matrix arrays.
    """
    for (origin_pnr, k), next_tree in tree.items():
        # Convert PNR origin to full lattice coordinates
        # Shape: (bra_0, ket_0, bra_1, ket_1, ..., leftover_bra, leftover_ket)
        origin = np.array([q for p in origin_pnr for q in (p, p)] + [0, 0])

        # Advance along ket direction (2k+1), then bra direction (2k)
        # This moves to the next PNR value in dimension k
        hypercube_mid = _next_hypercube_1leftover(A, b, hypercube, tuple(origin), 2 * k + 1)
        origin[2 * k + 1] += 1
        hypercube_out = _next_hypercube_1leftover(A, b, hypercube_mid, tuple(origin), 2 * k)
        origin[2 * k] += 1

        # Extract and store the conditional density matrix at this PNR outcome
        new_pnr = tuple(origin[:-2:2])  # Extract PNR from bra coordinates
        results[new_pnr] = hypercube_out[(0, 0) * len(origin_pnr)].copy()

        # Recursively process subtree
        _branching_wormhole(A, b, c, hypercube_out, next_tree, results)


@njit(cache=True)
def _next_hypercube_1leftover(
    A: np.ndarray,
    b: np.ndarray,
    current_hypercube: np.ndarray,
    origin: tuple[int, ...],
    direction: int,
) -> np.ndarray:  # pragma: no cover
    r"""Advance hypercube by one step along given direction for density matrices.

    This is the core lattice propagation for the DM wormhole. It shifts the
    hypercube forward by one Fock index in the specified direction, computing
    new amplitudes using the renormalized Hermite polynomial recurrence relation.

    The hypercube is a (2, 2, ..., 2, cutoff, cutoff) array where:
    - Binary dimensions (2×2×...×2) track neighboring Fock amplitudes for
      measured mode bra/ket pairs (needed for the recurrence relation)
    - Last two dimensions (cutoff × cutoff) hold the full conditional DM
      for the leftover mode

    Algorithm for each position on the front face (where hypercube_pivot[direction] == 1):
    1. Copy current front face → new back face (sliding window)
    2. Compute new front face using recurrence:
       val = b_eff * current_mat + A_eff @ neighbors
       where neighbors include:
       - Measured mode neighbors (from hypercube binary dimensions)
       - Leftover bra neighbors (shifts along first DM axis)
       - Leftover ket neighbors (shifts along second DM axis)

    Note: Direction indices use paired bra/ket ordering. For measured mode k:
    - Index 2k is the bra direction
    - Index 2k+1 is the ket direction
    Incrementing PNR for mode k requires advancing in BOTH directions.

    Args:
        A: Bargmann matrix (2*num_modes × 2*num_modes) in wormhole ordering
        b: Bargmann vector (2*num_modes,) in wormhole ordering
        current_hypercube: Current enriched hypercube with shape
            (2, 2, ..., 2, cutoff, cutoff)
        origin: Lattice coordinates of the hypercube origin in paired ordering.
            Last two elements are leftover mode (bra, ket), should be (0, 0)
            since we track the full conditional DM.
        direction: Index to advance (0 to 2*M-1 where M is measured mode count).
            Even indices are bra directions, odd indices are ket directions.

    Returns:
        New hypercube with same shape, representing the lattice region shifted
        by +1 in the specified direction.
    """
    next_hypercube = np.zeros_like(current_hypercube)
    num_measured_dims = len(current_hypercube.shape) - 2  # Exclude leftover dims

    for hypercube_pivot in np.ndindex(current_hypercube.shape[:-2]):
        if hypercube_pivot[direction] == 1:
            # Front face: compute new values
            mat = current_hypercube[hypercube_pivot]

            # Copy to back face of next hypercube
            next_hypercube[tuple_setitem(hypercube_pivot, direction, 0)] = mat

            # Compute recurrence coefficients at this pivot
            # Append (1, 1) for the leftover mode position
            A_, b_ = Ab_at_pivot_directional(A, b, (*hypercube_pivot, 1, 1), origin, direction)

            # Start with b contribution
            val = b_[direction] * mat

            # Leftover mode bra neighbor contribution (second-to-last dimension)
            val[1:, :] += A_[direction, -2] * SQRT[1 : mat.shape[0]][:, None] * mat[:-1, :]

            # Leftover mode ket neighbor contribution (last dimension)
            val[:, 1:] += A_[direction, -1] * SQRT[1 : mat.shape[1]][None, :] * mat[:, :-1]

            # Measured mode neighbor contributions
            for m in range(num_measured_dims):
                val += (
                    A_[direction, m]
                    * current_hypercube[tuple_setitem(hypercube_pivot, m, hypercube_pivot[m] - 1)]
                )

            next_hypercube[hypercube_pivot] = val

    return next_hypercube


# =============================================================================
# BATCHED VERSIONS (Optimized: single prange over triples)
# =============================================================================


def _count_tree_nodes(tree: dict) -> int:
    """Count total nodes in a visiting tree."""
    count = len(tree)
    for subtree in tree.values():
        count += _count_tree_nodes(subtree)
    return count


def _flatten_tree_for_dm(
    tree: dict,
    num_measured_modes: int,
    pnr_targets: set[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Flatten the visiting tree into arrays suitable for Numba processing.

    For DM, each PNR step requires TWO lattice steps (ket then bra direction).
    This function performs a DFS traversal and records all steps.

    Args:
        tree: Visiting tree from create_visiting_tree.
        num_measured_modes: Number of measured modes.
        pnr_targets: Set of target PNR outcomes to collect.

    Returns:
        origins: Array of shape (num_steps, 2*num_measured_modes + 2) with lattice origins.
        directions: Array of shape (num_steps,) with direction indices.
        result_pnrs: List of PNR tuples in the order results are produced.
                     Length equals number of PNR steps (half of lattice steps).
    """
    num_nodes = _count_tree_nodes(tree)
    num_steps = 2 * num_nodes  # DM needs two lattice steps (ket + bra) per PNR result
    dim = 2 * num_measured_modes + 2

    if num_steps == 0:
        return np.zeros((0, dim), dtype=np.int64), np.zeros(0, dtype=np.int64), []

    origins = np.empty((num_steps, dim), dtype=np.int64)
    directions = np.empty(num_steps, dtype=np.int64)
    result_pnrs: list[tuple[int, ...]] = []
    step_idx = [0]

    def dfs(subtree: dict) -> None:
        for (origin_pnr, k), next_tree in subtree.items():
            origin = np.array([q for p in origin_pnr for q in (p, p)] + [0, 0])
            i = step_idx[0]

            # Step 1: Advance in ket direction (2k+1)
            origins[i] = origin
            directions[i] = 2 * k + 1

            # Step 2: Advance in bra direction (2k)
            origin[2 * k + 1] += 1
            origins[i + 1] = origin
            directions[i + 1] = 2 * k
            step_idx[0] += 2

            # Record the resulting PNR
            origin[2 * k] += 1
            result_pnrs.append(tuple(origin[:-2:2]))

            # Recurse into subtree
            dfs(next_tree)

    dfs(tree)
    return origins, directions, result_pnrs


def _flatten_tree_for_ket(
    tree: dict,
    num_measured_modes: int,
    pnr_targets: set[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Flatten the visiting tree into arrays suitable for Numba processing (Ket version).

    For Ket, each PNR step requires ONE lattice step.

    Args:
        tree: Visiting tree from create_visiting_tree.
        num_measured_modes: Number of measured modes.
        pnr_targets: Set of target PNR outcomes to collect.

    Returns:
        origins: Array of shape (num_steps, num_measured_modes + 1) with lattice origins.
        directions: Array of shape (num_steps,) with direction indices.
        result_pnrs: List of PNR tuples in the order results are produced.
    """
    num_nodes = _count_tree_nodes(tree)

    if num_nodes == 0:
        dim = num_measured_modes + 1
        return np.zeros((0, dim), dtype=np.int64), np.zeros(0, dtype=np.int64), []

    origins = np.empty((num_nodes, num_measured_modes + 1), dtype=np.int64)
    directions = np.empty(num_nodes, dtype=np.int64)
    result_pnrs: list[tuple[int, ...]] = []
    idx = [0]

    def dfs(subtree: dict) -> None:
        for (origin_pnr, k), next_tree in subtree.items():
            i = idx[0]
            origins[i] = [*origin_pnr, 0]
            directions[i] = k
            idx[0] += 1

            result_pnrs.append(tuple(p + 1 if j == k else p for j, p in enumerate(origin_pnr)))

            dfs(next_tree)

    dfs(tree)
    return origins, directions, result_pnrs


def _wormhole_1leftover_dm_batched(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    output_cutoff: int,
    pnr_outcomes: list[PNR],
    leftover_mode: int = -1,
    first_hypercube_pnr: PNR | None = None,
    stable: bool = True,
) -> dict[PNR, np.ndarray]:
    r"""Compute conditional density matrices for batched inputs.

    Optimized batched version that processes multiple (A, b, c) triples in
    parallel. The tree traversal is pre-computed once, then each triple
    independently follows the same step sequence in parallel.

    Args:
        A: Bargmann matrices with shape (batch, 2*n_modes, 2*n_modes).
        b: Bargmann vectors with shape (batch, 2*n_modes).
        c: Bargmann scalars with shape (batch,).
        output_cutoff: Maximum photon number for output density matrices.
        pnr_outcomes: List of PNR measurement outcomes (shared across batch).
        leftover_mode: Which mode to keep unmeasured (default: -1).
        first_hypercube_pnr: Starting PNR point for traversal.
        stable: Use numerically stable algorithm.

    Returns:
        Dict mapping PNR outcomes to batched density matrices.
        Each array has shape (batch, output_cutoff+1, output_cutoff+1).

    Example:
        >>> # Process 100 different states in parallel
        >>> A = np.random.randn(100, 4, 4) + 1j * np.random.randn(100, 4, 4)
        >>> b = np.random.randn(100, 4) + 1j * np.random.randn(100, 4)
        >>> c = np.ones(100, dtype=complex)
        >>> results = wormhole_1leftover_dm_batched(
        ...     A, b, c, output_cutoff=20, pnr_outcomes=[(5,)], leftover_mode=0
        ... )
        >>> # results[(5,)] has shape (100, 21, 21)
    """
    if not pnr_outcomes:
        raise ValueError("pnr_outcomes cannot be empty")

    A = np.asarray(A)
    b = np.asarray(b)
    c = np.asarray(c)
    batch_size = A.shape[0]

    # Reorder A and b for all batch elements
    A, b, num_modes = _reorder_for_wormhole(A, b, leftover_mode, is_dm=True)
    num_measured = num_modes - 1

    # Validate inputs
    first_hypercube_pnr = _validate_wormhole_inputs(pnr_outcomes, first_hypercube_pnr, num_measured)

    # Create the visiting tree (shared across batch) and flatten to step sequence
    tree = create_visiting_tree(pnr_outcomes, first_hypercube_pnr)
    origins, directions, result_pnrs = _flatten_tree_for_dm(tree, num_measured, set(pnr_outcomes))

    # Shape for hermite computation
    hypercube_shape = (
        *(q + 2 for p in first_hypercube_pnr for q in (p, p)),
        output_cutoff + 1,
        output_cutoff + 1,
    )

    # Compute initial hypercubes for all batch elements
    first_hypercube = vanilla_batched(hypercube_shape, A, b, c, stable, None)

    # Extract just the last 2x2x...x2 hypercube for each batch element
    slice_tuple = (slice(None),) + (slice(-2, None),) * 2 * num_measured
    first_hypercube = np.ascontiguousarray(first_hypercube[slice_tuple])

    # Allocate output array for all results: (num_pnr_results, batch, cutoff, cutoff)
    num_pnr_results = len(result_pnrs)
    cutoff_size = output_cutoff + 1
    all_results = np.zeros(
        (num_pnr_results, batch_size, cutoff_size, cutoff_size),
        dtype=first_hypercube.dtype,
    )

    # Process all triples in parallel, each following the same step sequence
    if num_pnr_results > 0:
        _process_all_triples_dm(A, b, first_hypercube, origins, directions, all_results)

    # Build result dictionary
    extract_tuple = (slice(None),) + (0, 0) * num_measured
    results: dict[PNR, np.ndarray] = {first_hypercube_pnr: first_hypercube[extract_tuple].copy()}
    for i, pnr in enumerate(result_pnrs):
        results[pnr] = all_results[i]

    return results


def _wormhole_1leftover_ket_batched(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    output_cutoff: int,
    pnr_outcomes: list[PNR],
    leftover_mode: int = -1,
    first_hypercube_pnr: PNR | None = None,
    stable: bool = True,
) -> dict[PNR, np.ndarray]:
    r"""Compute conditional ket amplitudes for batched inputs.

    Optimized batched version that processes multiple (A, b, c) triples in
    parallel. The tree traversal is pre-computed once, then each triple
    independently follows the same step sequence in parallel.

    Args:
        A: Bargmann matrices with shape (batch, n_modes, n_modes).
        b: Bargmann vectors with shape (batch, n_modes).
        c: Bargmann scalars with shape (batch,).
        output_cutoff: Maximum photon number for output ket amplitudes.
        pnr_outcomes: List of PNR measurement outcomes (shared across batch).
        leftover_mode: Which mode to keep unmeasured (default: -1).
        first_hypercube_pnr: Starting PNR point for traversal.
        stable: Use numerically stable algorithm.

    Returns:
        Dict mapping PNR outcomes to batched ket arrays.
        Each array has shape (batch, output_cutoff+1).

    Example:
        >>> # Process 100 different states in parallel
        >>> A = np.random.randn(100, 2, 2) + 1j * np.random.randn(100, 2, 2)
        >>> b = np.random.randn(100, 2) + 1j * np.random.randn(100, 2)
        >>> c = np.ones(100, dtype=complex)
        >>> results = wormhole_1leftover_ket_batched(
        ...     A, b, c, output_cutoff=20, pnr_outcomes=[(5,)], leftover_mode=0
        ... )
        >>> # results[(5,)] has shape (100, 21)
    """
    if not pnr_outcomes:
        raise ValueError("pnr_outcomes cannot be empty")

    A = np.asarray(A)
    b = np.asarray(b)
    c = np.asarray(c)
    batch_size = A.shape[0]

    # Reorder A and b for all batch elements
    A, b, num_modes = _reorder_for_wormhole(A, b, leftover_mode, is_dm=False)
    num_measured = num_modes - 1

    # Validate inputs
    first_hypercube_pnr = _validate_wormhole_inputs(pnr_outcomes, first_hypercube_pnr, num_measured)

    # Create the visiting tree (shared across batch) and flatten to step sequence
    tree = create_visiting_tree(pnr_outcomes, first_hypercube_pnr)
    origins, directions, result_pnrs = _flatten_tree_for_ket(tree, num_measured, set(pnr_outcomes))

    # Shape for hermite computation
    hypercube_shape = (*(p + 2 for p in first_hypercube_pnr), output_cutoff + 1)

    # Compute initial hypercubes for all batch elements
    first_hypercube = vanilla_batched(hypercube_shape, A, b, c, stable, None)

    # Extract just the last 2x2x...x2 hypercube for each batch element
    slice_tuple = (slice(None),) + (slice(-2, None),) * num_measured
    first_hypercube = np.ascontiguousarray(first_hypercube[slice_tuple])

    # Allocate output array for all results: (num_pnr_results, batch, cutoff)
    num_pnr_results = len(result_pnrs)
    cutoff_size = output_cutoff + 1
    all_results = np.zeros(
        (num_pnr_results, batch_size, cutoff_size),
        dtype=first_hypercube.dtype,
    )

    # Process all triples in parallel, each following the same step sequence
    if num_pnr_results > 0:
        _process_all_triples_ket(A, b, first_hypercube, origins, directions, all_results)

    # Build result dictionary
    extract_tuple = (slice(None),) + (0,) * num_measured
    results: dict[PNR, np.ndarray] = {first_hypercube_pnr: first_hypercube[extract_tuple].copy()}
    for i, pnr in enumerate(result_pnrs):
        results[pnr] = all_results[i]

    return results


@njit(cache=True)
def _process_single_dm(
    A: np.ndarray,
    b: np.ndarray,
    initial_hypercube: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    results_out: np.ndarray,
) -> None:  # pragma: no cover
    r"""Process all wormhole DM steps for a single (A, b) pair.

    Follows the pre-computed step sequence (flattened from the visiting tree),
    calling ``_next_hypercube_1leftover`` at each step and extracting density
    matrices at every second step (ket + bra pair = one PNR result).

    Args:
        A: Bargmann matrix (2*num_modes, 2*num_modes).
        b: Bargmann vector (2*num_modes,).
        initial_hypercube: Initial hypercube (2, 2, ..., cutoff, cutoff).
        origins: Step origins array (num_steps, lattice_dims).
        directions: Step directions array (num_steps,).
        results_out: Output array (num_pnr_results, cutoff, cutoff).
                     Modified in place.
    """
    hypercube = initial_hypercube.copy()
    cutoff = hypercube.shape[-1]
    pnr_result_idx = 0
    for step_idx in range(origins.shape[0]):
        hypercube = _next_hypercube_1leftover(
            A, b, hypercube, origins[step_idx], directions[step_idx]
        )
        # Every 2 steps completes a PNR result (ket step + bra step)
        if step_idx % 2 == 1:
            # Use flat indexing since tuple size must be compile-time constant
            result_slice = hypercube.ravel()[: cutoff * cutoff].reshape(cutoff, cutoff)
            results_out[pnr_result_idx] = result_slice
            pnr_result_idx += 1


@njit(parallel=True, cache=True)
def _process_all_triples_dm(
    A: np.ndarray,
    b: np.ndarray,
    initial_hypercube: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    all_results: np.ndarray,
) -> None:  # pragma: no cover
    r"""Process all (A, b) triples in parallel, each following the same step sequence.

    Parallelizes ``_process_single_dm`` over the batch dimension using prange.

    Args:
        A: Bargmann matrices (batch, 2*num_modes, 2*num_modes).
        b: Bargmann vectors (batch, 2*num_modes).
        initial_hypercube: Initial hypercubes (batch, 2, 2, ..., cutoff, cutoff).
        origins: Step origins array (num_steps, lattice_dims).
        directions: Step directions array (num_steps,).
        all_results: Output array (num_pnr_results, batch, cutoff, cutoff).
                     Modified in place.
    """
    for batch_idx in prange(A.shape[0]):
        _process_single_dm(
            A[batch_idx],
            b[batch_idx],
            initial_hypercube[batch_idx],
            origins,
            directions,
            all_results[:, batch_idx],
        )


@njit(cache=True)
def _process_single_ket(
    A: np.ndarray,
    b: np.ndarray,
    initial_hypercube: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    results_out: np.ndarray,
) -> None:  # pragma: no cover
    r"""Process all wormhole Ket steps for a single (A, b) pair.

    Follows the pre-computed step sequence (flattened from the visiting tree),
    calling ``_next_hypercube_1leftover_ket`` at each step. Each step produces
    one PNR result (unlike DM which needs two steps per result).

    Args:
        A: Bargmann matrix (num_modes, num_modes).
        b: Bargmann vector (num_modes,).
        initial_hypercube: Initial hypercube (2, 2, ..., cutoff).
        origins: Step origins array (num_steps, lattice_dims).
        directions: Step directions array (num_steps,).
        results_out: Output array (num_pnr_results, cutoff).
                     Modified in place.
    """
    hypercube = initial_hypercube.copy()
    cutoff = hypercube.shape[-1]
    for step_idx in range(origins.shape[0]):
        hypercube = _next_hypercube_1leftover_ket(
            A, b, hypercube, origins[step_idx], directions[step_idx]
        )
        # Each step produces one PNR result
        # Use flat indexing since tuple size must be compile-time constant
        result_slice = hypercube.ravel()[:cutoff]
        results_out[step_idx] = result_slice


@njit(parallel=True, cache=True)
def _process_all_triples_ket(
    A: np.ndarray,
    b: np.ndarray,
    initial_hypercube: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
    all_results: np.ndarray,
) -> None:  # pragma: no cover
    r"""Process all (A, b) triples in parallel for Kets.

    Parallelizes ``_process_single_ket`` over the batch dimension using prange.

    Args:
        A: Bargmann matrices (batch, num_modes, num_modes).
        b: Bargmann vectors (batch, num_modes).
        initial_hypercube: Initial hypercubes (batch, 2, 2, ..., cutoff).
        origins: Step origins array (num_steps, lattice_dims).
        directions: Step directions array (num_steps,).
        all_results: Output array (num_pnr_results, batch, cutoff).
                     Modified in place.
    """
    for batch_idx in prange(A.shape[0]):
        _process_single_ket(
            A[batch_idx],
            b[batch_idx],
            initial_hypercube[batch_idx],
            origins,
            directions,
            all_results[:, batch_idx],
        )

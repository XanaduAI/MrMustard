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

r"""Core wormhole computation primitives.

This module contains the fundamental operations for the wormhole algorithm:
- Ab_at_pivot_directional: Compute rescaled A and b at a given pivot using pinv

The key insight is that the Fock recurrence relation:
    G[n + e_k] = (1/sqrt(n_k+1)) * (b_k * G[n] + sum_j A_kj * sqrt(n_j) * G[n - e_j])

can be algebraically rearranged when some neighbors G[n - e_j] are outside
the current hypercube. The pinv (pseudo-inverse) handles potentially singular
systems that arise at boundary conditions.

CRITICAL: Do NOT replace pinv with solve. The backward matrix can be singular
at certain pivots (when multiple hypercube indices are 0), and pinv handles
this correctly by finding the minimum-norm solution.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def Ab_at_pivot_directional(
    A: np.ndarray,
    b: np.ndarray,
    hypercube_pivot: tuple[int, ...],
    origin: tuple[int, ...],
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compute effective A row and b scalar at a pivot for hypercube advancement.

    This is the core recurrence computation that enables wormhole to advance
    through the Fock lattice. It handles the algebraic rearrangement needed
    when some backward neighbors are outside the current hypercube.

    The computation uses position-dependent rescaling based on the renormalized
    Hermite polynomial recurrence relation, and uses pinv (pseudo-inverse) to
    handle potentially singular systems at boundary conditions.

    Args:
        A: Bargmann matrix (n x n), complex
        b: Bargmann vector (n,), complex
        hypercube_pivot: Position within current hypercube (binary coordinates 0 or 1)
        origin: Lattice coordinates of hypercube origin
        direction: Axis along which to advance (0 to n-1)

    Returns:
        A_row: Effective A coefficients for the recurrence (n,), complex
        b_scalar: Effective b coefficient for the recurrence, complex

    Note:
        The hypercube_pivot should include the payload dimensions (e.g., (1, 1)
        for the leftover mode's bra/ket position). This ensures sqrt factors
        are computed correctly at all positions.
    """
    full_pivot = np.asarray(origin) + np.asarray(hypercube_pivot)
    sqrt = np.sqrt(full_pivot)
    sqrt1 = np.sqrt(full_pivot + 1)

    # Rescale A and b according to the recurrence relation
    A_rescaled = A * sqrt[None, :] / sqrt1[:, None]
    b_rescaled = b / sqrt1

    # Initialize forward and backward neighbor matrices
    # backward_neighbors: coefficients for neighbors inside the hypercube
    # forward_neighbors: coefficients for neighbors outside (to be solved for)
    backward_neighbors = -A_rescaled.copy()
    forward_neighbors = -A_rescaled.copy()

    for i, p in enumerate(np.asarray(hypercube_pivot)):
        if p == 0 and i != direction:
            # At back of hypercube in non-stepping dimension: backward neighbor is OUTSIDE
            # (it would be at origin - 1 in this direction, which doesn't exist)
            forward_neighbors[:, i] = 0
            forward_neighbors[i, i] = 1
        else:
            # Either in stepping direction OR at front of hypercube:
            # backward neighbor is known (inside hypercube)
            backward_neighbors[:, i] = 0
            backward_neighbors[i, i] = 1

    # Use pinv (NOT solve) to handle potentially singular systems
    # This is critical for correctness at boundary conditions
    inv = np.linalg.pinv(backward_neighbors)
    return -(inv @ forward_neighbors), (inv @ b_rescaled)

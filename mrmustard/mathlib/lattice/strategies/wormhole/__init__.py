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

r"""Wormhole algorithm for computing conditional Fock representations.

The wormhole algorithm efficiently computes conditional Fock representations
of quantum states given PNR (photon number resolving) measurement outcomes
on a subset of modes.

The key insight is that instead of computing the full Fock tensor and slicing,
the wormhole "worms" through the lattice directly to the target outcomes,
computing only the values needed along the path.

Main entry points:
    wormhole_1leftover_dm: Compute conditional 1-mode DMs given PNR measurements
    wormhole_1leftover_ket: Compute conditional 1-mode ket amplitudes (faster)

Both functions automatically dispatch to optimized batched implementations when
given inputs with a batch dimension:
    - Single: A.shape = (n, n) → returns dict with arrays of shape (cutoff, ...)
    - Batched: A.shape = (batch, n, n) → returns dict with arrays of shape (batch, cutoff, ...)

Example:
    >>> from mrmustard.mathlib.lattice.strategies.wormhole import wormhole_1leftover_dm
    >>> # Single computation: conditional DM of mode 0 given PNR (10, 12) on modes 1, 2
    >>> results = wormhole_1leftover_dm(
    ...     A, b, c, output_cutoff=20,  pnr_outcomes=[(10, 12)], leftover_mode=0
    ... )
    >>> cond_dm = results[(10, 12)]  # Shape: (21, 21)

    >>> # Batched computation: 100 states in parallel
    >>> A_batch = np.stack([A for _ in range(100)])  # Shape: (100, n, n)
    >>> results = wormhole_1leftover_dm(
    ...     A_batch, b_batch, c_batch, output_cutoff=20, pnr_outcomes=[(10, 12)], leftover_mode=0
    ... )
    >>> cond_dm_batch = results[(10, 12)]  # Shape: (100, 21, 21)
"""

from .one_leftover import wormhole_1leftover_dm, wormhole_1leftover_ket

__all__ = ["wormhole_1leftover_dm", "wormhole_1leftover_ket"]

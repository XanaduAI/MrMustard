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

r"""Visiting tree construction for wormhole traversal.

This module provides tree construction and traversal utilities for efficiently
computing multiple PNR outcomes. The visiting tree encodes which lattice positions
need to be visited and in what order.

Tree Data Structure
-------------------
The visiting tree is a recursive nested dictionary with structure::

    VisitingTree = dict[tuple[PNR, int], VisitingTree]

Where each key is a (parent_pnr, dimension) tuple representing a step from
parent_pnr to parent_pnr + e_dimension (unit vector in that dimension).
Values are subtrees for further traversal from that position.

Example: To reach targets [(2, 0), (1, 1)] from origin (0, 0)::

    {
        ((0, 0), 0): {                # (0,0) → (1,0) in dim 0
            ((1, 0), 0): {},          # (1,0) → (2,0) in dim 0 [TARGET]
            ((1, 0), 1): {},          # (1,0) → (1,1) in dim 1 [TARGET]
        }
    }

This structure enables:
1. **Path-sharing**: Common prefixes computed once (both targets share (0,0)→(1,0))
2. **Branching**: Different targets branch at any point
3. **DFS traversal**: Only one path active at a time, minimizing memory

Optimization: Dimensions are traversed in order of DESCENDING PNR values.
This minimizes total computation because:
- The hypercube grows as we traverse more dimensions (2 → 2×2 → 2×2×2 → ...)
- Traversing larger PNR values first (when hypercube is thin) is cheaper
- Example: For PNR (10, 20), traversing 20 first then 10 costs ~80 ops,
  while traversing 10 first then 20 costs ~100 ops.
"""

from __future__ import annotations

# Type aliases
PNR = tuple[int, ...]
type VisitingTree = dict[tuple[PNR, int], "VisitingTree"]


def create_visiting_tree(
    targets: list[PNR],
    origin: PNR | None = None,
) -> VisitingTree:
    r"""Build a tree structure for visiting all target PNR outcomes.

    The tree encodes the minimal set of lattice steps needed to reach all targets
    from the origin. Each node represents a position in PNR space, and edges
    represent steps in a particular dimension.

    Optimization: Dimensions are traversed in order of DESCENDING target values.
    This minimizes computation because the hypercube grows with each traversed
    dimension, so it's cheaper to traverse large values first (thin hypercube).

    Args:
        targets: List of target PNR outcomes to visit (e.g., [(1, 4), (2, 3)])
        origin: Starting PNR position. Default is (0, ..., 0).

    Returns:
        VisitingTree: Recursive nested dictionary where keys are (parent_pnr, dimension)
        tuples representing edges, and values are subtrees for further traversal.
        See module docstring for detailed structure description.

    Example:
        >>> create_visiting_tree([(1, 0), (0, 1)])
        {((0, 0), 0): {}, ((0, 0), 1): {}}  # Two branches from origin
    """
    dimension = len(targets[0])
    origin = origin or (0,) * dimension

    # Build adjacency list: adj[node] = {dim: child_node}
    adj: dict[PNR, dict[int, PNR]] = {}

    for target in set(targets):
        current = list(origin)
        parent = origin

        # Sort dimensions by target value in DESCENDING order
        # This ensures we traverse larger PNR values first when hypercube is thin
        sorted_dims = sorted(range(dimension), key=lambda d: target[d] - origin[d], reverse=True)

        for dim in sorted_dims:
            steps_needed = target[dim] - current[dim]
            for _ in range(steps_needed):
                if current[dim] < target[dim]:
                    current[dim] += 1
                    child = tuple(current)

                    if parent not in adj:
                        adj[parent] = {}
                    adj[parent][dim] = child
                    parent = child

    return _build_branches_dict(origin, adj)


def _build_branches_dict(
    current_node: PNR,
    adj: dict[PNR, dict[int, PNR]],
) -> VisitingTree:
    r"""Recursively build VisitingTree structure from adjacency list.

    Transforms the flat adjacency representation into the nested tree structure
    expected by the branching wormhole traversal algorithm.

    Args:
        current_node: Current position in the tree (PNR tuple)
        adj: Adjacency list mapping nodes to {dimension: child_node}

    Returns:
        VisitingTree with (parent_pnr, dimension) keys and subtree values
    """
    branches: VisitingTree = {}

    if adj.get(current_node):
        for dim, child_node in adj[current_node].items():
            key = (current_node, dim)
            branches[key] = _build_branches_dict(child_node, adj)

    return branches

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

"""Tests for wormhole tree construction."""

import pytest

from mrmustard.mathlib.lattice.strategies.wormhole.tree import create_visiting_tree

PNR = tuple[int, ...]
VisitingTree = dict[tuple[PNR, int], "VisitingTree"]


def collect_reachable_nodes(tree: VisitingTree, origin: PNR) -> set[PNR]:
    """Collect all PNR nodes reachable from origin via the visiting tree.

    The visiting tree is a recursive dictionary structure where:
    - Keys are (parent_pnr, dimension) tuples representing edges
    - Values are subtrees with the same structure

    For example, a tree to reach (2, 1) from (0, 0) might look like:
        {
            ((0, 0), 0): {           # Step from (0,0) in dim 0 → (1, 0)
                ((1, 0), 0): {       # Step from (1,0) in dim 0 → (2, 0)
                    ((2, 0), 1): {}  # Step from (2,0) in dim 1 → (2, 1)
                }
            }
        }

    Args:
        tree: Visiting tree dictionary with (parent_pnr, dim) keys and subtree values
        origin: Starting PNR position to traverse from

    Returns:
        Set of all PNR tuples reachable by following branches from origin
    """
    nodes: set[PNR] = {origin}
    for (node, dim), subtree in tree.items():
        if node in nodes:
            child: PNR = tuple(n + 1 if i == dim else n for i, n in enumerate(node))
            nodes.add(child)
            nodes.update(collect_reachable_nodes(subtree, child))
    return nodes


class TestCreateVisitingTree:
    """Tests for the visiting tree construction."""

    def test_empty_tree_when_target_equals_origin(self):
        """No tree needed when already at target."""
        assert create_visiting_tree([(0, 0)]) == {}
        assert create_visiting_tree([(5, 5)], origin=(5, 5)) == {}

    @pytest.mark.parametrize(
        "targets,origin,expected_nodes",
        [
            # 1D linear chain
            ([(3,)], None, {(0,), (1,), (2,), (3,)}),
            # Branching targets (diverge immediately)
            ([(1, 0), (0, 1)], None, {(0, 0), (1, 0), (0, 1)}),
            # Targets sharing a prefix
            ([(2, 0), (2, 1)], None, {(0, 0), (1, 0), (2, 0), (2, 1)}),
            # Multiple 1D targets (chain)
            ([(1,), (2,), (3,)], None, {(0,), (1,), (2,), (3,)}),
            # Custom origin (traverses dim 0 first: 2 steps > 1 step)
            ([(3, 2)], (1, 1), {(1, 1), (2, 1), (3, 1), (3, 2)}),
        ],
    )
    def test_tree_visits_expected_nodes(self, targets, origin, expected_nodes):
        """Tree should contain exactly the nodes needed to reach all targets."""
        tree = create_visiting_tree(targets, origin)
        actual_origin = origin or (0,) * len(targets[0])
        reachable = collect_reachable_nodes(tree, actual_origin)
        assert reachable == expected_nodes

    @pytest.mark.parametrize(
        "targets",
        [
            [(2, 1)],
            [(2, 1), (2, 3), (1, 2)],
            [(3, 2), (1, 4), (5, 1)],
        ],
    )
    def test_all_targets_reachable(self, targets):
        """All specified targets must be reachable from origin."""
        tree = create_visiting_tree(targets)
        origin = (0,) * len(targets[0])
        reachable = collect_reachable_nodes(tree, origin)
        assert all(t in reachable for t in targets)

    @pytest.mark.parametrize(
        "targets,expected_first_dim",
        [
            ([(5, 20)], 1),  # 5 < , traverse dim 1 first
            ([(2, 100)], 1),  # 2 < 100, traverse dim 1 first
            ([(20, 5)], 0),  # 20 > 5, traverse dim 0 first
            ([(5, 50, 10)], 1),  # 50 > 10 > 5, traverse dim 1 first
        ],
    )
    def test_traverses_larger_dimensions_first(self, targets, expected_first_dim):
        """Optimization: traverse larger PNR values first (thinner hypercube)."""
        tree = create_visiting_tree(targets)
        first_key = next(iter(tree.keys()))
        _, first_dim = first_key
        assert first_dim == expected_first_dim

    def test_branching_creates_multiple_branches_from_origin(self):
        """Targets that diverge immediately should create multiple branches."""
        targets = [(1, 0), (0, 1)]
        tree = create_visiting_tree(targets)

        # Should have two branches from origin
        origin_branches = [k for k in tree if k[0] == (0, 0)]
        assert len(origin_branches) == 2
        dims = {k[1] for k in origin_branches}
        assert dims == {0, 1}

    def test_custom_origin_excludes_default_origin(self):
        """Tree with custom origin should not include (0, 0, ...)."""
        origin = (1, 1)
        targets = [(3, 2)]
        tree = create_visiting_tree(targets, origin)

        reachable = collect_reachable_nodes(tree, origin)
        assert origin in reachable
        assert (0, 0) not in reachable

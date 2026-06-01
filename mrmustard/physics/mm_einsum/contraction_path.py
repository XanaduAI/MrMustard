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

"""Utilities for validating and normalizing contraction paths (LA, SSA, UA)."""

from __future__ import annotations

import bisect
from typing import Literal

__all__ = [
    "normalize_path",
    "ua_to_linear",
    "validate_path",
]


def ua_to_linear(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert union assignment path to linear assignment path.

    Args:
        path: Union-assignment steps, each a pair ``(i, j)`` of operand indices to merge.

    Returns:
        The equivalent contraction path in linear-assignment (pair) form.
    """
    sets = [{i} for i in sorted({j for pair in path for j in pair})]
    steps = []
    for a, b in path:
        set1, set2 = [s for s in sets if a in s or b in s]
        i, j = sorted([sets.index(set1), sets.index(set2)])
        del sets[j], sets[i]  # must be deleted in order (higher index first)
        sets.append(set1 | set2)
        steps.append((i, j))
    return steps


def ssa_to_linear(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert static single assignment path to linear assignment path.

    Args:
        path: SSA-format steps. Each step is a tuple of indices (in SSA id space) that
            participate in that contraction.

    Returns:
        The equivalent path as binary linear-assignment pairs ``(i, j)``.
    """
    n = sum(map(len, path)) - len(path) + 1
    ids = list(range(n))
    output_path: list[tuple[int, int]] = []
    ssa = n
    for scon in path:
        con = sorted(bisect.bisect_left(ids, s) for s in scon)
        for j in reversed(con):
            ids.pop(j)
        ids.append(ssa)
        output_path.append((con[0], con[1]))
        ssa += 1
    return output_path


def normalize_path(
    path: list[tuple[int, int]] | None, path_type: Literal["LA", "SSA", "UA"]
) -> list[tuple[int, int]] | None:
    """Normalize contraction path to linear assignment of binary steps.

    Converts SSA or UA path formats to linear assignment (LA). Assumes the input
    path is well-formed.

    Args:
        path: The contraction path, or None.
        path_type: One of "LA", "SSA", or "UA".

    Returns:
        Normalized path in linear assignment format, or None if path is None.

    Raises:
        ValueError: If ``path_type`` is not ``"LA"``, ``"SSA"``, or ``"UA"``.
    """
    if path is None:
        return None
    if len(path) == 0:
        return []
    if path_type == "LA":
        return list(path)
    if path_type == "SSA":
        return ssa_to_linear(path)
    if path_type == "UA":
        return ua_to_linear(path)
    raise ValueError(f"Invalid path_type: {path_type!r}. Must be one of 'LA', 'SSA', 'UA'.")


def validate_path(
    steps_la: list[tuple[int, int]] | None,
    num_operands: int,
) -> None:
    """Validate that a path has exactly the steps needed for the given number of operands.

    For a contraction over ``num_operands`` operands, the path must have exactly
    ``max(num_operands - 1, 0)`` binary steps. When only one operand exists, the
    path must be empty or ``[(0,)]`` (unary/no-op).

    Args:
        steps_la: The path steps in linear assignment format.
        num_operands: The number of operands to be contracted.

    Raises:
        ValueError: If the path has too few or too many steps for the given
            number of operands.
    """
    if steps_la is None:
        return

    if num_operands == 1:
        if steps_la in ([], [(0,)]):
            return
        raise ValueError(
            "contraction_path must describe the full contraction: "
            f"expected [] or [(0,)] for the final unary contraction, got {steps_la}."
        )

    expected_steps = max(num_operands - 1, 0)
    if len(steps_la) != expected_steps:
        raise ValueError(
            "contraction_path must describe the full contraction: "
            f"expected {expected_steps} step(s) for {num_operands} operand(s), "
            f"got {len(steps_la)}."
        )

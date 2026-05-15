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
Utility functions for the math module.

This module contains pure functions for einsum string parsing and tensor shape manipulation.
These utilities support extended einsum notation with parenthesized index groups.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_collapsed_shape",
    "parse_einsum_output_with_parentheses",
    "strip_parentheses",
]


def strip_parentheses(equation: str) -> tuple[str, list[tuple[int, int]]]:
    """Strip parenthesized groups from an einsum equation string.

    Takes a full einsum equation (e.g. ``"ij,jk->(ik)"``) and returns the
    clean equation with parentheses removed, plus the group spans needed
    to collapse dimensions after contraction.

    If the equation has no ``->`` or no parentheses, returns it unchanged
    with an empty group list.

    Args:
        equation: Full einsum equation string, e.g. ``"ij,jk->h(ik)"``.

    Returns:
        Tuple of (clean equation, list of (start, end) group spans in the output).

    Raises:
        ValueError: If parentheses are nested, unmatched, or contain fewer than
            2 indices.
        ValueError: If ellipsis notation is combined with parenthesized groups.

    Examples:
        >>> strip_parentheses("ij,jk->(ik)")
        ('ij,jk->ik', [(0, 2)])

        >>> strip_parentheses("ijk->(ij)k")
        ('ijk->ijk', [(0, 2)])

        >>> strip_parentheses("ij,jk->ik")
        ('ij,jk->ik', [])

        >>> strip_parentheses("ij,jk")
        ('ij,jk', [])
    """
    if "->" not in equation:
        return equation, []

    lhs, output_string = equation.split("->", 1)

    if "(" not in output_string and ")" not in output_string:
        return equation, []

    output_clean, groups = parse_einsum_output_with_parentheses(output_string)

    if "..." in output_clean and groups:
        raise ValueError("Ellipsis notation with parenthesized groups is not supported")

    return lhs + "->" + output_clean, groups


def parse_einsum_output_with_parentheses(
    output_string: str,
) -> tuple[str, list[tuple[int, int]]]:
    """Parse output string, extracting indices and parenthesized group spans.

    Parses an einsum output string that may contain parentheses to indicate
    indices that should be flattened/vectorized together.

    Args:
        output_string: The output string from einsum equation (after "->").

    Returns:
        Tuple of (cleaned output string without parentheses, list of (start, end) group spans).
        The group spans indicate which consecutive output indices should be collapsed.

    Raises:
        ValueError: If parentheses are nested.
        ValueError: If parentheses are unmatched.
        ValueError: If a parenthesized group contains fewer than 2 indices.

    Examples:
        >>> parse_einsum_output_with_parentheses("h(ik)")
        ('hik', [(1, 3)])

        >>> parse_einsum_output_with_parentheses("(ab)c(de)")
        ('abcde', [(0, 2), (3, 5)])

        >>> parse_einsum_output_with_parentheses("abc")
        ('abc', [])
    """
    cleaned = ""
    groups = []
    in_group = False
    group_start = None

    for char in output_string:
        if char == "(":
            if in_group:
                raise ValueError("Nested parentheses not supported")
            in_group = True
            group_start = len(cleaned)
        elif char == ")":
            if not in_group:
                raise ValueError("Unmatched ')' in output")
            in_group = False
            # Groups must contain at least 2 indices
            if len(cleaned) - group_start < 2:
                raise ValueError("Parenthesized groups must contain at least 2 indices")
            groups.append((group_start, len(cleaned)))
        else:
            cleaned += char

    if in_group:
        raise ValueError("Unclosed '(' in output")

    return cleaned, groups


def compute_collapsed_shape(
    shape: tuple[int, ...],
    groups: list[tuple[int, int]],
) -> tuple[int, ...]:
    """Compute the new shape after collapsing grouped dimensions.

    Given a tensor shape and a list of index groups, computes the resulting shape
    after flattening each group of consecutive dimensions into a single dimension.

    Args:
        shape: The original tensor shape.
        groups: List of (start, end) tuples indicating which dimensions to collapse.
            Each group specifies a range of consecutive dimensions [start, end) that
            will be flattened into a single dimension.

    Returns:
        The new shape with grouped dimensions collapsed.

    Examples:
        >>> compute_collapsed_shape((2, 3, 4, 5), [(1, 3)])  # Collapse dims 1,2 (sizes 3,4)
        (2, 12, 5)

        >>> compute_collapsed_shape((2, 3, 4, 5, 6), [(0, 2), (3, 5)])  # Two groups
        (6, 4, 30)

        >>> compute_collapsed_shape((2, 3, 4), [])  # No groups
        (2, 3, 4)
    """
    if not groups:
        return shape

    shape_list = list(shape)
    # Sort groups by start position to process in order
    sorted_groups = sorted(groups, key=lambda x: x[0])

    # Compute new shape by collapsing groups
    new_shape = []
    cursor = 0
    for start, end in sorted_groups:
        # Add dimensions before this group
        new_shape.extend(shape_list[cursor:start])
        # Add collapsed group dimension
        group_size = int(np.prod(shape_list[start:end]))
        new_shape.append(group_size)
        cursor = end

    # Add remaining dimensions after last group
    new_shape.extend(shape_list[cursor:])

    return tuple(new_shape)

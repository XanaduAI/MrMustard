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

"""Implementation of the mm_einsum function."""

from __future__ import annotations
from mrmustard import math
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz.base import Ansatz


def mm_einsum(
    *args: list[PolyExpAnsatz | ArrayAnsatz | list[int | str]],
    contraction_order: list[tuple[int, ...]],
    shapes: dict[int, int],
):
    r"""Performs tensor contractions between multiple Ansatze using their indices.

    This function is analogous to numpy's einsum but specialized for MrMustard's Ansatze.
    It automatically determines the optimal contraction order and handles both continuous-variable
    (CV) and Fock-space representations.

    The arguments are passed as an alternating sequence of Ansatze and their corresponding index lists,
    followed by a final output index list. The index lists can contain strings and integers.
    The strings (which come before the integers) are used to label the batch dimensions.
    The integers are used to label the Hilbert space variables/indices.

    The rule is that equal strings or equal integers will be contracted together. The surviving (or
    copied) strings and integers have to appear in the output index list.

    Args:
        *args: Alternating sequence of Ansatze and their corresponding index lists,
            followed by a final output index list. The format should be:
            [ansatz1, indices1, ansatz2, indices2, ..., ansatzN, indicesN, output_indices]
        contraction_order: list of tuples of integers specifying the wires to contract together at
            each step.
        shapes: dict mapping Hilbert space indices to their Fock space dimensions.

    Returns:
        Ansatz: The resulting Ansatz after performing all the contractions.
    """
    all_strings = [s for idx in args[1:-2:2] for s in idx if isinstance(s, str)]
    strings_to_chars = _map_descriptive_strings_to_chars(all_strings)
    indices = {}
    batch_strings = {}
    ansatze = {}
    it = iter(args[:-1])
    for k, (ans, idx) in enumerate(zip(it, it)):
        indices[k] = [i for i in idx if isinstance(i, int)]
        batch_strings[k] = [strings_to_chars[s] for s in idx if isinstance(s, str)]
        ansatze[k] = ans
    output_indices = args[-1]

    for wires in contraction_order:
        a, b = [i for i, idx in indices.items() if set(wires).issubset(idx)]
        convert = type(ansatze[a]) is not type(ansatze[b])
        ansatz_a = to_fock(ansatze[a], [shapes[i] for i in indices[a]]) if convert else ansatze[a]
        ansatz_b = to_fock(ansatze[b], [shapes[i] for i in indices[b]]) if convert else ansatze[b]
        idx_a, idx_b, new_batch, new_indices = _prepare_idx(a, b, indices, batch_strings)
        einsum_string = _batch_einsum_string(batch_strings[a], batch_strings[b], new_batch)
        new_ansatz = ansatz_a.contract(ansatz_b, idx_a, idx_b, einsum_string)
        ansatze[a] = new_ansatz
        indices[a] = new_indices
        batch_strings[a] = new_batch
        del indices[b], batch_strings[b], ansatze[b]

    print(indices)

    if len(indices) > 1 or len(batch_strings) > 1 or len(ansatze) > 1:
        raise ValueError("More than one ansatz left after contraction.")
    result = ansatze.pop(0)
    if any(isinstance(i, int) for i in output_indices):
        final_indices = indices.pop(0)
        index_perm = [final_indices.index(i) for i in output_indices if isinstance(i, int)]
        result = result.reorder(index_perm)
    if any(isinstance(i, str) for i in output_indices):
        final_strings = batch_strings.pop(0)
        batch_perm = [final_strings.index(i) for i in output_indices if isinstance(i, str)]
        result = result.reorder_batch(batch_perm)
    return result


def _prepare_idx(a, b, indices, batch_strings):
    r"""Prepares the indices for the contraction."""
    common_batch = [i for i in batch_strings[a] if i in batch_strings[b]]
    common_indices = [i for i in indices[a] if i in indices[b]]
    new_batch = [i for i in batch_strings[a] + batch_strings[b] if i not in common_batch]
    new_indices = [i for i in indices[a] + indices[b] if i not in common_indices]
    idx_a = [indices[a].index(i) for i in common_indices]
    idx_b = [indices[b].index(i) for i in common_indices]
    return idx_a, idx_b, new_batch, new_indices


def _batch_einsum_string(
    indices_a: list[int | str],
    indices_b: list[int | str],
    remaining_indices: list[int | str],
) -> str:
    r"""Creates an einsum-style string for batch dimension contractions.

    Args:
        indices_a: List of indices for first ansatz
        indices_b: List of indices for second ansatz
        remaining_indices: List of indices that will remain after contraction

    Returns:
        str: Einsum-style string for batch contractions (e.g., "a,ab->b")
    """
    batch_a = [i for i in indices_a if isinstance(i, str)]
    batch_b = [i for i in indices_b if isinstance(i, str)]
    batch_out = [i for i in remaining_indices if isinstance(i, str)]

    a_str = "".join(batch_a)
    b_str = "".join(batch_b)
    out_str = "".join(batch_out)

    return f"{a_str},{b_str}->{out_str}"


def _map_descriptive_strings_to_chars(indices: list[list[int | str]]) -> dict[str, str]:
    r"""Creates a mapping from descriptive strings to single letters.

    Args:
        indices: List of lists of indices

    Returns:
        dict: Mapping from descriptive strings to single letters

    >>> _map_descriptive_strings_to_chars([["foo", "bar", "baz"]])
    {'foo': 'a', 'bar': 'b', 'baz': 'c'}
    """
    all_strings = [i for idx in indices for i in idx if isinstance(i, str)]
    return {string: chr(97 + i) for i, string in enumerate(all_strings)}


def to_fock(ansatz: Ansatz, shape: tuple[int, ...]) -> ArrayAnsatz:
    r"""Converts a poly exp ansatz to an array ansatz."""
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)
    array = math.hermite_renormalized_full_batch(*ansatz.triple, shape)
    return ArrayAnsatz(array, ansatz.batch_dims)

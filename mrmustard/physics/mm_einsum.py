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
from dataclasses import dataclass
from mrmustard import math
from mrmustard.physics import ansatz
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz.base import Ansatz


def _ints(seq: list[int | str]) -> list[int]:
    return [i for i in seq if isinstance(i, int)]


def _strings(seq: list[int | str]) -> list[str]:
    return [i for i in seq if isinstance(i, str)]


@dataclass
class Node:
    ansatz: PolyExpAnsatz | ArrayAnsatz
    batch_strings: list[str]
    wires_ints: list[int]
    shape: tuple[int, ...]
    id: int | tuple[int, ...]

    def __post_init__(self):
        if isinstance(self.id, int):
            self.id = (self.id,)


def mm_einsum(
    nodes: list[Node],
    output_batch_strings: list[str],
    output_wires_ints: list[int],
    contraction_order: list[tuple[int, int]],
) -> PolyExpAnsatz | ArrayAnsatz:
    r"""
    Contracts a network of Ansatze using a custom contraction order.
    This function is analogous to numpy's einsum but specialized for MrMustard's Ansatze.
    It handles both continuous-variable (CV) and Fock-space representations.

    Args:

    Returns:
        Ansatz: The resulting Ansatz after performing all the contractions.
    """
    for a, b in contraction_order:
        nodes = _perform_contraction(nodes, a, b)

    if len(nodes) > 1:
        raise ValueError("More than one ansatz left after contraction.")

    resuansatzodes[0]
    output_idx_str = _strings(output_batch_string)
    output_idx_int = _ints(output_index_ints)
    final_idx = output_index_ints
    if len(output_idx_int) > 1:
        final_idx = _ints(final_idx)
        index_perm = [final_idx.index(i) for i in output_idx_int]
        result = result.reorder(index_perm)
    if len(output_idx_str) > 1:
        final_strings = _strings(final_idx)
        batch_perm = [final_strings.index(i) for i in output_idx_str]
        result = result.reorder_batch(batch_perm)
    return result


def _perform_contraction(nodes, a, b):
    ansatz_a, ansatz_b = [ansatz for ansatz in nodes if a in ansatz.id or b in ansatz.id]
    if type(ansatz_a) is not type(ansatz_b):
        ansatz_a = to_fock(ansatz_a, ansatz_a._shape)
        ansatz_b = to_fock(ansatz_b, ansatz_b._shape)
    idx_a, idx_b, new_batch, new_int, new_shape = _prepare_idx(ansatz_a, ansatz_b)
    einsum_string = _batch_einsum_string(ansatz_a._batch_string, ansatz_b._batch_string, new_batch)
    new_ansatz = ansatz_a.contract(ansatz_b, idx_a, idx_b, einsum_string)
    new_ansatz._batch_string = new_batch
    new_ansatz._int_idx = new_int
    new_ansatz._shape = new_shape
    del ansatze[ta], ansatze[tb]
    ansatze[ta + tb] = new_ansatz
    return ansatze


def _prepare_idx(ansatz_a, ansatz_b):
    r"""Prepares the indices for the contraction."""
    common_batch = [i for i in ansatz_a._batch_string if i in ansatz_b._batch_string]
    common_int = [i for i in ansatz_a._int_idx if i in ansatz_b._int_idx]
    new_batch = [
        i for i in ansatz_a._batch_string + ansatz_b._batch_string if i not in common_batch
    ]
    new_int = [i for i in ansatz_a._int_idx + ansatz_b._int_idx if i not in common_int]
    idx_a = [ansatz_a._int_idx.index(i) for i in common_int]
    idx_b = [ansatz_b._int_idx.index(i) for i in common_int]
    new_shape = tuple(ansatz_a._shape[i] for i in idx_a) + tuple(ansatz_b._shape[i] for i in idx_b)
    return idx_a, idx_b, new_batch, new_int


def _batch_einsum_string(
    idx_a: list[int | str],
    idx_b: list[int | str],
    new_idx: list[int | str],
) -> str:
    r"""Creates an einsum-style string for batch dimension contractions.

    Args:
        indices_a: List of indices for first ansatz
        indices_b: List of indices for second ansatz
        remaining_indices: List of indices that will remain after contraction

    Returns:
        str: Einsum-style string for batch contractions (e.g., "a,ab->b")
    """
    a_str = "".join(_strings(idx_a))
    b_str = "".join(_strings(idx_b))
    out_str = "".join(_strings(new_idx))
    return f"{a_str},{b_str}->{out_str}"


def to_fock(ansatz: Ansatz, shape: tuple[int, ...]) -> ArrayAnsatz:
    r"""Converts a poly exp ansatz to an array ansatz."""
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)
    array = math.hermite_renormalized_full_batch(*ansatz.triple, shape)
    return ArrayAnsatz(array, ansatz.batch_dims)

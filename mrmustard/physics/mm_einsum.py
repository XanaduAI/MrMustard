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
from mrmustard.physics.triples import identity_Abc


def _ints(seq: list[int | str]) -> list[int]:
    return [i for i in seq if isinstance(i, int)]


def _strings(seq: list[int | str]) -> list[str]:
    return [i for i in seq if isinstance(i, str)]


def mm_einsum(
    *args: Ansatz | list[int | str],
    output: list[int | str],
    contraction_order: list[tuple[int, int]],
    fock_dims: dict[int, int],
) -> PolyExpAnsatz | ArrayAnsatz:
    r"""
    Contracts a network of Ansatze using a custom contraction order.
    This function is analogous to numpy's einsum but specialized for MrMustard's Ansatze.
    It handles both continuous-variable (PolyExpAnsatz) and Fock-space (ArrayAnsatz) Ansatze.

    Args:
        args: Alternating Ansatze and lists of indices.
        output: The output indices.
        contraction_order: The order in which to perform the contractions.
        fock_dims: The Fock space dimensions of the Hilbert space indices.

    Returns:
        Ansatz: The resulting Ansatz after performing all the contractions.
    """
    ansatze = {(i,): ansatz for i, ansatz in enumerate(args[::2])}
    indices = {(i,): index for i, index in enumerate(args[1::2])}
    all_batch_names = set(name for index in indices.values() for name in _strings(index))
    names_to_chars = {name: chr(97 + i) for i, name in enumerate(all_batch_names)}
    indices = {
        k: [names_to_chars[s] for s in _strings(index)] + _ints(index)
        for k, index in indices.items()
    }
    output = [names_to_chars[s] for s in _strings(output)] + _ints(output)

    for a, b in contraction_order:
        relevant_ansatze = {ids: ansatz for ids, ansatz in ansatze.items() if a in ids or b in ids}
        (id_a, ansatz_a), (id_b, ansatz_b) = relevant_ansatze.items()
        index_a, index_b = indices[id_a], indices[id_b]
        int_a, int_b = _ints(index_a), _ints(index_b)
        str_a, str_b = _strings(index_a), _strings(index_b)
        convert = type(ansatz_a) is not type(ansatz_b)
        ansatz_a = convert_ansatz(ansatz_a, [fock_dims[i] for i in int_a]) if convert else ansatz_a
        ansatz_b = convert_ansatz(ansatz_b, [fock_dims[i] for i in int_b]) if convert else ansatz_b

        common_int = [i for i in int_a if i in int_b]
        idx_a = [int_a.index(i) for i in common_int]
        idx_b = [int_b.index(i) for i in common_int]
        new_idx = [i for i in int_a + int_b if i not in common_int]

        other_indices = [index for ids, index in indices.items() if ids not in (id_a, id_b)]
        other_names = [name for idx in other_indices for name in _strings(idx)] + _strings(output)
        keep = {name for name in str_a + str_b if name in other_names}
        eins_str = "".join(str_a) + "," + "".join(str_b) + "->" + "".join(keep)

        new_ansatz = ansatz_a.contract(ansatz_b, idx_a, idx_b, eins_str)
        del ansatze[id_a]
        del ansatze[id_b]
        ansatze[id_a + id_b] = new_ansatz
        del indices[id_a]
        del indices[id_b]
        indices[id_a + id_b] = list(keep) + new_idx

    if len(ansatze) > 1:
        raise ValueError("More than one ansatz left after contraction.")

    # Get the only element from the dictionaries
    result = list(ansatze.values())[0]
    final_idx = list(indices.values())[0]
    output_idx_str = _strings(output)
    output_idx_int = _ints(output)

    if len(output_idx_str) > 1:
        final_idx_str = _strings(final_idx)
        batch_perm = [final_idx_str.index(i) for i in output_idx_str]
        result = result.reorder_batch(batch_perm)

    if len(output_idx_int) > 1:
        final_idx_int = _ints(final_idx)
        index_perm = [final_idx_int.index(i) for i in output_idx_int]
        result = result.reorder(index_perm)

    return result


def convert_ansatz(ansatz: Ansatz, shape: tuple[int, ...]) -> ArrayAnsatz:
    r"""
    Converts an ansatz to PolyExpAnsatz if the shape is all zeros.
    Otherwise, it converts the ansatz to an ArrayAnsatz with the given shape.

    Args:
        ansatz: The ansatz to convert.
        shape: The shape of the ArrayAnsatz.

    Returns:
        Ansatz: The converted Ansatz.
    """
    if all(shape[i] == 0 for i in range(len(shape))):
        return to_bargmann(ansatz)
    else:
        return to_fock(ansatz, shape)


def to_fock(ansatz: Ansatz, shape: tuple[int, ...]) -> ArrayAnsatz:
    r"""
    Converts a PolyExpAnsatz to an ArrayAnsatz.
    If the ansatz is already an ArrayAnsatz, it reduces the shape to the given shape.

    Args:
        ansatz: The ansatz to convert.
        shape: The shape of the ArrayAnsatz.

    Returns:
        ArrayAnsatz: The converted ArrayAnsatz.
    """
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)
    array = math.hermite_renormalized_full_batch(*ansatz.triple, shape)
    return ArrayAnsatz(array, ansatz.batch_dims)


def to_bargmann(ansatz: Ansatz) -> PolyExpAnsatz:
    r"""
    Converts an ArrayAnsatz to a PolyExpAnsatz.
    If the ansatz is already a PolyExpAnsatz, it returns the ansatz unchanged.

    Args:
        ansatz: The ansatz to convert.

    Returns:
        PolyExpAnsatz: The converted PolyExpAnsatz.
    """
    if isinstance(ansatz, PolyExpAnsatz):
        return ansatz
    try:
        A, b, c = ansatz._original_abc_data
    except (TypeError, AttributeError):
        # TODO: update identity_Abc when it supports batching
        A, b, _ = identity_Abc(ansatz.core_dims)
        A = math.broadcast_to(A, ansatz.batch_shape + (2 * ansatz.core_dims, 2 * ansatz.core_dims))
        b = math.broadcast_to(b, ansatz.batch_shape + (2 * ansatz.core_dims,))
        c = ansatz.array
    return PolyExpAnsatz(A, b, c)

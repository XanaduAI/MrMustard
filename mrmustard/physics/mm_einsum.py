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
from numpy.typing import ArrayLike


def _ints(seq: list[int | str]) -> list[int]:
    return [i for i in seq if isinstance(i, int)]


def _strings(seq: list[int | str]) -> list[str]:
    return [i for i in seq if isinstance(i, str)]


def mm_einsum(
    *args: Ansatz | list[int | str],
    output: list[int | str],
    contraction_order: list[tuple[int, int]],
    fock_dims: dict[int, int],
) -> PolyExpAnsatz | ArrayAnsatz | ArrayLike:
    r"""
    This function contracts a network of Ansatze using a custom contraction order and a dictionary
    of Fock space dimensions. If a dimension is 0 its ansatz is converted to PolyExpAnsatz.

    Args:
        args: Alternating Ansatze and lists of indices.
        output: The output indices.
        contraction_order: The order in which to perform the contractions.
        fock_dims: The Fock space dimensions of the Hilbert space indices.

    Returns:
        Ansatz: The resulting Ansatz after performing all the contractions.
    """
    # --- prepare ansatz and indices, convert names to characters ---
    ansatze = {(i,): ansatz for i, ansatz in enumerate(args[::2])}
    all_batch_names = set(name for index in args[1::2] for name in _strings(index))
    names_to_chars = {name: chr(97 + i) for i, name in enumerate(all_batch_names)}
    indices = {
        (i,): [names_to_chars[s] for s in _strings(index)] + _ints(index)
        for i, index in enumerate(args[1::2])
    }
    output = [names_to_chars[s] for s in _strings(output)] + _ints(output)

    # --- perform contractions ---
    for a, b in contraction_order:
        relevant_ansatze = {ids: ansatz for ids, ansatz in ansatze.items() if a in ids or b in ids}
        (id_a, ansatz_a), (id_b, ansatz_b) = relevant_ansatze.items()
        ints_a, ints_b = _ints(indices[id_a]), _ints(indices[id_b])
        convert = type(ansatz_a) is not type(ansatz_b)
        ansatz_a = convert_ansatz(ansatz_a, [fock_dims[i] for i in ints_a]) if convert else ansatz_a
        ansatz_b = convert_ansatz(ansatz_b, [fock_dims[i] for i in ints_b]) if convert else ansatz_b
        idx_out = prepare_idx_out(indices, id_a, id_b, output)
        print(indices[id_a], indices[id_b], idx_out)
        ansatze[id_a + id_b] = ansatz_a.contract(ansatz_b, indices[id_a], indices[id_b], idx_out)
        indices[id_a + id_b] = idx_out
        del ansatze[id_a], ansatze[id_b]
        del indices[id_a], indices[id_b]

    if len(ansatze) > 1:
        raise ValueError("More than one ansatz left after contraction.")

    # --- reorder the output ---
    result = list(ansatze.values())[0]
    final_idx = list(indices.values())[0]

    if len(output_idx_str := _strings(output)) > 1:
        final_idx_str = _strings(final_idx)
        batch_perm = [final_idx_str.index(i) for i in output_idx_str]
        result = result.reorder_batch(batch_perm)

    if len(output_idx_int := _ints(output)) > 1:
        final_idx_int = _ints(final_idx)
        index_perm = [final_idx_int.index(i) for i in output_idx_int]
        result = result.reorder(index_perm)

    return result


def prepare_idx_out(indices, id_a, id_b, output):
    r"""
    Prepares the index of the output of the contraction of two ansatze.
    """
    other_indices = [index for ids, index in indices.items() if ids not in (id_a, id_b)]
    other_chars = {char for idx in other_indices for char in _strings(idx)} | set(_strings(output))
    other_ints = {i for idx in other_indices for i in _ints(idx)} | set(_ints(output))

    idx_out = []
    for char in _strings(indices[id_a]) + _strings(indices[id_b]):
        if char in other_chars and char not in idx_out:
            idx_out.append(char)
    for i in _ints(indices[id_a]) + _ints(indices[id_b]):
        if i in other_ints and i not in idx_out:
            idx_out.append(i)
    return idx_out


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


def to_fock(ansatz: Ansatz, shape: tuple[int, ...], stable: bool = False) -> ArrayAnsatz:
    r"""
    Converts a PolyExpAnsatz to an ArrayAnsatz.
    If the ansatz is already an ArrayAnsatz, it reduces the shape to the given shape.

    Args:
        ansatz: The ansatz to convert.
        shape: The shape of the ArrayAnsatz.
        stable: Whether to use the stable version of the hermite_renormalized function.

    Returns:
        ArrayAnsatz: The converted ArrayAnsatz.
    """
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)
    array = math.hermite_renormalized(*ansatz.triple, shape, stable)
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

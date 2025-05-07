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
    Contracts a network of Ansatze according to a custom contraction order, supporting both Fock
    and Bargmann representations, batch dimensions, and named indices.

    This function generalizes the concept of Einstein summation (einsum) to quantum states and
    operators, allowing for flexible contraction of complex tensor networks in quantum optics.

    Args:
        *args: Alternating sequence of Ansatz objects and their associated index lists.
            Each Ansatz is followed by a list of indices (strings for batch axes, integers for Hilbert space indices).
        output (list[int | str]): The indices (batch names and Hilbert space indices) to keep in the output.
        contraction_order (list[tuple[int, int]]): The order in which to contract the Ansatze,
            specified as pairs of their positions in the input.
        fock_dims (dict[int, int]): Mapping from Hilbert space indices (int) to Fock space dimensions.
            If a dimension is 0, the corresponding Ansatz is converted to Bargmann (PolyExpAnsatz) form.

    Returns:
        PolyExpAnsatz | ArrayAnsatz | ArrayLike: The contracted Ansatz or array, depending on the output indices and Fock dimensions.

    Example:
        >>> from mrmustard.lab_dev import Ket, BSgate
        >>> from mrmustard.physics.mm_einsum import mm_einsum
        >>> # Prepare two single-mode states and a beamsplitter
        >>> ket0 = Ket.random([0])
        >>> ket1 = Ket.random([1])
        >>> bs = BSgate((0, 1), theta=0.5, phi=0.3)
        >>> # Contract the network: (ket0 & ket1) >> BS
        >>> result = mm_einsum(
        ...     ket0.ansatz, [0],
        ...     ket1.ansatz, [1],
        ...     bs.ansatz, [2, 3, 0, 1],
        ...     output=[2, 3],
        ...     contraction_order=[(0, 2), (0, 1)],
        ...     fock_dims={0: 20, 1: 20, 2: 10, 3: 10},
        ... )
        >>> assert isinstance(result, ArrayAnsatz)
        >>> assert result.array.shape == (10, 10)

    Notes:
        - Batch indices (strings) allow for batched contraction over multiple states/operators.
        - If the Fock dimensions of the shared indices are set to 0, the result is in Bargmann (PolyExpAnsatz) form.
        - The function raises ValueError if the Fock dimensions of the shared indices are mixed.

    """
    # --- prepare ansatz and indices, convert names to characters ---
    ansatze = list(args[::2])
    all_batch_names = set(name for index in args[1::2] for name in _strings(index))
    names_to_chars = {name: chr(97 + i) for i, name in enumerate(all_batch_names)}
    indices = [[names_to_chars[s] for s in _strings(index)] + _ints(index) for index in args[1::2]]
    output = [names_to_chars[s] for s in _strings(output)] + _ints(output)

    # --- perform contractions ---
    for a, b in contraction_order:
        ansatz_a, ansatz_b = ansatze[a], ansatze[b]
        core_idx_a, core_idx_b = _ints(indices[a]), _ints(indices[b])
        common_core_idx = set(core_idx_a) & set(core_idx_b)
        common_shape = [fock_dims[i] for i in common_core_idx]
        force_bargmann = all(s == 0 for s in common_shape)
        force_fock = not any(s == 0 for s in common_shape)
        if force_bargmann:
            ansatz_a = to_bargmann(ansatz_a)
            ansatz_b = to_bargmann(ansatz_b)
        elif force_fock:
            ansatz_a = to_fock(ansatz_a, [fock_dims[i] for i in core_idx_a])
            ansatz_b = to_fock(ansatz_b, [fock_dims[i] for i in core_idx_b])
        else:
            raise ValueError(f"Shared indices of mixed type ({common_shape}) for ({a}, {b}).")
        idx_out = prepare_idx_out(indices, a, b, output)
        ansatze[a] = ansatz_a.contract(ansatz_b, indices[a], indices[b], idx_out)
        indices[a] = idx_out
        ansatze.pop(b)
        indices.pop(b)

    if len(ansatze) > 1:
        raise ValueError("More than one ansatz left after contraction.")

    # --- reorder the output ---
    result = ansatze[0]
    final_idx = indices[0]

    if len(output_idx_str := _strings(output)) > 1:
        final_idx_str = _strings(final_idx)
        batch_perm = [final_idx_str.index(i) for i in output_idx_str]
        result = result.reorder_batch(batch_perm)

    if len(output_idx_int := _ints(output)) > 1:
        final_idx_int = _ints(final_idx)
        index_perm = [final_idx_int.index(i) for i in output_idx_int]
        result = result.reorder(index_perm)

    return result


def prepare_idx_out(
    indices: dict[int, list[int | str]], a: int, b: int, output: list[int | str]
) -> list[int | str]:
    r"""
    Prepares the index of the output of the contraction of two ansatze.
    """
    other_indices = [index for i, index in enumerate(indices) if i not in (a, b)]
    other_chars = {char for idx in other_indices for char in _strings(idx)} | set(_strings(output))
    other_core_idxs = {i for idx in other_indices for i in _ints(idx)} | set(_ints(output))

    idx_out = []
    for char in _strings(indices[a]) + _strings(indices[b]):  # first add batch indices
        if char in other_chars and char not in idx_out:
            idx_out.append(char)
    for i in _ints(indices[a]) + _ints(indices[b]):  # then add core indices
        if i in other_core_idxs and i not in idx_out:
            idx_out.append(i)
    return idx_out


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
    if 0 in shape:
        raise ValueError("Fock space dimension is 0.")
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)
    array = math.hermite_renormalized(*ansatz.triple, shape, stable)
    if ansatz._lin_sup:
        array = math.sum(array, axis=ansatz.batch_dims)
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

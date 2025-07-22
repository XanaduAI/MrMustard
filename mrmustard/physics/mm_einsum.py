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

from typing import Literal

from numpy.typing import ArrayLike
from opt_einsum.paths import ssa_to_linear

from mrmustard import math
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz.base import Ansatz
from mrmustard.physics.triples import identity_Abc


def _ints(seq: list[int | str]) -> list[int]:
    return [i for i in seq if isinstance(i, int)]


def _strings(seq: list[int | str]) -> list[str]:
    return [i for i in seq if isinstance(i, str)]


def ua_to_linear(ua: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert a path with union assignment ids to a path with recycled linear ids."""
    sets = [{i} for i in sorted({j for pair in ua for j in pair})]
    path = []
    for a, b in ua:
        to_union = [s for s in sets if a in s or b in s]
        a_, b_ = sorted([sets.index(u) for u in to_union])
        sets.pop(b_), sets.pop(a_)
        sets.append(to_union[0] | to_union[1])
        path.append((a_, b_))
    return path


def mm_einsum(
    *args: Ansatz | list[int | str],
    output: list[int | str],
    fock_dims: dict[int, int],
    contraction_path: list[tuple[int, int]],
    path_type: Literal["SSA", "LA", "UA"] = "LA",
) -> PolyExpAnsatz | ArrayAnsatz | ArrayLike:
    r"""
    Contracts a network of Ansatze according to a custom contraction order, supporting both Fock
    and Bargmann representations, batch dimensions, and named indices.

    This function generalizes the concept of Einstein summation (einsum) to quantum states and
    operators, allowing for flexible contraction of complex tensor networks in quantum optics.

    The path_type argument controls the style in which the contraction path is expressed.
    - SSA uses a dictionary to keep track of the contracted ansatze, so that there can be static single
      assignment to each ansatz. Every new result is added to the dictionary with a new key.
      If [a0,a1,a2] is the list of ansatze we compute a2 @ (a0 @ a1) with the contraction order [(0,1), (0,2)].
    - LA pops the contrated ansatze from the list and adds the result to the end.
      If [a0,a1,a2] is the list of ansatze, we compute a2 @ (a0 @ a1) with the contraction order [(0,1), (0,1)].
    - UA (union assignment) is a variant of SSA where we can refer to an intermediate result by any of the indices of the
      ansatze that participated in its computation.
      If [a0,a1,a2] is the list of ansatze we compute a2 @ (a0 @ a1) with the contraction order [(0,1), (2,0)] or [(0,1), (2,1)].

    Args:
        *args: Alternating sequence of Ansatz objects and their associated index lists.
            Each Ansatz is followed by a list of indices (strings for batch axes, integers for Hilbert space indices).
        output (list[int | str]): The indices (batch names and Hilbert space indices) to keep in the output.
        contraction_path (list[tuple[int, int]]): The order in which to contract the Ansatze,
            specified as pairs of their positions in the input.
        fock_dims (dict[int, int | None]): Mapping from Hilbert space indices (int) to Fock space dimensions.
            If a dimension is 0, the *entire* corresponding Ansatz is converted to Bargmann (PolyExpAnsatz) form.
            If a key is not present, the dimension is taken from the Ansatz.
        path_type (str): Single Static Assigment ("SSA"), Linear Assignment ("LA"), or Union Assignment ("UA").
            Default is "LA".

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
        ...     contraction_path=[(0, 2), (0, 1)],
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
    all_batch_names = {name for index in args[1::2] for name in _strings(index)}
    names_to_chars = {name: chr(97 + i) for i, name in enumerate(all_batch_names)}
    indices = [[names_to_chars[s] for s in _strings(index)] + _ints(index) for index in args[1::2]]
    output = [names_to_chars[s] for s in _strings(output)] + _ints(output)

    # --- perform contractions ---
    if path_type == "SSA":
        contraction_path = ssa_to_linear(contraction_path)
    elif path_type == "UA":
        contraction_path = ua_to_linear(contraction_path)
    for a, b in contraction_path:
        core_idx_a, core_idx_b = _ints(indices[a]), _ints(indices[b])
        common_core_idx = set(core_idx_a) & set(core_idx_b)
        force_bargmann = all(fock_dims.get(i, 1) == 0 for i in common_core_idx)
        force_fock = all(fock_dims.get(i, 1) != 0 for i in common_core_idx)
        if force_bargmann:
            ansatz_a = to_bargmann(ansatze[a])
            ansatz_b = to_bargmann(ansatze[b])
        elif force_fock:
            shape_a, shape_b = get_shapes(ansatze[a], ansatze[b], core_idx_a, core_idx_b, fock_dims)
            ansatz_a = to_fock(ansatze[a], shape_a)
            ansatz_b = to_fock(ansatze[b], shape_b)
        else:
            raise ValueError(f"Attempted contraction of {a} and {b} with mixed-type indices.")
        idx_out = prepare_idx_out(indices, a, b, output)
        result = ansatz_a.contract(ansatz_b, indices[a], indices[b], idx_out)
        a_sorted, b_sorted = sorted((a, b))
        ansatze.pop(b_sorted), ansatze.pop(a_sorted), indices.pop(b_sorted), indices.pop(a_sorted)
        ansatze.append(result)
        indices.append(idx_out)

    # --- reorder and convert the output ---
    if len(ansatze) > 1:
        raise ValueError("More than one ansatz left after contraction.")

    result = ansatze[0]
    final_idx_str, final_idx_int = _strings(indices[0]), _ints(indices[0])
    output_idx_str, output_idx_int = _strings(output), _ints(output)

    if len(output_idx_str) > 1:
        result = result.reorder_batch([final_idx_str.index(i) for i in output_idx_str])
    if len(output_idx_int) > 1:
        result = result.reorder([final_idx_int.index(i) for i in output_idx_int])
    if final_idx_int and all(fock_dims.get(i, 1) == 0 for i in final_idx_int):
        result = to_bargmann(result)
    return result


SHAPE = tuple[int, ...]


def get_shapes(
    ansatz_a: Ansatz,
    ansatz_b: Ansatz,
    core_idx_a: list[int],
    core_idx_b: list[int],
    fock_dims: dict[int, int],
) -> tuple[SHAPE, SHAPE]:
    r"""
    Gets Fock shapes for two ansatze. If the Fock dimension is not set, it is inferred from the ansatz.

    Args:
        ansatz_a: The first ansatz.
        ansatz_b: The second ansatz.
        core_idx_a: The core indices of the first ansatz.
        core_idx_b: The core indices of the second ansatz.
        fock_dims: The Fock dimensions of the indices.

    Returns:
        tuple[SHAPE, SHAPE]: The Fock shapes of the two ansatze.
    """

    def get_shape_for_idx(
        idx: int,
        ansatz: Ansatz,
        core_idx: list[int],
        fock_dims: dict[int, int],
    ) -> int:
        if idx in fock_dims:
            return fock_dims[idx]
        if not isinstance(ansatz, ArrayAnsatz):
            raise ValueError(f"Fock dimension of index {idx} is not set.")
        return ansatz.core_shape[core_idx.index(idx)]

    def get_common_shape(
        idx: int,
        ansatz_a: Ansatz,
        ansatz_b: Ansatz,
        core_idx_a: list[int],
        core_idx_b: list[int],
        fock_dims: dict[int, int],
    ) -> int:
        if idx in fock_dims:
            return fock_dims[idx]
        dim = 1_000_000_000
        if isinstance(ansatz_a, ArrayAnsatz):
            dim = min(dim, ansatz_a.core_shape[core_idx_a.index(idx)])
        if isinstance(ansatz_b, ArrayAnsatz):
            dim = min(dim, ansatz_b.core_shape[core_idx_b.index(idx)])
        return dim

    common_core_idx = set(core_idx_a) & set(core_idx_b)
    leftover_a = set(core_idx_a) - common_core_idx
    leftover_b = set(core_idx_b) - common_core_idx

    common_shape = {
        i: get_common_shape(i, ansatz_a, ansatz_b, core_idx_a, core_idx_b, fock_dims)
        for i in common_core_idx
    }

    shape_a = {
        **common_shape,
        **{i: get_shape_for_idx(i, ansatz_a, core_idx_a, fock_dims) for i in leftover_a},
    }

    shape_b = {
        **common_shape,
        **{i: get_shape_for_idx(i, ansatz_b, core_idx_b, fock_dims) for i in leftover_b},
    }

    return tuple(shape_a[i] for i in core_idx_a), tuple(shape_b[i] for i in core_idx_b)


def prepare_idx_out(
    indices: dict[int, list[int | str]],
    a: int,
    b: int,
    output: list[int | str],
) -> list[int | str]:
    r"""
    Prepares the index of the output of the contraction of two ansatze.

    Args:
        indices: The indices of the ansatze.
        a: The index of the first ansatz.
        b: The index of the second ansatz.
        output: The indices of the output.

    Returns:
        list[int | str]: The indices of the output.
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
        A = math.broadcast_to(A, (*ansatz.batch_shape, 2 * ansatz.core_dims, 2 * ansatz.core_dims))
        b = math.broadcast_to(b, (*ansatz.batch_shape, 2 * ansatz.core_dims))
        c = ansatz.array
    return PolyExpAnsatz(A, b, c)

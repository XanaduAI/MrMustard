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


def process_ansatz_with_promotion(ansatz, batch_idx, core_idx, fock_dims):
    """
    Handle core-to-batch promotion when strings appear in core_indices.

    Args:
        ansatz: The ansatz to potentially modify.
        batch_idx: List of batch index names.
        core_idx: List of core indices (ints or strings).
        fock_dims: Dictionary mapping all indices to their dimensions.

    Returns:
        tuple: (modified_ansatz, updated_batch_idx, updated_core_idx)
    """
    promoted_strings = _strings(core_idx)

    if promoted_strings:
        shape = [fock_dims[idx] for idx in core_idx]
        ansatz = to_fock(ansatz, tuple(shape))
        promoted_core_indices = [i for i, idx in enumerate(core_idx) if isinstance(idx, str)]
        ansatz = ansatz.promote_core_to_batch(promoted_core_indices)
        batch_idx = batch_idx + promoted_strings
        core_idx = _ints(core_idx)

    return ansatz, batch_idx, core_idx


def update_fock_dims_from_indices(
    indices: list[int | str],
    shapes: tuple[int, ...],
    fock_dims: dict[int | str, int],
    ansatz_num: int,
) -> None:
    # Handle cases where shapes is longer than indices (e.g., linear superposition with internal batch dims)
    if len(indices) < len(shapes):
        # Take only the shapes corresponding to the provided indices (from the end)
        shapes = shapes[-len(indices) :] if indices else ()

    assert len(indices) == len(shapes)
    for j, idx in enumerate(indices):
        if idx in fock_dims:
            # Handle dimension inference like the old implementation:
            # - If fock_dims[idx] is 0, keep it as 0 (force bargmann)
            # - If shapes[j] is 0, keep existing fock_dims[idx]
            # - If both are > 0, take the minimum (for compatibility)
            if fock_dims[idx] == 0:
                continue  # Keep bargmann setting
            if shapes[j] == 0:
                continue  # Keep existing dimension, shape is bargmann
            # Both are positive - take minimum for compatibility
            fock_dims[idx] = min(fock_dims[idx], shapes[j])
        else:
            fock_dims[idx] = shapes[j]


def validate_polyexp_core_dims(
    core_idx: list[int | str], fock_dims: dict[int | str, int], ansatz_num: int
) -> None:
    for idx in core_idx:
        if idx not in fock_dims:
            raise ValueError(
                f"PolyExpAnsatz core dimension '{idx}' in ansatz {ansatz_num} must be pre-specified in fock_dims",
            )


def get_shapes_direct(
    ansatz_a: Ansatz,
    ansatz_b: Ansatz,
    core_idx_a: list[int],
    core_idx_b: list[int],
    fock_dims: dict[int | str, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape_a = tuple(fock_dims[i] for i in core_idx_a)
    shape_b = tuple(fock_dims[i] for i in core_idx_b)
    return shape_a, shape_b


def prepare_idx_out_separate(
    batch_idx_a: list[str],
    core_idx_a: list[int],
    batch_idx_b: list[str],
    core_idx_b: list[int],
    output_batch_chars: list[str],
    output_core: list[int],
) -> list[str | int]:
    # Return output indices in the order: batch first, then core
    return output_batch_chars + output_core


def mm_einsum(
    *args: Ansatz | list[int | str],
    output_batch: list[str],
    output_core: list[int | str],
    fock_dims: dict[int | str, int],
    contraction_path: list[tuple[int, int]],
    path_type: Literal["SSA", "LA", "UA"] = "LA",
) -> PolyExpAnsatz | ArrayAnsatz | ArrayLike:
    r"""
    Contracts a network of Ansatze according to a custom contraction order, supporting both Fock
    and Bargmann representations, batch dimensions, and named indices.

    This function generalizes the concept of Einstein summation (einsum) to quantum states and
    operators, allowing for flexible contraction of complex tensor networks in quantum optics.

    Args:
        *args: Sequence of (Ansatz, batch_indices, core_indices) triplets.
            Each Ansatz is followed by a list of batch indices (strings) and core indices (ints or strings for promotion).
        output_batch (list[str]): The batch indices to keep in the output.
        output_core (list[int | str]): The core indices to keep in the output.
        fock_dims (dict[int | str, int]): Mapping from indices to Fock space dimensions.
            Missing dimensions will be inferred from ansatz shapes.
            If a dimension is 0, the ansatz is kept in Bargmann (PolyExpAnsatz) form.
            If a dimension is > 0, the ansatz is converted to Fock (ArrayAnsatz) form.
        contraction_path (list[tuple[int, int]]): The order in which to contract the Ansatze.
        path_type (str): "SSA", "LA", or "UA". Default is "LA".

    Returns:
        PolyExpAnsatz | ArrayAnsatz | ArrayLike: The contracted result.

    Example:
        >>> # New API with separate batch and core indices
        >>> result = mm_einsum(
        ...     gbs.ansatz, [], [0, 'pnr'],        # batch=[], core=[0, 'pnr']
        ...     operation.ansatz, ['pnr'], [1, 0], # batch=['pnr'], core=[1, 0]
        ...     output_batch=['pnr'],
        ...     output_core=[1],
        ...     contraction_path=[(0, 1)],
        ...     fock_dims={0: 20, 1: 10, 'pnr': 5},
        ... )
    """
    # Step 1: Parse arguments as triplets
    ansatze = list(args[::3])
    batch_indices = list(args[1::3])
    core_indices = list(args[2::3])

    # Step 2: Populate fock_dims from ansatz shapes
    for i, (ansatz, batch_idx, core_idx) in enumerate(zip(ansatze, batch_indices, core_indices)):
        if isinstance(ansatz, ArrayAnsatz):
            update_fock_dims_from_indices(batch_idx, ansatz.batch_shape, fock_dims, i)
            update_fock_dims_from_indices(core_idx, ansatz.core_shape, fock_dims, i)
        elif isinstance(ansatz, PolyExpAnsatz):
            update_fock_dims_from_indices(batch_idx, ansatz.batch_shape, fock_dims, i)
            validate_polyexp_core_dims(core_idx, fock_dims, i)

    # Step 3: Do promotions
    for i in range(len(ansatze)):
        ansatze[i], batch_indices[i], core_indices[i] = process_ansatz_with_promotion(
            ansatze[i],
            batch_indices[i],
            core_indices[i],
            fock_dims,
        )

    # Step 4: Work directly with separate lists
    all_batch_names = set()
    for batch_idx in batch_indices:
        all_batch_names.update(batch_idx)
    names_to_chars = {name: chr(97 + i) for i, name in enumerate(all_batch_names)}

    batch_indices_chars = [[names_to_chars[s] for s in batch_idx] for batch_idx in batch_indices]
    output_batch_chars = [names_to_chars[s] for s in output_batch]

    # Contraction logic using separate lists
    if path_type == "SSA":
        contraction_path = ssa_to_linear(contraction_path)
    elif path_type == "UA":
        contraction_path = ua_to_linear(contraction_path)

    for a, b in contraction_path:
        core_idx_a, core_idx_b = core_indices[a], core_indices[b]
        batch_idx_a, batch_idx_b = batch_indices_chars[a], batch_indices_chars[b]

        common_core_idx = set(core_idx_a) & set(core_idx_b)
        force_bargmann = all(fock_dims[i] == 0 for i in common_core_idx)
        force_fock = all(fock_dims[i] != 0 for i in common_core_idx)

        if force_bargmann:
            ansatz_a = to_bargmann(ansatze[a])
            ansatz_b = to_bargmann(ansatze[b])
        elif force_fock:
            shape_a, shape_b = get_shapes_direct(
                ansatze[a], ansatze[b], core_idx_a, core_idx_b, fock_dims
            )
            ansatz_a = to_fock(ansatze[a], shape_a)
            ansatz_b = to_fock(ansatze[b], shape_b)
        else:
            raise ValueError(f"Attempted contraction of {a} and {b} with mixed-type indices.")

        # Build unified indices for contract() call
        idx_a = batch_idx_a + core_idx_a
        idx_b = batch_idx_b + core_idx_b
        idx_out = prepare_idx_out_separate(
            batch_idx_a,
            core_idx_a,
            batch_idx_b,
            core_idx_b,
            output_batch_chars,
            output_core,
        )

        result = ansatz_a.contract(ansatz_b, idx_a, idx_b, idx_out)

        # Update lists
        a_sorted, b_sorted = sorted((a, b))
        ansatze.pop(b_sorted), ansatze.pop(a_sorted)
        batch_indices_chars.pop(b_sorted), batch_indices_chars.pop(a_sorted)
        core_indices.pop(b_sorted), core_indices.pop(a_sorted)

        ansatze.append(result)
        batch_indices_chars.append([c for c in idx_out if isinstance(c, str)])
        core_indices.append([i for i in idx_out if isinstance(i, int)])

    # Step 5: Final processing
    if len(ansatze) > 1:
        raise ValueError("More than one ansatz left after contraction.")

    result = ansatze[0]
    final_batch_chars = batch_indices_chars[0]
    final_core_indices = core_indices[0]

    if len(output_batch_chars) > 1:
        result = result.reorder_batch([final_batch_chars.index(i) for i in output_batch_chars])
    if len(output_core) > 1:
        result = result.reorder([final_core_indices.index(i) for i in output_core])
    if final_core_indices and all(fock_dims[i] == 0 for i in final_core_indices):
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

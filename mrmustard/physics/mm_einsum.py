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

"""Clean implementation of the mm_einsum function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike
from opt_einsum.paths import ssa_to_linear

from mrmustard import math
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.ansatz.base import Ansatz
from mrmustard.physics.triples import identity_Abc


@dataclass
class ContractionData:
    """Holds the mutable data during contraction."""

    ansatze: list[Ansatz]
    batch_lists: list[list[str]]
    core_lists: list[list[int | str]]


@dataclass
class ContractionConfig:
    """Holds the configuration for contraction."""

    fock_dims: dict[int | str, int]
    output_batch: list[str]
    output_core: list[int | str]
    char_map: dict[str, str]


def mm_einsum(
    *args: Ansatz | list[int | str],
    output_batch: list[str],
    output_core: list[int | str],
    fock_dims: dict[int | str, int],
    contraction_path: list[tuple[int, int]],
    path_type: Literal["SSA", "LA", "UA"] = "LA",
) -> PolyExpAnsatz | ArrayAnsatz | ArrayLike:
    """Contract ansatze using specified path."""

    ansatze, batch_lists, core_lists = _parse_args(args)
    data = ContractionData(ansatze, batch_lists, core_lists)

    _promote_string_indices(data, fock_dims)

    char_map = _create_batch_char_map(data.batch_lists, output_batch)
    config = ContractionConfig(fock_dims, output_batch, output_core, char_map)

    path = _convert_path_to_LA(contraction_path, path_type)

    for i, j in path:
        _contract_pair(data, config, i, j)

    if len(data.ansatze) != 1:
        raise ValueError(f"Expected 1 result, got {len(data.ansatze)}")

    return _finalize_result(
        data.ansatze[0],
        data.batch_lists[0],
        data.core_lists[0],
        config,
    )


def _parse_args(args):
    """Parse args into (ansatz, batch, core) triplets."""
    return list(args[::3]), list(args[1::3]), list(args[2::3])


def _promote_string_indices(data, fock_dims):
    """Move string indices from core to batch via promotion."""
    for i in range(len(data.ansatze)):
        strings_in_core = [idx for idx in data.core_lists[i] if isinstance(idx, str)]
        if not strings_in_core:
            continue

        shape = tuple(fock_dims[idx] for idx in data.core_lists[i])
        data.ansatze[i] = to_fock(data.ansatze[i], shape)

        promote_pos = [j for j, idx in enumerate(data.core_lists[i]) if isinstance(idx, str)]
        data.ansatze[i] = data.ansatze[i].promote_core_to_batch(promote_pos)

        data.batch_lists[i] = data.batch_lists[i] + strings_in_core
        data.core_lists[i] = [idx for idx in data.core_lists[i] if isinstance(idx, int)]


def _create_batch_char_map(batch_lists, output_batch):
    """Create mapping from batch names to single characters.

    Includes all batch indices from input ansatze and final output.
    """
    all_names = set()
    # Collect all batch names from all input ansatze
    for batch_list in batch_lists:
        all_names.update(batch_list)
    # Also include final output batch names
    all_names.update(output_batch)
    return {name: chr(97 + i) for i, name in enumerate(sorted(all_names))}


def _convert_path_to_LA(path, path_type):
    """Convert path to linear format."""
    if path_type == "SSA":
        return ssa_to_linear(path)
    if path_type == "UA":
        return ua_to_linear(path)
    return path


def _get_shape_for_idx(idx, ansatz, core_idx, fock_dims):
    """Get shape for a specific index, inferring from ansatz if needed."""
    if idx in fock_dims:
        return fock_dims[idx]
    if not isinstance(ansatz, ArrayAnsatz):
        raise ValueError(f"Fock dimension of index {idx} is not set.")
    return ansatz.core_shape[core_idx.index(idx)]


def _get_shapes(ansatz_i, ansatz_j, core_i, core_j, fock_dims):
    """Get shapes for both ansatze, handling shared indices."""
    shared_core = set(core_i) & set(core_j)

    shape_i = {}
    shape_j = {}

    # Handle shared indices - take minimum shape
    for idx in shared_core:
        if idx in fock_dims:
            dim = fock_dims[idx]
        else:
            dim = 1_000_000_000
            if isinstance(ansatz_i, ArrayAnsatz):
                dim = min(dim, ansatz_i.core_shape[core_i.index(idx)])
            if isinstance(ansatz_j, ArrayAnsatz):
                dim = min(dim, ansatz_j.core_shape[core_j.index(idx)])
        shape_i[idx] = dim
        shape_j[idx] = dim

    # Handle leftover indices
    for idx in core_i:
        if idx not in shared_core:
            shape_i[idx] = _get_shape_for_idx(idx, ansatz_i, core_i, fock_dims)

    for idx in core_j:
        if idx not in shared_core:
            shape_j[idx] = _get_shape_for_idx(idx, ansatz_j, core_j, fock_dims)

    return (tuple(shape_i[idx] for idx in core_i), tuple(shape_j[idx] for idx in core_j))


def _prepare_ansatze_for_contraction(ansatz_i, ansatz_j, core_i, core_j, fock_dims):
    """Convert ansatze to appropriate representation for contraction."""
    shared_core = set(core_i) & set(core_j)

    # Follow original logic: check only shared indices
    if shared_core:
        force_bargmann = all(fock_dims.get(idx, 1) == 0 for idx in shared_core)
        force_fock = all(fock_dims.get(idx, 1) != 0 for idx in shared_core)
    else:
        # No shared indices - decide based on all indices being 0 (empty set -> True)
        force_bargmann = True  # Original behavior: all([]) returns True
        force_fock = False

    if force_bargmann:
        return to_bargmann(ansatz_i), to_bargmann(ansatz_j)
    if force_fock:
        shape_i, shape_j = _get_shapes(ansatz_i, ansatz_j, core_i, core_j, fock_dims)
        return to_fock(ansatz_i, shape_i), to_fock(ansatz_j, shape_j)
    raise ValueError("Mixed-type indices in contraction")


def _compute_output_indices(
    batch_i,
    batch_j,
    core_i,
    core_j,
    remaining_batches,
    remaining_cores,
    config,
):
    """Determine which indices to keep after contraction."""
    needed_batch = set(config.output_batch)
    needed_core = set(config.output_core)

    for batch_list in remaining_batches:
        needed_batch.update(batch_list)
    for core_list in remaining_cores:
        needed_core.update(core_list)

    out_batch = []
    for name in batch_i + batch_j:
        if name in needed_batch and name not in out_batch:
            out_batch.append(name)

    out_core = []
    for idx in core_i + core_j:
        if idx in needed_core and idx not in out_core:
            out_core.append(idx)

    return out_batch, out_core


def _contract_pair(data, config, i, j):
    """Contract two ansatze and update the lists."""
    # Prepare ansatze
    ansatz_a, ansatz_b = _prepare_ansatze_for_contraction(
        data.ansatze[i],
        data.ansatze[j],
        data.core_lists[i],
        data.core_lists[j],
        config.fock_dims,
    )

    # Build contraction indices
    chars_a = [config.char_map[name] for name in data.batch_lists[i]]
    chars_b = [config.char_map[name] for name in data.batch_lists[j]]
    idx_a = chars_a + data.core_lists[i]
    idx_b = chars_b + data.core_lists[j]

    # Determine output indices
    remaining_batches = [data.batch_lists[k] for k in range(len(data.ansatze)) if k not in (i, j)]
    remaining_cores = [data.core_lists[k] for k in range(len(data.ansatze)) if k not in (i, j)]

    out_batch, out_core = _compute_output_indices(
        data.batch_lists[i],
        data.batch_lists[j],
        data.core_lists[i],
        data.core_lists[j],
        remaining_batches,
        remaining_cores,
        config,
    )

    out_chars = [config.char_map[name] for name in out_batch]
    idx_out = out_chars + out_core

    # Contract and update lists
    result = ansatz_a.contract(ansatz_b, idx_a, idx_b, idx_out)

    i, j = sorted([i, j])
    data.ansatze.pop(j), data.ansatze.pop(i)
    data.batch_lists.pop(j), data.batch_lists.pop(i)
    data.core_lists.pop(j), data.core_lists.pop(i)

    data.ansatze.append(result)
    data.batch_lists.append(out_batch)
    data.core_lists.append(out_core)


def _finalize_result(result, final_batch, final_core, config):
    """Apply final reordering and conversion."""
    if len(config.output_batch) > 1:
        batch_order = [final_batch.index(name) for name in config.output_batch]
        result = result.reorder_batch(batch_order)

    if len(config.output_core) > 1:
        core_order = [final_core.index(idx) for idx in config.output_core]
        result = result.reorder(core_order)

    if final_core and all(config.fock_dims.get(idx, 1) == 0 for idx in final_core):
        result = to_bargmann(result)

    return result


def ua_to_linear(ua: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert union assignment path to linear."""
    sets = [{i} for i in sorted({j for pair in ua for j in pair})]
    path = []
    for a, b in ua:
        to_union = [s for s in sets if a in s or b in s]
        a_, b_ = sorted([sets.index(u) for u in to_union])
        sets.pop(b_), sets.pop(a_)
        sets.append(to_union[0] | to_union[1])
        path.append((a_, b_))
    return path


def to_fock(ansatz: Ansatz, shape: tuple[int, ...], stable: bool = False) -> ArrayAnsatz:
    """Convert to Fock representation."""
    if 0 in shape:
        raise ValueError("Fock space dimension is 0.")
    if isinstance(ansatz, ArrayAnsatz):
        return ansatz.reduce(shape)

    array = math.hermite_renormalized(*ansatz.triple, shape, stable)
    if ansatz._lin_sup:
        array = math.sum(array, axis=0)
    return ArrayAnsatz(array, ansatz.batch_dims)


def to_bargmann(ansatz: Ansatz) -> PolyExpAnsatz:
    """Convert to Bargmann representation."""
    if isinstance(ansatz, PolyExpAnsatz):
        return ansatz

    try:
        A, b, c = ansatz._original_abc_data
    except (TypeError, AttributeError):
        A, b, _ = identity_Abc(ansatz.core_dims)
        A = math.broadcast_to(A, (*ansatz.batch_shape, 2 * ansatz.core_dims, 2 * ansatz.core_dims))
        b = math.broadcast_to(b, (*ansatz.batch_shape, 2 * ansatz.core_dims))
        c = ansatz.array
    return PolyExpAnsatz(A, b, c)

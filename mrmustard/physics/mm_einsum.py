# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import itertools


def _CV_flops(nA: int, nB: int, m: int) -> int:
    r"""Calculate the cost of contracting two tensors with CV indices.
    Args:
        nA: Number of CV indices in the first tensor
        nB: Number of CV indices in the second tensor
        m: Number of CV indices involved in the contraction
    """
    cost = (
        m * m * m  # M inverse
        + (m + 1) * m * nA  # left matmul
        + (m + 1) * m * nB  # right matmul
        + (m + 1) * m  # addition
        + m * m * m
    )  # determinant of M
    return cost


def _fock_flops(
    fock_contracted_shape: tuple[int, ...], fock_remaining_shape: tuple[int, ...]
) -> int:
    r"""Calculate the cost of contracting two tensors with Fock indices.
    Args:
        fock_contracted_shape: shape of the indices that participate in the contraction
        fock_remaining_shape: shape of the indices that do not
    """
    if len(fock_contracted_shape) > 0:
        return np.prod(fock_contracted_shape) * np.prod(fock_remaining_shape)
    else:
        return 0


def new_indices_and_flops(
    idx1: frozenset[int], idx2: frozenset[int], fock_size_dict: dict[int, int]
) -> tuple[frozenset[int], int]:
    r"""Calculate the cost of contracting two tensors with mixed CV and Fock indices.

    This function computes both the surviving indices and the computational cost (in FLOPS)
    of contracting two tensors that contain a mixture of continuous-variable (CV) and
    Fock-space indices.

    Args:
        idx1: Set of indices for the first tensor. CV indices are integers not present
            in fock_size_dict.
        idx2: Set of indices for the second tensor. CV indices are integers not present
            in fock_size_dict.
        fock_size_dict: Dict mapping Fock index labels to their dimensions. Any index
            not in this dict is treated as a CV index.

    Returns:
        tuple[frozenset[int], int]: A tuple containing:
            - frozenset of indices that survive the contraction
            - total computational cost in FLOPS (including CV operations,
              Fock contractions, and potential decompositions)

    Example:
        >>> idx1 = frozenset({0, 1})  # 0 is CV, 1 is Fock
        >>> idx2 = frozenset({1, 2})  # 2 is Fock
        >>> fock_size_dict = {1: 2, 2: 3}
        >>> new_indices_and_flops(idx1, idx2, fock_size_dict)
        (frozenset({0, 2}), 9)
    """

    # Calculate index sets for contraction
    contracted_indices = idx1 & idx2  # Indices that get contracted away
    remaining_indices = idx1 ^ idx2  # Indices that remain after contraction
    all_fock_indices = set(fock_size_dict.keys())

    # Count CV and get Fock shapes
    num_cv_contracted = len(contracted_indices - all_fock_indices)
    fock_contracted_shape = [fock_size_dict[idx] for idx in contracted_indices & all_fock_indices]
    fock_remaining_shape = [fock_size_dict[idx] for idx in remaining_indices & all_fock_indices]

    # Calculate flops
    cv_flops = _CV_flops(
        nA=len(idx1) - num_cv_contracted, nB=len(idx2) - num_cv_contracted, m=num_cv_contracted
    )

    fock_flops = _fock_flops(fock_contracted_shape, fock_remaining_shape)

    # Try decomposing the remaining indices
    new_indices, decomp_flops = attempt_decomposition(remaining_indices, fock_size_dict)

    # flops for evaluating the ansatz with the remaining indices (measures ansatz complexity)
    eval_flops = np.prod([fock_size_dict[idx] for idx in new_indices if idx in fock_size_dict])

    total_flops = int(cv_flops + fock_flops + decomp_flops + eval_flops)
    return new_indices, total_flops


def attempt_decomposition(
    indices: set[int], fock_size_dict: dict[int, int]
) -> tuple[set[int], int]:
    r"""Attempt to reduce the number of indices by combining Fock indices,
    which is possible if there is only one CV index and multiple Fock indices.
    (This is Kasper's decompose method).

    Args:
        indices: Set of indices to potentially decompose
        fock_size_dict: Dictionary mapping indices to their dimensions

    Returns:
        tuple[frozenset[int], int]: A tuple containing:
            - frozenset of decomposed indices
            - computational cost of decomposition in FLOPS
    """
    fock_indices_shape = [fock_size_dict[idx] for idx in indices if idx in fock_size_dict]
    cv_indices = [idx for idx in indices if idx not in fock_size_dict]

    if len(cv_indices) == 1 and len(fock_indices_shape) > 1:
        new_index = max(fock_size_dict) + 1  # Create new index with size = sum of Fock index sizes
        decomposed_indices = {cv_indices[0], new_index}
        fock_size_dict[new_index] = sum(fock_indices_shape)
        decomp_flops = np.prod(fock_indices_shape)
        return frozenset(decomposed_indices), decomp_flops
    return frozenset(indices), 0


def optimal(
    inputs: list[frozenset[int]],
    fock_size_dict: dict[int, int],
    info: bool = False,
) -> list[tuple[int, int]]:
    r"""Find the optimal contraction path for a mixed CV-Fock tensor network.

    This function performs an exhaustive search over all possible contraction orders
    for a tensor network containing both continuous-variable (CV) and Fock-space tensors.
    It uses a depth-first recursive strategy to find the sequence of pairwise contractions
    that minimizes the total computational cost (FLOPS).

    CV indices are represented by integers not present in fock_size_dict, while Fock
    indices must be keys in fock_size_dict. The algorithm caches intermediate results,
    skips outer products (contractions between tensors with no shared indices), and
    prunes the search when partial paths exceed the current best cost.

    Args:
        inputs: List of index sets representing tensor indices
        fock_size_dict: Mapping from Fock index labels to dimensions
        info: If True, prints cache size diagnostics

    Returns:
        tuple[tuple[int, int], ...]: The optimal contraction path as a sequence of pairs.
            Each pair (i, j) indicates that tensors at positions i and j should be
            contracted together. The resulting tensor is placed at position len(inputs).

    Example:
        >>> inputs = [frozenset({0, 1}), frozenset({1, 2}), frozenset({2, 3})]
        >>> fock_size_dict = {1: 2, 2: 2}  # indices 0 and 3 are CV indices
        >>> optimal(inputs, fock_size_dict)
        ((0, 1), (2, 3))

    Reference:
        Based on the optimal path finder in opt_einsum:
        https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/paths.py
    """
    best_flops: int = float("inf")
    best_path: tuple[tuple[int, int], ...] = ()
    result_cache: dict[tuple[frozenset[int], frozenset[int]], tuple[frozenset[int], int]] = {}

    def _optimal_iterate(path, remaining, inputs, flops):
        nonlocal best_flops
        nonlocal best_path

        if len(remaining) == 1:
            best_flops = flops
            best_path = path
            return

        # check all remaining paths
        for i, j in itertools.combinations(remaining, 2):
            if i > j:
                i, j = j, i

            # skip outer products
            if not inputs[i] & inputs[j]:
                continue

            key = (inputs[i], inputs[j])
            try:
                new_indices, flops_ij = result_cache[key]
            except KeyError:
                new_indices, flops_ij = result_cache[key] = new_indices_and_flops(
                    *key, fock_size_dict
                )

            # sieve based on current best flops
            new_flops = flops + flops_ij
            if new_flops >= best_flops:
                continue

            # add contraction and recurse into all remaining
            _optimal_iterate(
                path=path + ((i, j),),
                inputs=inputs + (new_indices,),
                remaining=remaining - {i, j} | {len(inputs)},
                flops=new_flops,
            )

    _optimal_iterate(
        path=(), inputs=tuple(map(frozenset, inputs)), remaining=set(range(len(inputs))), flops=0
    )

    if info:
        print("len(fock_size_dict)", len(fock_size_dict), "len(result_cache)", len(result_cache))
    return best_path

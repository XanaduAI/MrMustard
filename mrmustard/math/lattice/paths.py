# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Path functions"""

from functools import lru_cache
from numba import njit, typed, typeof, types
from numba.cpython.unsafe.tuple import tuple_setitem

BINOMIAL_PATHS_PYTHON = {}


@njit
def _binomial_subspace_basis(
    cutoffs: tuple[int, ...],
    weight: int,
    mode: int,
    basis_element: tuple[int, ...],
    basis: typed.List[tuple[int, ...]],
):
    r"""Step of the recursive function to generate all indices
    of a tensor with equal weight.
    If cutoffs is an empty tuple, the the basis element is appended to the list.
    Otherwise it loops over the values of the given mode, and it calls itself recursively
    to construct the rest of the basis elements.

    Arguments:
        cutoffs (tuple[int, ...]): the cutoffs of the tensor
        weight (int): the weight of the subspace
        mode (int): the mode to loop over
        basis_element (tuple[int, ...]): the current basis element to construct
        basis (list[tuple[int, ...]]): the list of basis elements to eventually append to
    """
    if mode == len(cutoffs):
        if weight == 0:  # ran out of photons to distribute
            basis.append(basis_element)
        return

    for photons in range(cutoffs[mode]):  # could be prange?
        if weight - photons >= 0:
            basis_element = tuple_setitem(basis_element, mode, photons)
            _binomial_subspace_basis(cutoffs, weight - photons, mode + 1, basis_element, basis)


@njit
def binomial_subspace_basis(cutoffs: tuple[int, ...], weight: int):
    r"""Returns all indices of a tensor with given weight.

    Arguments:
        cutoffs (tuple[int, ...]): the cutoffs of the tensor
        weight (int): the weight of the subspace

    Returns:
        list[tuple[int, ...]]: the list of basis elements of the subspace
    """
    basis = typed.List(
        [cutoffs]
    )  # this is just so that numba can infer the type, then we remove it
    _binomial_subspace_basis(cutoffs, weight, 0, cutoffs, basis)
    return basis[1:]  # remove the dummy element


def BINOMIAL_PATHS_NUMBA_n(modes):
    r"Creates a numba dictionary to store the paths and effectively cache them."
    return typed.Dict.empty(
        key_type=typeof(((0,) * modes, 0)),
        value_type=types.ListType(typeof((0,) * modes)),
    )


Pair = tuple[int, int]
Triple = tuple[int, int, int]


@lru_cache(maxsize=None)
def constrained_binomial_subspace(
    weight: int,
    zeros: tuple[int, ...],
    singles: tuple[int, ...],
    pairs: tuple[tuple[Pair, int], ...],
    triples: tuple[tuple[Triple, int], ...],
):
    r"""Efficient implementation of the subspace basis with cutoffs (not based on filtering).

    Directly builds only the basis elements that satisfy both the weight constraint
    and the tuple cutoffs constraints, without generating the full basis first.

    Arguments:
        weight (int): the weight of the subspace
        zeros (tuple[int, ...]): a tuple of zeros of the same length as the number of modes
        singles (tuple[int, ...]): constraints on individual axes
        pairs (tuple[tuple[Pair, int], ...]): constraints on pairs of axes
        triples (tuple[tuple[Triple, int], ...]): constraints on triples of axes

    Returns:
        list[tuple[int, ...]]: the list of basis elements of the subspace
    """
    # this is just so that numba can infer the type, we remove it at the end
    basis = typed.List([singles])
    _recursive(weight, 0, zeros, basis, singles, pairs, triples)
    return basis[1:]  # remove the dummy element


@njit
def _recursive(
    photons_left: int,
    axis: int,
    current_basis_element: tuple[int, ...],
    basis: list[tuple[int, ...]],
    singles: tuple[int, ...],
    pairs: tuple[tuple[tuple[int, int], int], ...],
    triples: tuple[tuple[tuple[int, int, int], int], ...],
):
    r"""Recursive helper for efficient subspace basis with constraints."""

    if axis == len(singles):
        if photons_left == 0:
            basis.append(current_basis_element)
        return

    # Try different photon number for current basis element and axis
    for photons in range(singles[axis]):
        new_basis_element = tuple_setitem(current_basis_element, axis, photons)

        # Only recurse if constraints are not violated
        if valid(pairs, new_basis_element, axis) and valid(triples, new_basis_element, axis):
            _recursive(
                photons_left - photons, axis + 1, new_basis_element, basis, singles, pairs, triples
            )


@njit
def valid(constraints, new_basis_element, axis):
    r"""Check if the new basis element satisfies the constraints."""
    valid = True
    check = True
    for axis_tuple, max_sum in constraints:
        for m in axis_tuple:
            if m > axis:
                check = False
                break
        if check:
            constraint_sum = 0
            for i in axis_tuple:
                constraint_sum += new_basis_element[i]
                if constraint_sum >= max_sum:
                    valid = False
                    break
    return valid

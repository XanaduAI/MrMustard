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

from typing import Iterator, Optional

from numba import njit, typed, typeof, types
from numba.cpython.unsafe.tuple import tuple_setitem

from mrmustard.typing import IntVector

# Strategies are generators of indices that follow paths in an N-dim positive integer lattice.
# The paths can cover the entire lattice, or just a subset of it.
# Strategies have to be generators because they enumerate lots of indices
# and we don't want to allocate memory for all of them at once.


BINOMIAL_PATHS_PYTHON = {}


@njit
def _binomial_subspace(cutoffs, weight, mode, basis_element, basis):
    r"""Recursive function to generate all indices of a tensor with equal weight."""
    if mode == len(cutoffs):
        if weight == 0:
            basis.append(basis_element)
        return

    for photons in range(cutoffs[mode]):  # could be prange?
        if weight - photons >= 0:
            basis_element = tuple_setitem(basis_element, mode, photons)
            _binomial_subspace(cutoffs, weight - photons, mode + 1, basis_element, basis)


@njit
def binomial_subspace(cutoffs, weight):
    r"""Returns all indices of a tensor with equal weight."""
    basis = typed.List(
        [cutoffs]
    )  # this is just so that numba can infer the type, then we remove it
    _binomial_subspace(cutoffs, weight, 0, cutoffs, basis)
    return basis[1:]  # remove the dummy element


def BINOMIAL_PATHS_NUMBA_n(modes):
    return typed.Dict.empty(
        key_type=typeof(((0,) * modes, 0)),
        value_type=types.ListType(typeof((0,) * modes)),
    )


BINOMIAL_PATHS_NUMBA = {modes: BINOMIAL_PATHS_NUMBA_n(modes) for modes in range(1, 100)}


@njit
def equal_weight_path(
    cutoffs: tuple[int, ...], max_photons: Optional[int] = None
) -> Iterator[list[tuple[int, ...]]]:
    r"""yields the indices of a tensor with equal weight.
    Effectively, `cutoffs` contains local cutoffs (the maximum value of each index)
    and `max_photons` is the global cutoff (the maximum sum of all indices).
    If `max_photons` is not given, only the local cutoffs are used and the iterator
    yields  all possible indices within the tensor `cutoffs`. In this case it becomes
    like `ndindex_iter` just in a different order.
    """
    if max_photons is None:
        max_photons = sum(cutoffs) - len(cutoffs)
    for s in range(max_photons + 1):
        yield binomial_subspace(cutoffs, s)


@njit
def grey_code_iter(shape: IntVector) -> Iterator[IntVector]:
    raise NotImplementedError("Grey code order strategy not implemented yet")

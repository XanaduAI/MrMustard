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


# Note: we do not cache this and ``_binomial_subspace_basis`` as caching recursive numba functions
# has known issues. See https://github.com/numba/numba/issues/6061
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
        [cutoffs],
    )  # this is just so that numba can infer the type, then we remove it
    _binomial_subspace_basis(cutoffs, weight, 0, cutoffs, basis)
    return basis[1:]  # remove the dummy element


def BINOMIAL_PATHS_NUMBA_n(modes):
    r"Creates a numba dictionary to store the paths and effectively cache them."
    return typed.Dict.empty(
        key_type=typeof(((0,) * modes, 0)),
        value_type=types.ListType(typeof((0,) * modes)),
    )

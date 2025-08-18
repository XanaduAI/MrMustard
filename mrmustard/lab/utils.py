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

"""
This module contains the utility functions used by the classes in ``mrmustard.lab``.
"""

from __future__ import annotations

from collections.abc import Generator

from mrmustard import math


def reshape_params(n_modes: int, **kwargs) -> Generator:
    r"""
    A utility function to turn the input parameters of states and gates into
    1-dimensional tensors of length ``n_modes``.

    Args:
        n_modes: The number of modes.

    Raise:
        ValueError: If a parameter has a length which is neither equal to ``1``
        nor ``n_modes``.
    """
    for name, val in kwargs.items():
        val = math.atleast_nd(val, 1)  # noqa: PLW2901
        if len(val) == 1:
            val = math.tile(val, (n_modes,))  # noqa: PLW2901
        elif len(val) != n_modes:
            msg = f"Parameter {name} has an incompatible shape."
            raise ValueError(msg)
        yield val


def shape_check(mat, vec, dim: int, name: str):
    r"""
    Check that the given Gaussian representation is consistent with the given modes.

    Args:
        mat: matrix (e.g. A or cov, etc.)
        vec: vector (e.g. b or means, etc.)
        dim: The required dimension of the representation
        name: The name of the representation for error messages
    """
    if mat.shape[-2:] != (dim, dim) or vec.shape[-1:] != (dim,):
        msg = f"{name} representation is incompatible with the required dimension {dim}: "
        msg += f"{mat.shape[-2:]}!=({dim},{dim}) or {vec.shape[-1:]} != ({dim},)."
        raise ValueError(msg)

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

# pylint: disable=no-member

"""
This module contains the utility functions used by the classes in ``mrmustard.lab``.
"""
from __future__ import annotations

from typing import Any, Callable, Generator

from mrmustard import math
from mrmustard.math.parameters import update_euclidean, Constant, Variable


def make_parameter(
    is_trainable: bool,
    value: Any,
    name: str,
    bounds: tuple[float | None, float | None],
    update_fn: Callable = update_euclidean,
    dtype: Any = None,
):
    r"""
    Returns a constant or variable parameter with given name, value, bounds, and update function.

    Args:
        is_trainable: Whether to return a variable (``True``) or constant (``False``) parameter.
        value: The value of the returned parameter.
        name: The name of the returned parameter.
        bounds: The bounds of the returned parameter (ignored if ``is_trainable`` is ``False``).
        update_fn: The update_fn of the returned parameter (ignored if ``is_trainable`` is ``False``).
        dtype: The dtype of the returned parameter.
    """
    if isinstance(value, (Constant, Variable)):
        return value
    if not is_trainable:
        return Constant(value=value, name=name, dtype=dtype)
    return Variable(value=value, name=name, bounds=bounds, update_fn=update_fn, dtype=dtype)


def reshape_params(n_modes: str, **kwargs) -> Generator:
    r"""
    A utility function to turn the input parameters of states and gates into
    1-dimensional tensors of length ``n_modes``.

    Args:
        n_modes: The number of modes.

    Raise:
        ValueError: If a parameter has a length which is neither equal to ``1``
        nor ``n_modes``.
    """
    names = list(kwargs.keys())
    vars = list(kwargs.values())

    vars = [math.atleast_1d(var) for var in vars]

    for i, var in enumerate(vars):
        if len(var) == 1:
            var = math.tile(var, (n_modes,))
        else:
            if len(var) != n_modes:
                msg = f"Parameter {names[i]} has an incompatible shape."
                raise ValueError(msg)
        yield var


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


def multi_rep_contraction(str1, str2):
    r"""
    A representation's name in multi_rep can be either of the following:
        None: Bargmann
        Q: quadrature
        PS: phase space
        BtoQ: Bargmann to quadrature
        BtoPS: Bargmann to phase space
        QtoB: quadrature to Bargmann
        PStoB: phase space to Bargmann

    This function's goal is to compute the multiplication between any of these objects.

        Args:
            str1: The representation of the first object (on the input side)
            str2: The representation of the second object (on the output side)

    We have the following multiplication table implemented:
    None * None = None
    Q * Q = Q
    PS * PS = PS

    None * BtoQ = Q
    None * BtoPS = PS

    anything else raises an error.
    """

    if str1 == str2:
        return str1
    elif str1 == None and str2 == "BtoQ":
        return "Q"
    elif str1 == None and str2 == "BtoPS":
        return "PS"
    else:
        raise ValueError(
            f"Trying to multiply objects of inconsistant representation i.e., rep1={str1} and rep2={str2}"
        )


def BtoQ_mult_table(s: str) -> str:
    if s:
        return None
    return "Q"

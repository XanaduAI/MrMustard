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
from typing import Callable, Optional, Tuple, Any, Protocol
from functools import wraps
from mrmustard.math.parameters import update_euclidean, Constant, Variable
from mrmustard import settings


def make_parameter(
    is_trainable: bool,
    value: Any,
    name: str,
    bounds: Tuple[Optional[float], Optional[float]],
    update_fn: Callable = update_euclidean,
):
    r"""
    Returns a constant or variable parameter with given name, value, bounds, and update function.

    Args:
        is_trainable: Whether to return a variable (``True``) or constant (``False``) parameter.
        value: The value of the returned parameter.
        name: The name of the returned parameter.
        bounds: The bounds of the returned parameter (ignored if ``is_trainable`` is ``False``).
        update_fn: The update_fn of the returned parameter (ignored if ``is_trainable`` is ``False``).
    """
    if isinstance(value, (Constant, Variable)):
        return value
    if not is_trainable:
        return Constant(value=value, name=name)
    return Variable(value=value, name=name, bounds=bounds, update_fn=update_fn)


def light_copy(obj, duplicate: list[str]):
    r""" """
    instance = object.__new__(type(obj))
    instance.__dict__.update({k: v for k, v in obj.__dict__.items() if k in duplicate})
    return instance


def trainable_lazy_property(func):
    r"""
    Decorator that makes a property lazily evaluated or not depending on the settings.BACKEND flag.
    If settings.BACKEND is tensorflow, we need the property to be re-evaluated every time it is accessed
    for the computation of the gradient. If settings.BACKEND is numpy, we want to avoid re-computing
    the property every time it is accessed, so we make it lazy.

    Arguments:
        func (callable): The function to be made into a trainable property.

    Returns:
        callable: The decorated function.
    """
    attr_name = "_" + func.__name__

    if settings.BACKEND == "numpy":
        import functools  # pylint: disable=import-outside-toplevel

        @wraps(func)
        @property
        def _trainable_lazy_property(self):
            r"""
            Property getter that lazily evaluates its value. Computes the value only on the first
            call and caches the result in a private attribute for future access.

            Returns:
                any: The value of the lazy property.
            """
            if not hasattr(self, attr_name):
                setattr(self, attr_name, func(self))
            return getattr(self, attr_name)

    elif settings.BACKEND == "tensorflow":
        _trainable_lazy_property = property(func)

    else:
        raise ValueError(f"Unknown backend {settings.BACKEND}.")

    return _trainable_lazy_property


class HasWires(Protocol):
    r"""A protocol for objects that have wires."""

    @property
    def wires(self) -> Wires:
        ...


def connect(components: Sequence[HasWires]):
    r"""Connects all components (sets the same id of connected wire pairs).
    Supports mode reinitialization."""
    for i, c in enumerate(components):
        ket_modes = set(c.wires.output.ket.modes)
        bra_modes = set(c.wires.output.bra.modes)
        for c_ in components[i + 1 :]:
            common_ket = ket_modes.intersection(c_.wires.input.ket.modes)
            common_bra = bra_modes.intersection(c_.wires.input.bra.modes)
            c.wires[common_ket].output.ket.ids = c_.wires[common_ket].input.ket.ids
            c.wires[common_bra].output.bra.ids = c_.wires[common_bra].input.bra.ids
            ket_modes -= common_ket
            bra_modes -= common_bra
            if not ket_modes and not bra_modes:
                break

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

"""This module contains the classes to describe constant and variable parameters used in Mr Mustard."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from mrmustard.math.backend_manager import BackendManager

math = BackendManager()


__all__ = ["Constant", "Variable"]


# ~~~~~~~~~
# Functions
# ~~~~~~~~~


def format_bounds(param: Constant | Variable) -> str:
    r"""
    Format parameter bounds string.

    Args:
        param: The parameter to format.

    Returns:
        A string representation of the parameter bounds.
    """
    if not isinstance(param, Variable):
        return "—"

    bounds = param.bounds
    low = "-∞" if bounds[0] is None else f"{bounds[0]:.3g}"
    high = "+∞" if bounds[1] is None else f"{bounds[1]:.3g}"
    return f"({low}, {high})"


def format_dtype(param: Constant | Variable) -> str:
    r"""
    Format parameter dtype string.

    Args:
        param: The parameter to format.

    Returns:
        A string representation of the parameter dtype.
    """
    return param.value.dtype.name


def format_value(param: Constant | Variable) -> tuple[str, str]:
    r"""
    Format parameter value and shape strings.

    Args:
        param: The parameter to format.

    Returns:
        A tuple of strings representing the parameter value and shape.
    """
    value = math.asnumpy(param.value)

    # Handle arrays
    if hasattr(param.value, "shape") and param.value.shape != ():
        shape_str = str(param.value.shape)

        # Check if values should be formatted as integers
        is_integer_like = math.issubdtype(value.dtype, np.integer) or math.all(
            math.equal(math.mod(value, 1), 0),
        )

        flat = value.flat
        if len(flat) <= 3:
            # Small arrays: preserve structure, format integers appropriately
            value_str = str(value.astype(int).tolist()) if is_integer_like else str(value.tolist())
        else:
            # Large arrays: show preview with ellipsis
            if is_integer_like:
                preview = [str(int(x)) for x in flat[:3]]
            else:
                preview = [f"{x:.3g}" for x in flat[:3]]
            value_str = f"[{', '.join(preview)}, ...]"
        return value_str, shape_str

    # Handle scalars
    if math.issubdtype(value.dtype, np.integer):
        value_str = str(int(value))
    elif math.iscomplexobj(value) or math.issubdtype(value.dtype, np.complexfloating):
        # Format complex numbers with g format for both real and imaginary parts
        real_part = f"{value.real:.6g}"
        imag_part = f"{value.imag:.6g}"
        value_str = f"{real_part}+{imag_part}j" if value.imag >= 0 else f"{real_part}{imag_part}j"
    else:
        value_str = f"{float(value):.6g}"

    return value_str, "scalar"


# ~~~~~~~
# Classes
# ~~~~~~~


class Constant:
    r"""
    A parameter with a constant, immutable value.

    .. code::

      my_const = Constant(1, "my_const")

    Args:
        value: The value of this constant.
        name: The name of this constant.
        dtype: The dtype of this constant.
    """

    def __init__(self, value: Any, name: str, dtype: Any = None):
        self._value = math.astensor(value, dtype=dtype)
        self._name = name

    @property
    def name(self) -> str:
        r"""
        The name of this constant.
        """
        return self._name

    @property
    def value(self) -> Any:
        r"""
        The value of this constant.
        """
        return self._value

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        ret._value, ret._name = aux_data
        return ret

    def _tree_flatten(self):  # pragma: no cover
        children = ()
        aux_data = (self.value, self.name)
        return (children, aux_data)

    def __mul__(self, value):
        return type(self)(value=value * self.value, name=self.name)

    def __rmul__(self, value):
        return type(self)(value=self.value * value, name=self.name)


class Variable:
    r"""
    A parameter whose value can change.

    .. code::

      my_var = Variable(1, "my_var")

    Args:
        value: The value of this variable.
        name: The name of this variable.
        bounds: The numerical bounds of this variable.
        update_fn: The name of the function used to update this variable during training.
        dtype: The dtype of this variable.
    """

    def __init__(
        self,
        value: Any,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        update_fn: Literal[
            "update_euclidean", "update_orthogonal", "update_symplectic", "update_unitary"
        ] = "update_euclidean",
        dtype: Any = None,
    ):
        self._value = math.astensor(value, dtype=dtype)
        self._name = name
        self._bounds = bounds
        self._update_fn = update_fn

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        r"""
        The numerical bounds of this variable.
        """
        return self._bounds

    @property
    def name(self) -> str:
        r"""
        The name of this variable.
        """
        return self._name

    @property
    def update_fn(
        self,
    ) -> Literal["update_euclidean", "update_orthogonal", "update_symplectic", "update_unitary"]:
        r"""
        The function used to update this variable during training.
        """
        return self._update_fn

    @update_fn.setter
    def update_fn(self, value):
        self._update_fn = value

    @property
    def value(self) -> Any:
        r"""
        The value of this variable.
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = math.astensor(value)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = cls.__new__(cls)
        ret._value = children[0]
        ret._name, ret._bounds, ret._update_fn = aux_data
        return ret

    @staticmethod
    def orthogonal(
        value: Any | None,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        N: int = 1,
    ):
        r"""
        Initializes a variable with ``update_fn`` for orthogonal optimization.

        Args:
            value: The value of the returned variable. If ``None``, a random orthogonal
                matrix of dimension ``N`` is initialized.
            name: The name of the returned variable.
            bounds: The numerical bounds of the returned variable.
            N: The dimension of the random orthogonal matrix. This value is ignored if
                ``value`` is not ``None``.

        Returns:
            A variable with ``update_fn`` for orthogonal optimization.
        """
        value = value or math.random_orthogonal(N)
        return Variable(value, name, bounds, "update_orthogonal")

    @staticmethod
    def symplectic(
        value: Any,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        N: int = 1,
    ):
        r"""
        Initializes a variable with ``update_fn`` for simplectic optimization.

        Args:
            value: The value of the returned variable. If ``None``, a random symplectic
                matrix of dimension ``N`` is initialized.
            name: The name of the returned variable.
            bounds: The numerical bounds of the returned variable.
            N: The dimension of the random symplectic matrix. This value is ignored if
                ``value`` is not ``None``.

        Returns:
            A variable with ``update_fn`` for simplectic optimization.
        """
        value = value or math.random_symplectic(N)
        return Variable(value, name, bounds, "update_symplectic")

    @staticmethod
    def unitary(
        value: Any,
        name: str,
        bounds: tuple[float | None, float | None] = (None, None),
        N: int = 1,
    ):
        r"""
        Initializes a variable with ``update_fn`` for unitary optimization.

        Args:
            value: The value of the returned variable. If ``None``, a random unitary
                matrix of dimension ``N`` is initialized.
            name: The name of the returned variable.
            bounds: The numerical bounds of the returned variable.
            N: The dimension of the random unitary matrix. This value is ignored if
                ``value`` is not ``None``.

        Returns:
            A variable with ``update_fn`` for unitary optimization.
        """
        value = value or math.random_unitary(N)
        return Variable(value, name, bounds, "update_unitary")

    def _tree_flatten(self):  # pragma: no cover
        children = (self.value,)
        aux_data = (self.name, self.bounds, self.update_fn)
        return (children, aux_data)

    def __mul__(self, value):
        return type(self)(
            value=value * self.value,
            name=self.name,
            bounds=self.bounds,
            update_fn=self.update_fn,
        )

    def __rmul__(self, value):
        return type(self)(
            value=self.value * value,
            name=self.name,
            bounds=self.bounds,
            update_fn=self.update_fn,
        )

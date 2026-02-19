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

from typing import Literal

import numpy as np
from numpy.typing import DTypeLike

from mrmustard import settings
from mrmustard.math.backend_manager import BackendManager
from mrmustard.utils.typing import Tensor

math = BackendManager()

__all__ = ["Constant", "Parameter", "Variable"]


# ~~~~~~~~~
# Functions
# ~~~~~~~~~


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
        int_like = isinstance(value, np.integer)
        flat = value.flat
        if len(flat) <= 3:
            # Small arrays: preserve structure, format integers appropriately
            value_str = str(value.astype(int).tolist()) if int_like else str(value.tolist())
        else:
            # Large arrays: show preview with ellipsis
            preview = (
                [str(int(x)) for x in flat[:3]] if int_like else [f"{x:.3g}" for x in flat[:3]]
            )
            value_str = f"[{', '.join(preview)}, ...]"
        return value_str, shape_str

    # Handle scalars
    if isinstance(value, np.integer):
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


class Parameter:
    r"""
    Superclass for Constant and Variable.
    """

    @property
    def dtype(self) -> DTypeLike:
        r"""
        Returns the dtype of the parameter.
        """
        return self.value.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        r"""
        Returns the shape of the parameter.
        """
        return self.value.shape

    @classmethod
    def orthogonal(
        cls,
        name: str = "orthogonal",
        N: int = 1,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""
        Initializes a parameter in O(N) with ``update_fn`` for orthogonal optimization.

        Args:
            name: The name of the returned parameter. Defaults to "orthogonal".
            N: The dimension of the random orthogonal matrix.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
                Defaults to ``()`` for a single matrix.
        Returns:
            A variable with ``update_fn`` for orthogonal optimization or a constant.
        """
        return cls(
            value=math.random_orthogonal(N, seed=seed, batch_shape=batch_shape),
            name=name,
            update_fn="update_orthogonal",
        )

    @classmethod
    def symplectic(
        cls,
        name: str = "symplectic",
        N: int = 1,
        max_r: float = 1.0,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""
        Initializes a parameter in SP(2N, R) with ``update_fn`` for simplectic optimization.

        Args:
            name: The name of the returned parameter. Defaults to "symplectic".
            N: (half) the dimension of the random symplectic matrix.
            max_r: The maximum squeezing value sampled uniformly. Defaults to 1.0.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
                Defaults to ``()`` for a single matrix.
        Returns:
            A variable with ``update_fn`` for simplectic optimization or a constant.
        """
        return cls(
            value=math.random_symplectic(N, max_r, seed=seed, batch_shape=batch_shape),
            name=name,
            update_fn="update_symplectic",
        )

    @classmethod
    def unitary(
        cls,
        name: str = "unitary",
        N: int = 1,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""
        Initializes a parameter in U(N) with ``update_fn`` for unitary optimization.

        Args:
            name: The name of the returned parameter. Defaults to "unitary".
            N: The dimension of the random unitary matrix.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
                Defaults to ``()`` for a single matrix.
        Returns:
            A variable with ``update_fn`` for unitary optimization or a constant.
        """
        return cls(
            value=math.random_unitary(N, seed=seed, batch_shape=batch_shape),
            name=name,
            update_fn="update_unitary",
        )

    @classmethod
    def complex_normal(
        cls,
        name: str = "complex",
        variance: float | Tensor | Parameter = 1.0,
        mean: complex | Tensor | Parameter = 0.0 + 0.0j,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""
        Initializes a parameter with a circular complex normal distribution.

        This samples complex numbers where the real and imaginary parts are independent
        Gaussian random variables, each with variance ``variance/2``. This ensures the
        total variance of ``|z|²`` is ``variance``, which is the standard convention
        for complex normal distributions.

        Args:
            name: The name of the returned parameter. Defaults to "complex".
            variance: The variance of the complex distribution. The real and imaginary
                     parts each have variance ``variance/2``. Can be a scalar or array
                     for element-wise variance. Defaults to 1.0.
            mean: The mean of the complex normal distribution. Can be a scalar or array.
                  Defaults to 0.0 + 0.0j.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random values.
                         Defaults to ``()`` for a single value.

        Returns:
            A parameter with complex normal distribution.
        """
        rng = settings.get_rng(seed)
        std = math.sqrt(variance / 2.0)
        real = rng.normal(loc=math.real(mean), scale=std, size=batch_shape)
        imag = rng.normal(loc=math.imag(mean), scale=std, size=batch_shape)

        return cls(value=real + 1j * imag, name=name, update_fn="update_euclidean")

    @classmethod
    def complex_uniform(
        cls,
        name: str = "complex",
        max_r: float = 1.0,
        min_r: float = 0.0,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""
        Initializes a parameter with a complex uniform distribution in the unit disk,
        i.e. a complex number with radius between ``min_r`` and ``max_r`` and angle between 0 and 2*pi.

        Args:
            name: The name of the returned parameter. Defaults to "complex".
            max_r: The maximum radius of the complex uniform distribution. Defaults to 1.0.
            min_r: The minimum radius of the complex uniform distribution. Defaults to 0.0.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
                Defaults to ``()`` for a single matrix.

        Returns:
            A variable with ``update_fn`` for complex uniform optimization or a constant.
        """
        rng = settings.get_rng(seed)
        r = rng.uniform(low=min_r, high=max_r, size=batch_shape)
        phi = rng.uniform(low=0, high=2 * np.pi, size=batch_shape)
        return cls(value=r * math.exp(1j * phi), name=name, update_fn="update_euclidean")

    @classmethod
    def real_normal(
        cls,
        name: str = "real",
        variance: float = 1.0,
        mean: float = 0.0,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""
        Initializes a parameter with a real normal distribution,
        i.e. a real number with normally distributed.
        It defaults to a real number with mean 0.0 and variance 1.0.

        Args:
            name: The name of the returned parameter. Defaults to "real".
            variance: The variance of the real normal distribution. Defaults to 1.0.
            mean: The mean of the real normal distribution. Defaults to 0.0.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
                Defaults to ``()`` for a single matrix.
        """
        rng = settings.get_rng(seed)
        std = math.sqrt(variance)
        value = rng.normal(loc=mean, scale=std, size=batch_shape)
        return cls(value=value, name=name, update_fn="update_euclidean")

    @classmethod
    def real_uniform(
        cls,
        name: str = "real",
        low: float = 0.0,
        high: float = 1.0,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""
        Initializes a parameter with a real uniform distribution,
        i.e. a real number with uniformly distributed between min_r and max_r.
        It defaults to a real number with mean 0.0 and variance 1.0.

        Args:
            name: The name of the returned parameter. Defaults to "real".
            max_r: The maximum of the real uniform distribution. Defaults to 1.0.
            min_r: The minimum of the real uniform distribution. Defaults to 0.0.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
                Defaults to ``()`` for a single matrix.
        """
        rng = settings.get_rng(seed)
        value = rng.uniform(low=low, high=high, size=batch_shape)
        return cls(value=value, name=name, update_fn="update_euclidean")

    @classmethod
    def from_cc_init(cls, value: Tensor | Parameter, expected_dtype: str, name: str):
        r"""
        Raise error if Parameter has wrong dtype, or simply cast to expected dtype if not a
        Parameter.

        Args:
            val: The input value. Can be an array, a scalar, a Parameter, or a nested list or
                tuple.
            expected_dtype: The expected dtype string (e.g. "float64", "complex128").
            name: The name of the parameter for error messages.

        Returns:
            A Constant object with the value cast to the expected dtype (if raw value), or the
            original Parameter (if dtype matches).

        Raises:
            ValueError: If a Parameter object has the wrong dtype.
        """
        if isinstance(value, Parameter):
            dtype_name = value.value.dtype.name
            if dtype_name == expected_dtype:
                return value
            raise ValueError(
                f"Parameter {name} is a {type(value).__name__} with dtype {dtype_name}, expected "
                f"{expected_dtype}."
            )
        return Constant(value=value, name=name, dtype=expected_dtype)


class Constant(Parameter):
    r"""
    A parameter with a constant, immutable value.

    .. code::

      my_const = Constant(1, "my_const")

    Args:
        value: The value of this constant.
        name: The name of this constant.
        dtype: The dtype of this constant.
    """

    def __init__(
        self, value: Tensor | Parameter, name: str | None = None, dtype: str | None = None, **kwargs
    ):
        if isinstance(value, Parameter):
            self._value = math.astensor(value.value, dtype=dtype or value.value.dtype)
            self.name = value.name if name is None else name
        else:
            self._value = math.astensor(value, dtype=dtype or getattr(value, "dtype", None))
            self.name = name or f"const_{id(self)}"

    @property
    def value(self) -> Tensor:
        return self._value

    def __repr__(self):
        return f"Constant(name={self.name}, value={format_value(self)[0]})"

    def __mul__(self, value):
        return Constant(value=value * self.value, name=self.name)

    def __rmul__(self, value):
        return Constant(value=self.value * value, name=self.name)


class Variable(Parameter):
    r"""
    A parameter whose value can change.

    .. code::

      my_var = Variable(1, "my_var")

    Args:
        value: The value of this variable.
        name: The name of this variable.
        update_fn: The name of the function used to update this variable during training.\
        dtype: The dtype of this variable.
    """

    def __init__(
        self,
        value: Tensor | Parameter | None = None,
        name: str | None = None,
        dtype: str | None = None,
        update_fn: Literal[
            "update_euclidean", "update_orthogonal", "update_symplectic", "update_unitary"
        ] = "update_euclidean",
    ):
        if isinstance(value, Parameter):
            self.value = math.astensor(value.value, dtype=dtype or value.value.dtype)
            self.name = value.name if name is None else name
            self.update_fn = update_fn
        else:
            self.value = math.astensor(value, dtype=dtype or getattr(value, "dtype", None))
            self.name = name if name is not None else f"var_{id(self)}"
            self.update_fn = update_fn

    def __repr__(self):
        return (
            f"Variable(name={self.name}, value={format_value(self)[0]}, update_fn={self.update_fn})"
        )

    def __mul__(self, value):
        return Variable(
            value=value * self.value,
            name=self.name,
            update_fn=self.update_fn,
        )

    def __rmul__(self, value):
        return Variable(
            value=self.value * value,
            name=self.name,
            update_fn=self.update_fn,
        )

    # ~~~~~~
    # PyTree
    # ~~~~~~

    @classmethod
    def _tree_unflatten(cls, aux_data, children):  # pragma: no cover
        ret = object.__new__(cls)
        ret.value = children[0]
        ret.name, ret.update_fn = aux_data
        return ret

    def _tree_flatten(self):  # pragma: no cover
        children = (self.value,)
        aux_data = (self.name, self.update_fn)
        return (children, aux_data)

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

from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from mrmustard import settings
from mrmustard.math.backend_manager import BackendManager
from mrmustard.utils.typing import RealTensor, Tensor

math = BackendManager()

__all__ = ["Constant", "Parameter", "Variable"]

UpdateFn = Literal[
    "update_euclidean",
    "update_orthogonal",
    "update_siegel",
    "update_symplectic",
    "update_unitary",
]


# ~~~~~~~~~
# Functions
# ~~~~~~~~~


def format_dtype(param: Constant | Variable) -> str:
    r"""Format parameter dtype string.

    Args:
        param: The parameter to format.

    Returns:
        A string representation of the parameter dtype.
    """
    return param.value.dtype.name


def format_value(param: Constant | Variable) -> tuple[str, str]:
    r"""Format parameter value and shape strings.

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
    r"""Superclass for Constant and Variable."""

    def __init__(
        self,
        value: ArrayLike | Parameter | None = None,
        name: str | None = None,
        dtype: str | None = None,
        *,
        update_fn: UpdateFn = "update_euclidean",  # accepted for classmethod API; only Variable uses it
    ) -> None:
        """Common initialization for value and name. Subclasses set defaults and extra attributes."""
        if value is None:
            value = np.asarray(None)  # type: ignore[arg-type]
        if isinstance(value, Parameter):
            self._value = math.astensor(value.value, dtype=dtype or value.value.dtype)
            self.name = name if name is not None else value.name
        else:
            self._value = math.astensor(value, dtype=dtype or getattr(value, "dtype", None))
            self.name = name
        if self.name is None:
            prefix = getattr(type(self), "_default_name_prefix", None)
            if prefix is not None:
                self.name = f"{prefix}_{id(self)}"

    @property
    def value(self) -> Tensor:
        r"""Returns the value of the parameter."""
        return self._value

    @value.setter
    def value(self, value: Tensor) -> None:
        self._value = value

    @property
    def dtype(self) -> DTypeLike:
        r"""Returns the dtype of the parameter."""
        return self.value.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Returns the shape of the parameter."""
        return self.value.shape

    @classmethod
    def orthogonal(
        cls,
        name: str = "orthogonal",
        N: int = 1,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""Initializes a parameter in O(N) with ``update_fn`` for orthogonal optimization.

        Args:
            name: The name of the returned parameter.
            N: The dimension of the random orthogonal matrix.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.

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
        r"""Initializes a parameter in SP(2N, R) with ``update_fn`` for symplectic optimization.

        Args:
            name: The name of the returned parameter.
            N: (half) the dimension of the random symplectic matrix.
            max_r: The maximum squeezing value sampled uniformly.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.

        Returns:
            A variable with ``update_fn`` for symplectic optimization or a constant.
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
        r"""Initializes a parameter in U(N) with ``update_fn`` for unitary optimization.

        Args:
            name: The name of the returned parameter.
            N: The dimension of the random unitary matrix.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.

        Returns:
            A variable with ``update_fn`` for unitary optimization or a constant.
        """
        return cls(
            value=math.random_unitary(N, seed=seed, batch_shape=batch_shape),
            name=name,
            update_fn="update_unitary",
        )

    @classmethod
    def siegel(
        cls,
        name: str = "siegel",
        n: int = 1,
        max_r: float = 0.9,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""Initializes a parameter in the open Siegel disk
        :math:`\mathcal{D}_n = \{Z \in \mathbb{C}^{n\times n} : Z = Z^T,\ I - Z^* Z \succ 0\}`
        with ``update_fn`` for Siegel-disk Riemannian optimization (Bergman metric).

        The matrix is sampled with Haar-random unitary eigenbasis and Takagi values
        drawn uniformly in :math:`[0, \mathtt{max\_r})`, ensuring
        :math:`\|Z\|_\mathrm{op} < 1`. All eigenvalues of :math:`Z` lie in the open
        unit disk.

        Args:
            name: The name of the returned parameter.
            n: The dimension of the matrix.
            max_r: The maximum Takagi value of the initial matrix. Must satisfy
                ``0 <= max_r < 1`` to start strictly inside the disk.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.

        Returns:
            A variable with ``update_fn`` for Siegel-disk optimization or a constant.
        """
        return cls(
            value=math.random_siegel(n, max_r=max_r, seed=seed, batch_shape=batch_shape),
            name=name,
            update_fn="update_siegel",
        )

    @classmethod
    def complex_normal(
        cls,
        name: str = "complex",
        variance: float | RealTensor | Parameter = 1.0,
        mean: complex | Tensor | Parameter = 0.0 + 0.0j,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Parameter:
        r"""Initializes a parameter with a circular complex normal distribution.

        This samples complex numbers where the real and imaginary parts are independent
        Gaussian random variables, each with variance ``variance/2``. This ensures the
        total variance of ``|z|²`` is ``variance``, which is the standard convention
        for complex normal distributions.

        Args:
            name: The name of the returned parameter.
            variance: The variance of the complex distribution. The real and imaginary
                     parts each have variance ``variance/2``. Can be a scalar or array
                     for element-wise variance.
            mean: The mean of the complex normal distribution. Can be a scalar or array.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random values.

        Returns:
            A parameter with complex normal distribution.
        """
        rng = settings.get_rng(seed)
        variance_value = variance.value if isinstance(variance, Parameter) else variance
        mean_value = mean.value if isinstance(mean, Parameter) else mean
        std = cast(float | RealTensor, math.sqrt(variance_value / 2.0))
        real = rng.normal(loc=math.real(mean_value), scale=std, size=batch_shape)
        imag = rng.normal(loc=math.imag(mean_value), scale=std, size=batch_shape)

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
        r"""Initializes a parameter with a complex uniform distribution in the unit disk,
        i.e. a complex number with radius between ``min_r`` and ``max_r`` and angle between 0 and
        2*pi.

        Args:
            name: The name of the returned parameter.
            max_r: The maximum radius of the complex uniform distribution.
            min_r: The minimum radius of the complex uniform distribution.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.

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
        r"""Initializes a parameter with a real normal distribution,
        i.e. a real number drawn from a normal distribution with mean ``mean`` and variance
        ``variance``.

        Args:
            name: The name of the returned parameter.
            variance: The variance of the real normal distribution.
            mean: The mean of the real normal distribution.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
        """
        rng = settings.get_rng(seed)
        std = cast(RealTensor, math.sqrt(variance))
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
        r"""Initializes a parameter with a real uniform distribution,
        i.e. a real number drawn from a uniform distribution over the half-open interval
        [min_r, max_r).

        Args:
            name: The name of the returned parameter.
            low: The minimum of the real uniform distribution.
            high: The maximum of the real uniform distribution.
            seed: The seed for the random number generator.
            batch_shape: The batch shape for generating multiple random matrices.
        """
        rng = settings.get_rng(seed)
        value = rng.uniform(low=low, high=high, size=batch_shape)
        return cls(value=value, name=name, update_fn="update_euclidean")

    @classmethod
    def from_cc_init(
        cls, value: ArrayLike | Parameter, expected_dtype: str, name: str
    ) -> Constant | Variable:
        r"""Raise error if Parameter has wrong dtype, or simply cast to expected dtype if not a
        Parameter.

        Args:
            value: The input value. Can be an array, a scalar, a Parameter, or a nested list or
                tuple.
            expected_dtype: The expected dtype string (e.g. "float64", "complex128").
            name: The name of the parameter for error messages.

        Returns:
            A Constant object with the value cast to the expected dtype (if raw value), or the
            original Parameter (if dtype matches).

        Raises:
            ValueError: If a Parameter object has the wrong dtype.
        """
        if isinstance(value, (Constant, Variable)):
            dtype_name = value.value.dtype.name
            if dtype_name == expected_dtype:
                return value
            raise ValueError(
                f"Parameter {name} is a {type(value).__name__} with dtype {dtype_name}, expected "
                f"{expected_dtype}."
            )
        return Constant(value=value, name=name, dtype=expected_dtype)


class Constant(Parameter):
    r"""A parameter with a constant, immutable value.

    Args:
        value: The value of this constant.
        name: The name of this constant.
        dtype: The dtype of this constant.

    Example:
        .. code::

            my_const = Constant(1, "my_const")
    """

    _default_name_prefix = "const"

    @property
    def value(self) -> Tensor:
        return self._value

    @value.setter
    def value(self, value: Tensor) -> None:
        raise AttributeError("Constant.value is immutable")

    def __repr__(self) -> str:
        return f"Constant(name={self.name}, value={format_value(self)[0]})"

    def __mul__(self, value: Tensor | complex) -> Constant:
        return Constant(value=value * self.value, name=self.name)

    def __rmul__(self, value: Tensor | complex) -> Constant:
        return Constant(value=self.value * value, name=self.name)


class Variable(Parameter):
    r"""A parameter whose value can change.

    Args:
        value: The value of this variable.
        name: The name of this variable.
        dtype: The dtype of this variable.
        update_fn: The name of the function used to update this variable during training.

    Example:
        .. code::

            my_var = Variable(1, "my_var")
    """

    _default_name_prefix = "var"
    _update_fn: UpdateFn
    name: str  # always set by Parameter.__init__ (via _default_name_prefix when None)

    def __init__(
        self,
        value: Tensor | Parameter | None = None,
        name: str | None = None,
        dtype: str | None = None,
        update_fn: UpdateFn = "update_euclidean",
    ) -> None:
        super().__init__(value=value, name=name, dtype=dtype)
        self._update_fn = update_fn

    @property
    def update_fn(self) -> UpdateFn:
        r"""The name of the function used to update this variable during training."""
        return self._update_fn

    @update_fn.setter
    def update_fn(self, value: UpdateFn) -> None:
        self._update_fn = value

    def __repr__(self) -> str:
        return (
            f"Variable(name={self.name}, value={format_value(self)[0]}, update_fn={self.update_fn})"
        )

    def __mul__(self, value: Tensor | complex) -> Variable:
        return Variable(
            value=value * self.value,
            name=self.name,
            update_fn=self.update_fn,
        )

    def __rmul__(self, value: Tensor | complex) -> Variable:
        return Variable(
            value=self.value * value,
            name=self.name,
            update_fn=self.update_fn,
        )

    # ~~~~~~
    # PyTree
    # ~~~~~~

    @classmethod
    def _tree_unflatten(
        cls,
        aux_data: tuple[str, UpdateFn],
        children: tuple[Tensor, ...],
    ) -> Variable:  # pragma: no cover
        ret = object.__new__(cls)
        ret.value = children[0]
        ret.name, ret.update_fn = aux_data
        return ret

    def _tree_flatten(self) -> tuple[tuple[Tensor, ...], tuple[str, UpdateFn]]:  # pragma: no cover
        children = (self.value,)
        aux_data = (self.name, self._update_fn)
        return (children, aux_data)

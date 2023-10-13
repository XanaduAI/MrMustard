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

import importlib.util
import sys

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from .backend_numpy import BackendNumpy
from ..utils.typing import (
    Matrix,
    Scalar,
    Tensor,
    Trainable,
    Vector,
)

# ~~~~~~~
# Helpers
# ~~~~~~~


def lazy_import(module_name: str):
    r"""
    Returns module and loader for lazy import.

    Args:
        module_name: The name of the module to import.
    """
    try:
        return sys.modules[module_name]
    except KeyError:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        loader = importlib.util.LazyLoader(spec.loader)
        return module, loader


# lazy impost for tensorflow
module_name_tf = "mrmustard.backend.backend_tensorflow"
module_tf, loader_tf = lazy_import(module_name_tf)

# ~~~~~~~
# Classes
# ~~~~~~~


class Backend:
    _backend = BackendNumpy()

    @property
    def backend(cls):
        r"""
        The backend that is being used.
        """
        return cls._backend

    def change_backend(cls, new_backend: str):
        r"""
        Changes backend.

        Args:
            new_backend: Must be one of ``numpy`` and ``tensorflow``.

        Raises:
            ValueError: If ``new_backend`` is not one of ``numpy`` and ``tensorflow``.
        """
        if new_backend == "numpy":
            cls._backend = BackendNumpy()
        elif new_backend == "tensorflow":
            loader_tf.exec_module(module_tf)
            cls._backend = module_tf.BackendTensorflow()
        else:
            msg = f"Backend {new_backend} not supported"
            raise ValueError(msg)

    def __new__(cls):
        # singleton
        if not hasattr(cls, "instance"):
            cls.instance = super(Backend, cls).__new__(cls)
        return cls.instance

    def _apply(self, fn: str, args: Optional[Sequence[any]] = ()):
        r"""
        Applies a function ``fn`` from the backend in use to the given ``args``.
        """
        try:
            return getattr(self.backend, fn)(*args)
        except AttributeError:
            msg = f"Function ``{fn}`` not implemented for backend ``{self.backend.name}``."
            raise NotImplementedError(msg)

    # ~~~~~~~
    # Methods
    # ~~~~~~~
    # Below are the methods supported by the various backends.

    def hello(self):
        r"""A function to say hello."""
        self._apply("hello")

    def sum(self, x, y):
        r"""A function to sum two numbers."""
        return self._apply("sum", (x, y))

    def abs(self, array: Tensor) -> Tensor:
        r"""Returns the absolute value of array.

        Args:
            array: The array to take the absolute value of.

        Returns:
            The absolute value of the given ``array``.
        """
        return self._apply("abs", (array,))

    def any(self, array: Tensor) -> bool:
        r"""Returns ``True`` if any element of array is ``True``, ``False`` otherwise.

        Args:
            array (array): array to check

        Returns:
            bool: True if any element of array is True
        """
        return self._apply("any", (array,))

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype: Any = None) -> Tensor:
        r"""Returns an array of evenly spaced values within a given interval.

        Args:
            start: start of the interval
            limit: end of the interval
            delta: step size
            dtype: dtype of the returned array

        Returns:
            array: array of evenly spaced values
        """
        # NOTE: is float64 by default
        return self._apply("arange", (start, limit, delta, dtype))

    def asnumpy(self, tensor: Tensor) -> Tensor:
        r"""Converts an array to a numpy array.

        Args:
            tensor: The tensor to convert.

        Returns:
            The corrsponidng numpy array.
        """
        return self._apply("asnumpy", (tensor,))

    def assign(self, tensor: Tensor, value: Tensor) -> Tensor:
        r"""Assigns value to tensor.

        Args:
            tensor: The tensor to assign to.
            value: The value to assign.

        Returns:
            The tensor with value assigned
        """
        return self._apply("assign", (tensor, value))

    def astensor(self, array: Tensor, dtype: str):
        r"""Converts a numpy array to a tensor.

        Args:
            array: The numpy array to convert.
            dtype: The dtype of the tensor.

        Returns:
            The tensor with dtype.
        """
        return self._apply("astensor", (array, dtype))

    def atleast_1d(self, array: Tensor, dtype: str = None) -> Tensor:
        r"""Returns an array with at least one dimension.

        Args:
            array: The array to convert.
            dtype: The data type of the array.

        Returns:
            array: The array with at least one dimension.
        """
        return self._apply("atleast_1d", (array, dtype))

    def cast(self, array: Tensor, dtype) -> Tensor:
        r"""Casts ``array`` to ``dtype``.

        Args:
            array: The array to cast.
            dtype: The data type to cast to.

        Returns:
            The array cast to dtype.
        """
        return self._apply("cast", (array, dtype))

    def clip(self, array: Tensor, a_min: float, a_max: float) -> Tensor:
        r"""Clips array to the interval ``[a_min, a_max]``.

        Args:
            array: The array to clip.
            a_min: The minimum value.
            a_max: The maximum value.

        Returns:
            The clipped array.
        """
        return self._apply("clip", (array, a_min, a_max))

    def concat(self, values: Sequence[Tensor], axis: int) -> Tensor:
        r"""Concatenates values along the given axis.

        Args:
            values: The values to concatenate.
            axis: The axis along which to concatenate.

        Returns:
            The concatenated values.
        """
        return self._apply("concat", (values, axis))

    def conj(self, array: Tensor) -> Tensor:
        r"""Returns the complex conjugate of array.

        Args:
            array: The array to take the complex conjugate of.

        Returns:
            The complex conjugate of the given ``array``.
        """
        return self._apply("conj", (array))

    def constraint_func(
        self, bounds: Tuple[Optional[float], Optional[float]]
    ) -> Optional[Callable]:
        r"""Returns a constraint function for the given bounds.

        A constraint function will clip the value to the interval given by the bounds.

        .. note::

            The upper and/or lower bounds can be ``None``, in which case the constraint
            function will not clip the value.

        Args:
            bounds: The bounds of the constraint.

        Returns:
            The constraint function.
        """
        return self._apply("constraint_func", (bounds))

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
This module contains the array ansatz.
"""

from __future__ import annotations
from typing import Any, Callable, Sequence, Literal

from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from IPython.display import display

from mrmustard import math, widgets
from mrmustard.math.parameters import Variable
from mrmustard.utils.typing import Batch, Scalar, Tensor, Vector

from .base import Ansatz

__all__ = ["ArrayAnsatz"]


class ArrayAnsatz(Ansatz):
    r"""
    The ansatz of the Fock-Bargmann representation.

    Represents the ansatz as a multidimensional array.

    .. code-block::

          >>> from mrmustard.physics.ansatz import ArrayAnsatz

          >>> array = np.random.random((2, 4, 5))
          >>> ansatz = ArrayAnsatz(array)

    Args:
        array: A (potentially) batched array.
        batched: Whether the array input has a batch dimension.

    Note: The args can be passed non-batched, as they will be automatically broadcasted to the
    correct batch shape if ``batched`` is set to ``False``.
    """

    def __init__(self, array: Batch[Tensor] | None, batch_dims: int = 0):
        super().__init__()
        self.batch_dims = batch_dims
        self._batch_shape = array.shape[:batch_dims] if array is not None else None
        self._array = array
        self._original_abc_data = None

    @property
    def batch_shape(self) -> tuple[int, ...] | None:
        if self._batch_shape is None:
            self._batch_shape = self.array.shape[: self.batch_dims]
        return self._batch_shape

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array of this ansatz.
        """
        self._generate_ansatz()
        return self._array

    @array.setter
    def array(self, value):
        if not math.from_backend(value):
            value = math.astensor(value)
        self._batch_shape = value.shape[: self.batch_dims]
        self._array = value

    def _generate_ansatz(self):
        r"""
        Computes and sets the array given a function and its kwargs.
        """
        if self._should_regenerate():
            params = {}
            for name, param in self._kwargs.items():
                try:
                    params[name] = param.value
                except AttributeError:
                    params[name] = param
            self.array = self._fn(**params)

    def _should_regenerate(self):
        r"""
        Determines if the ansatz needs to be regenerated based on its current state
        and parameter types.
        """
        return self._array is None or Variable in {type(param) for param in self._kwargs.values()}

    @property
    def batch_size(self):
        return int(np.prod(self.batch_shape))

    @property
    def conj(self):
        return ArrayAnsatz(math.conj(self.array), self.batch_dims)

    @property
    def data(self) -> Batch[Tensor]:
        return self.array

    @property
    def num_vars(self) -> int:
        return len(self.array.shape) - self.batch_dims

    @property
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz.
        I.e. the vacuum component of the Fock array, whatever it may be.
        """
        return self.array[(slice(None),) * self.batch_dims + (0,) * self.num_vars]

    @property
    def triple(self) -> tuple:
        r"""
        The data of the original PolyExpAnsatz if it exists.
        """
        if self._original_abc_data is None:
            raise AttributeError("This ArrayAnsatz does not have (A,b,c) data.")
        return self._original_abc_data

    @classmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> ArrayAnsatz:
        return cls(data["array"], batch_dims=data["batch_dims"])

    @classmethod
    def from_function(cls, fn: Callable, batch_dims: int = 0, **kwargs: Any) -> ArrayAnsatz:
        ret = cls(None, batch_dims=batch_dims)
        ret._fn = fn
        ret._kwargs = kwargs
        return ret

    def contract(
        self,
        other: ArrayAnsatz,
        batch_str: str = "",
        idx1: int | tuple[int, ...] = tuple(),
        idx2: int | tuple[int, ...] = tuple(),
    ) -> ArrayAnsatz:
        r"""
        Contracts two ansatze across the specified variables and batch dimensions.
        Variables are indexed by integers, while for batch dimensions the string has the same
        syntax as in ``np.einsum``.

        Args:
            other: The other ArrayAnsatz to contract with.
            batch_str: The batch dimensions to contract over with the same syntax as in ``np.einsum``.
                If not indicated, the batch dimensions are taken in outer product.
            idx1: The variables of the first ansatz to contract.
            idx2: The variables of the second ansatz to contract.

        Returns:
            The contracted ansatz.
        """
        if batch_str == "":
            batch_str = self._outer_product_batch_str(self.batch_dims, other.batch_dims)

        idx1 = (idx1,) if isinstance(idx1, int) else idx1
        idx2 = (idx2,) if isinstance(idx2, int) else idx2
        for i, j in zip(idx1, idx2):
            if i >= self.num_vars:
                raise IndexError(
                    f"Index {i} out of bounds for representation with {self.num_vars} variables."
                )
            if j >= other.num_vars:
                raise IndexError(
                    f"Index {j} out of bounds for representation with {other.num_vars} variables."
                )

        # Parse batch string
        input_str, output_str = batch_str.split("->")
        input_parts = input_str.split(",")
        if len(input_parts) != 2:
            raise ValueError("Batch string must have exactly two input parts")

        # Start variable indices after batch indices
        start_idx = max(len(input_parts[0]), len(input_parts[1]), len(output_str))
        var_idx1 = [chr(i + ord("a") + start_idx) for i in range(self.num_vars)]
        var_idx2 = [chr(i + ord("a") + start_idx + self.num_vars) for i in range(other.num_vars)]

        # Replace contracted indices in second array with corresponding indices from first
        for i, j in zip(idx1, idx2):
            var_idx2[j] = var_idx1[i]

        # Combine batch and variable indices for einsum
        einsum_str = (
            f"{input_parts[0]}{''.join(var_idx1)},"  # first array indices
            f"{input_parts[1]}{''.join(var_idx2)}->"  # second array indices
            f"{output_str}{''.join(sorted(set(var_idx1 + var_idx2)))}"  # output indices
        )

        result = math.einsum(einsum_str, self.array, other.array)
        return ArrayAnsatz(result, batch_dims=len(output_str))

    def reduce(self, shape: int | Sequence[int]) -> ArrayAnsatz:
        r"""
        Returns a new ``ArrayAnsatz`` with a sliced array.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.physics.ansatz import ArrayAnsatz

            >>> array1 = math.arange(27).reshape((3, 3, 3))
            >>> fock1 = ArrayAnsatz(array1)

            >>> fock2 = fock1.reduce(3)
            >>> assert fock1 == fock2

            >>> fock3 = fock1.reduce(2)
            >>> array3 = [[[0, 1], [3, 4]], [[9, 10], [12, 13]]]
            >>> assert fock3 == ArrayAnsatz(array3)

            >>> fock4 = fock1.reduce((1, 3, 1))
            >>> array4 = [[[0], [3], [6]]]
            >>> assert fock4 == ArrayAnsatz(array4)

        Args:
            shape: The shape of the array of the returned ``ArrayAnsatz``.
        """
        if shape == self.batch_shape:
            return self
        length = self.num_vars
        shape = (shape,) * length if isinstance(shape, int) else shape
        if len(shape) != length:
            msg = f"Expected shape of length {length}, "
            msg += f"given shape has length {len(shape)}."
            raise ValueError(msg)

        if any(s > t for s, t in zip(shape, self.batch_shape)):
            warn(
                "Warning: the fock array is being padded with zeros. If possible, slice the arrays this one will contract with instead."
            )
            padded = math.pad(
                self.array,
                [(0, 0)] * self.batch_dims + [(0, s - t) for s, t in zip(shape, self.batch_shape)],
            )
            return ArrayAnsatz(padded, batched=True)

        ret = self.array[(slice(0, None),) * self.batch_dims + tuple(slice(0, s) for s in shape)]
        return ArrayAnsatz(ret, batched=True)

    def reorder(self, order: tuple[int, ...] | list[int]) -> ArrayAnsatz:
        order = list(range(self.batch_dims)) + [i + self.batch_dims for i in order]
        return ArrayAnsatz(math.transpose(self.array, order), self.batch_dims)

    def sum_batch(self) -> ArrayAnsatz:
        r"""
        Sums over the batch dimension of the array. Turns an object with any batch size to a batch size of 1.

        Returns:
            The collapsed ArrayAnsatz object.
        """
        return ArrayAnsatz(math.expand_dims(math.sum(self.array, axis=0), 0), self.batch_dims)

    def to_dict(self) -> dict[str, ArrayLike]:
        return {"array": self.data}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> ArrayAnsatz:
        if len(idx_z) != len(idx_zconj) or not set(idx_z).isdisjoint(idx_zconj):
            raise ValueError("The idxs must be of equal length and disjoint.")
        order = (
            list(range(self.batch_dims))
            + [
                i + self.batch_dims
                for i in range(len(self.array.shape) - 1)
                if i not in idx_z + idx_zconj
            ]
            + [i + self.batch_dims for i in idx_z]
            + [i + self.batch_dims for i in idx_zconj]
        )
        new_array = math.transpose(self.array, order)
        n = np.prod(new_array.shape[-len(idx_zconj) :])
        new_array = math.reshape(new_array, new_array.shape[: -2 * len(idx_z)] + (n, n))
        trace = math.trace(new_array)
        return ArrayAnsatz(trace, self.batch_dims)

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL or (w := widgets.fock(self)) is None:
            print(self)
            return
        display(w)

    def __add__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        try:
            diff = sum(self.array.shape[1:]) - sum(other.array.shape[1:])
            if diff < 0:
                new_array = [
                    a + b for a in self.reduce(other.array.shape[1:]).array for b in other.array
                ]
            else:
                new_array = [
                    a + b for a in self.array for b in other.reduce(self.array.shape[1:]).array
                ]
            return ArrayAnsatz(array=new_array, batched=True)
        except Exception as e:
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.") from e

    def __and__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        new_array = [math.outer(a, b) for a in self.array for b in other.array]
        return ArrayAnsatz(array=new_array, batched=True)

    def __call__(self, z: Batch[Vector]) -> Scalar:
        raise AttributeError("Cannot call this ArrayAnsatz.")

    def __eq__(self, other: Ansatz) -> bool:
        slices = (slice(0, None),) + tuple(
            slice(0, min(si, oi)) for si, oi in zip(self.array.shape[1:], other.array.shape[1:])
        )
        return np.allclose(self.array[slices], other.array[slices], atol=1e-10)

    def __mul__(self, other: Scalar | ArrayAnsatz) -> ArrayAnsatz:
        if isinstance(other, ArrayAnsatz):
            try:
                diff = sum(self.array.shape[1:]) - sum(other.array.shape[1:])
                if diff < 0:
                    new_array = [
                        a * b for a in self.reduce(other.array.shape[1:]).array for b in other.array
                    ]
                else:
                    new_array = [
                        a * b for a in self.array for b in other.reduce(self.array.shape[1:]).array
                    ]
                return ArrayAnsatz(array=new_array, batched=True)
            except Exception as e:
                raise TypeError(f"Cannot multiply {self.__class__} and {other.__class__}.") from e
        else:
            ret = ArrayAnsatz(array=self.array * other, batched=True)
            ret._original_abc_data = (
                tuple(i * j for i, j in zip(self._original_abc_data, (1, 1, other)))
                if self._original_abc_data is not None
                else None
            )
            return ret

    def __neg__(self) -> ArrayAnsatz:
        return ArrayAnsatz(array=-self.array, batched=True)

    def __truediv__(self, other: Scalar | ArrayAnsatz) -> ArrayAnsatz:
        if isinstance(other, ArrayAnsatz):
            try:
                diff = sum(self.array.shape[1:]) - sum(other.array.shape[1:])
                if diff < 0:
                    new_array = [
                        a / b for a in self.reduce(other.array.shape[1:]).array for b in other.array
                    ]
                else:
                    new_array = [
                        a / b for a in self.array for b in other.reduce(self.array.shape[1:]).array
                    ]
                return ArrayAnsatz(new_array, self.batch_dims)
            except Exception as e:
                raise TypeError(f"Cannot divide {self.__class__} and {other.__class__}.") from e
        else:
            ret = ArrayAnsatz(self.array / other, self.batch_dims)
            ret._original_abc_data = (
                tuple(i / j for i, j in zip(self._original_abc_data, (1, 1, other)))
                if self._original_abc_data is not None
                else None
            )
            return ret

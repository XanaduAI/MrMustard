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
from typing import Any, Callable, Sequence
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from IPython.display import display

from mrmustard import math, widgets
from mrmustard.math.parameters import Variable
from mrmustard.utils.typing import Batch, Scalar, Tensor

from .base import Ansatz
from ..utils import outer_product_batch_str

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
        batch_dims: The number of batch dimensions.
    """

    def __init__(self, array: Batch[Tensor] | None, batch_dims: int = 0):
        super().__init__()
        self._array = math.astensor(array) if array is not None else None
        self._batch_dims = batch_dims
        self._batch_shape = self._array.shape[:batch_dims] if array is not None else ()
        self._original_abc_data = None

    @property
    def array(self) -> Batch[Tensor]:
        r"""
        The array of this ansatz.
        """
        self._generate_ansatz()
        return self._array

    @array.setter
    def array(self, value: Tensor):
        self._array = math.astensor(value)
        self._batch_shape = value.shape[: self.batch_dims]

    @property
    def batch_dims(self) -> int:
        return self._batch_dims

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self._batch_shape

    @property
    def batch_size(self):
        return int(np.prod(self.batch_shape)) if self.batch_shape != () else 0  # tensorflow

    @property
    def core_dims(self) -> int:
        r"""
        The number of core dimensions of this ansatz.
        """
        return len(self.core_shape)

    @property
    def conj(self):
        return ArrayAnsatz(math.conj(self.array), self.batch_dims)

    @property
    def core_shape(self) -> tuple[int, ...] | None:
        r"""
        The core dimensions of this ansatz.
        """
        return self.array.shape[self.batch_dims :]

    @property
    def data(self) -> Batch[Tensor]:
        return self.array

    @property
    def num_vars(self) -> int:
        return len(self.array.shape) - self.batch_dims

    @property
    def scalar(self) -> Scalar | ArrayLike:
        r"""
        The scalar part of the ansatz.
        I.e. the vacuum component of the Fock array, whatever it may be.
        """
        return self.array[(...,) + (0,) * self.core_dims]

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
        idx1: int | tuple[int, ...] = tuple(),
        idx2: int | tuple[int, ...] = tuple(),
        batch_str: str | None = None,
    ) -> ArrayAnsatz:
        r"""
        Contracts two ansatze across the specified variables and batch dimensions.
        Variables are indexed by integers, while for batch dimensions the string has the same
        syntax as in ``np.einsum``.

        Args:
            other: The other ArrayAnsatz to contract with.
            idx1: The variables of the first ansatz to contract.
            idx2: The variables of the second ansatz to contract.
            batch_str: The (optional) batch dimensions to contract over with the
                same syntax as in ``np.einsum``. If not indicated, the batch dimensions
                are taken in outer product.

        Returns:
            The contracted ansatz.
        """
        idx1 = (idx1,) if isinstance(idx1, int) else idx1
        idx2 = (idx2,) if isinstance(idx2, int) else idx2
        for i, j in zip(idx1, idx2):
            if i >= self.core_dims:
                raise IndexError(f"Valid indices are 0 to {self.core_dims-1}. Got {i}.")
            if j >= other.core_dims:
                raise IndexError(f"Valid indices are 0 to {other.core_dims-1}. Got {j}.")

        if batch_str is None:
            batch_str = outer_product_batch_str(self.batch_shape, other.batch_shape)
        input_str, output_str = batch_str.split("->")
        input_parts = input_str.split(",")
        if len(input_parts) != 2:
            raise ValueError("Batch string must have exactly two input parts")

        # reshape the arrays to match
        shape_s = self.core_shape
        shape_o = other.core_shape

        new_shape_s = list(shape_s)
        new_shape_o = list(shape_o)
        for s, o in zip(idx1, idx2):
            new_shape_s[s] = min(shape_s[s], shape_o[o])
            new_shape_o[o] = min(shape_s[s], shape_o[o])

        reduced_self = self.reduce(new_shape_s)
        reduced_other = other.reduce(new_shape_o)

        # Start variable indices after batch indices
        start_idx = max(len(input_parts[0]), len(input_parts[1]), len(output_str))
        var_idx1 = [chr(i + ord("a") + start_idx) for i in range(reduced_self.core_dims)]
        var_idx2 = [
            chr(i + ord("a") + start_idx + reduced_self.core_dims)
            for i in range(reduced_other.core_dims)
        ]

        # Replace contracted indices in second array with corresponding indices from first
        for i, j in zip(idx1, idx2):
            var_idx2[j] = var_idx1[i]

        # Combine batch and variable indices for einsum
        einsum_str = (
            f"{input_parts[0]}{''.join(var_idx1)},"
            f"{input_parts[1]}{''.join(var_idx2)}->"
            f"{output_str}{''.join([i for i in var_idx1 + var_idx2 if i not in set(var_idx1) & set(var_idx2)])}"
        )
        result = math.einsum(einsum_str, reduced_self.array, reduced_other.array)
        return ArrayAnsatz(result, batch_dims=len(output_str))

    def reduce(self, shape: Sequence[int]) -> ArrayAnsatz:
        r"""
        Returns a new ``ArrayAnsatz`` with a sliced core shape.

        .. code-block::

            >>> from mrmustard import math
            >>> from mrmustard.physics.ansatz import ArrayAnsatz

            >>> array1 = math.arange(27).reshape((3, 3, 3))
            >>> fock1 = ArrayAnsatz(array1)

            >>> fock2 = fock1.reduce((3, 3, 3))
            >>> assert fock1 == fock2

            >>> fock3 = fock1.reduce((2, 2, 2))
            >>> array3 = [[[0, 1], [3, 4]], [[9, 10], [12, 13]]]
            >>> assert fock3 == ArrayAnsatz(array3)

            >>> fock4 = fock1.reduce((1, 3, 1))
            >>> array4 = [[[0], [3], [6]]]
            >>> assert fock4 == ArrayAnsatz(array4)

        Args:
            shape: The shape of the array of the returned ``ArrayAnsatz``.
        """
        if shape == self.core_shape:
            return self
        if len(shape) != self.core_dims:
            raise ValueError(f"Expected shape of length {self.core_dims}, got {len(shape)}.")

        if any(s > t for s, t in zip(shape, self.core_shape)):
            warn("Warning: the fock array is being padded with zeros. Is this really necessary?")
            padded = math.pad(
                self.array,
                [(0, 0)] * self.batch_dims + [(0, s - t) for s, t in zip(shape, self.core_shape)],
            )
            return ArrayAnsatz(padded, self.batch_dims)

        return ArrayAnsatz(self.array[(...,) + tuple(slice(0, s) for s in shape)], self.batch_dims)

    def reorder(self, order: tuple[int, ...] | list[int]) -> ArrayAnsatz:
        order = list(range(self.batch_dims)) + [i + self.batch_dims for i in order]
        return ArrayAnsatz(math.transpose(self.array, order), self.batch_dims)

    def to_dict(self) -> dict[str, ArrayLike]:
        return {"array": self.data, "batch_dims": self.batch_dims}

    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> ArrayAnsatz:
        if len(idx_z) != len(idx_zconj) or not set(idx_z).isdisjoint(idx_zconj):
            raise ValueError("The idxs must be of equal length and disjoint.")
        order = (
            list(range(self.batch_dims))
            + [self.batch_dims + i for i in range(self.core_dims) if i not in idx_z + idx_zconj]
            + [self.batch_dims + i for i in idx_z]
            + [self.batch_dims + i for i in idx_zconj]
        )
        new_array = math.transpose(self.array, order)
        n = np.prod(new_array.shape[-len(idx_z) :])
        new_array = math.reshape(new_array, new_array.shape[: -2 * len(idx_z)] + (n, n))
        trace = math.trace(new_array)
        return ArrayAnsatz(trace, self.batch_dims)

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

    def _ipython_display_(self):
        if widgets.IN_INTERACTIVE_SHELL or (w := widgets.fock(self)) is None:
            print(self)
            return
        display(w)

    def _should_regenerate(self):
        r"""
        Determines if the ansatz needs to be regenerated based on its current state
        and parameter types.
        """
        return self._array is None or Variable in {type(param) for param in self._kwargs.values()}

    def __add__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        r"""
        Adds two ArrayAnsatz together. In order to use the __add__ method, the ansatze must have
        the same batch dimensions. The shape of the core arrays must be such that one can be reduced
        to the other. Other will be reduced to the shape of self. If you want the opposite use other + self.
        """
        if self.batch_dims != other.batch_dims:
            raise ValueError("Batch dimensions must match.")
        if self.core_shape != other.core_shape:
            if math.prod(self.core_shape) > math.prod(other.core_shape):
                self_array = self.array
                other_array = other.reduce(self.core_shape).array
            else:
                self_array = self.reduce(other.core_shape).array
                other_array = other.array
        else:
            self_array = self.array
            other_array = other.array
        return ArrayAnsatz(array=self_array + other_array, batch_dims=self.batch_dims)

    def __and__(self, other: ArrayAnsatz) -> ArrayAnsatz:
        if self.batch_shape != other.batch_shape:
            raise ValueError("Batch shapes must match.")
        batch_size = int(np.prod(self.batch_shape))
        self_reshaped = math.reshape(self.array, (batch_size, -1))
        other_reshaped = math.reshape(other.array, (batch_size, -1))
        new = math.einsum("ab,ac -> abc", self_reshaped, other_reshaped)
        new = math.reshape(new, self.batch_shape + self.core_shape + other.core_shape)
        return ArrayAnsatz(array=new, batch_dims=self.batch_dims)

    def __call__(self, _: Any):
        raise AttributeError("Cannot call an ArrayAnsatz.")

    def __eq__(self, other: Ansatz) -> bool:
        if self.batch_shape != other.batch_shape:
            return False
        slices = tuple(slice(0, min(si, oi)) for si, oi in zip(self.core_shape, other.core_shape))
        return np.allclose(self.array[(...,) + slices], other.array[(...,) + slices], atol=1e-10)

    def __mul__(self, other: Scalar) -> ArrayAnsatz:
        return ArrayAnsatz(array=self.array * other, batch_dims=self.batch_dims)

    def __neg__(self) -> ArrayAnsatz:
        return ArrayAnsatz(array=-self.array, batch_dims=self.batch_dims)

    def __truediv__(self, other: Scalar) -> ArrayAnsatz:
        return ArrayAnsatz(array=self.array / other, batch_dims=self.batch_dims)

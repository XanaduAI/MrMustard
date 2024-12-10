# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains the Batch class.
"""

# pylint: disable=too-many-instance-attributes

from __future__ import annotations
from typing import Any, Collection, Iterable

import string
import random
from numpy.typing import NDArray

from mrmustard import math
from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    Scalar,
)

__all__ = ["Batch"]


class Batch:
    r"""
    The class responsible for keeping track of and handling batch dimensions.

    Args:
        data: The batched array.
        batch_shape: The (optional) shape of the batch dims. Defaults to the first dimension of ``data``.
        batch_labels: The (optional) labels for the batch dims. Defaults to random characters.
    """

    def __init__(
        self,
        data: ComplexMatrix | ComplexVector | ComplexTensor,
        batch_shape: tuple[int, ...] | None = None,
        batch_labels: tuple[str, ...] | None = None,
    ):
        self._data = math.astensor(data)
        self.dtype = self._data.dtype
        self._batch_shape = batch_shape or self._data.shape[:1]
        if self._data.shape[: len(self._batch_shape)] != self._batch_shape:
            raise ValueError(
                f"Invalid batch shape {self._batch_shape} for data shape {self._data.shape}."
            )
        self._batch_labels = (
            batch_labels
            if batch_labels
            else tuple((random.choice(string.ascii_letters) for _ in self._batch_shape))
        )
        self._core_shape = self._data.shape[len(self._batch_shape) :]

    @property
    def batch_labels(self) -> tuple[str, ...]:
        r"""
        The batch labels.
        """
        return self._batch_labels

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""
        The batch shape.
        """
        return self._batch_shape

    @property
    def core_shape(self) -> tuple[int, ...]:
        r"""
        The core shape.
        """
        return self._core_shape

    @property
    def data(self) -> ComplexMatrix | ComplexVector | ComplexTensor:
        r"""
        The underlying batched data.
        """
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        r"""
        The overall shape (batch_shape + core_shape).
        """
        return self.data.shape

    def __array__(self) -> NDArray:
        return math.asnumpy(self.data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # pragma: no cover
        r"""
        Implement the NumPy ufunc interface.
        """
        if method == "__call__":
            inputs = [i.data if isinstance(i, Batch) else i for i in inputs]
            return Batch(ufunc(*inputs, **kwargs), self.batch_shape, self.batch_labels)

        elif method == "reduce":
            axis = kwargs.pop("axis") or 0
            if axis > len(self.batch_shape) - 1:
                raise ValueError("Axis out of bounds.")
            input = (
                inputs[0].data if isinstance(inputs[0], Batch) else inputs[0]
            )  # assume single input
            slices = [input[(slice(None),) * axis + (i,)] for i in range(input.shape[axis])]
            batch_shape = tuple(
                (shape for idx, shape in enumerate(self.batch_shape) if idx != axis)
            )
            batch_labels = tuple(
                (label for idx, label in enumerate(self.batch_labels) if idx != axis)
            )
            data = slices[0]
            for item in slices[1:]:
                data = ufunc(data, item, **kwargs)
            return Batch(data, batch_shape, batch_labels) if batch_shape else data
        else:
            # TODO: implement more methods as needed
            raise NotImplementedError(f"Cannot call {method} on {ufunc}.")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Batch):
            return False
        return (
            math.allclose(self.data, other.data)
            and self.batch_shape == other.batch_shape
            and self.batch_labels == other.batch_labels
        )

    def __getitem__(
        self, idxs: int | slice | tuple[int, ...] | tuple[slice, ...]
    ) -> ComplexMatrix | ComplexVector | ComplexTensor | Batch:
        r"""
        Index the batch dimensions.

        Note:
            To index core dimensions use ``self.data``.
        """
        idxs = (idxs,) if not isinstance(idxs, Collection) else idxs
        if len(idxs) > len(self.batch_shape):
            raise IndexError(
                f"Too many indices for batched array: batch is {len(self.batch_shape)}-dimensional, but {len(idxs)} were indexed."
            )
        new_data = self.data[idxs]
        new_batch_shape = (
            new_data.shape[: len(self.core_shape) - 1]
            if len(self.core_shape) < len(new_data.shape)
            else ()
        )
        new_batch_labels = (
            tuple(self.batch_labels[i] for i, j in enumerate(idxs) if isinstance(j, slice))
            + self.batch_labels[len(idxs) :]
        )
        return Batch(new_data, new_batch_shape, new_batch_labels) if new_batch_shape else new_data

    def __iter__(self) -> Iterable:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __mul__(self, other: Scalar) -> Batch:
        return Batch(self.data * other, self.batch_shape, self.batch_labels)

    def __neg__(self) -> Batch:
        return -1 * self

    def __rmul__(self, other: Scalar) -> Batch:
        return self * other

    def __rtruediv__(self, other: Scalar) -> Batch:
        return Batch(other / self.data, self.batch_shape, self.batch_labels)

    def __truediv__(self, other: Scalar) -> Batch:
        return Batch(self.data / other, self.batch_shape, self.batch_labels)

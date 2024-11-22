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
from typing import Iterable

import string
import random

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
        batch_labels: list[str] | None = None,
    ):
        self._data = data

        self._batch_shape = batch_shape or (data.shape[0],)
        self._batch_labels = (
            batch_labels
            if batch_labels
            else [random.choice(string.ascii_letters) for _ in self._batch_shape]
        )

        self._core_shape = data.shape[len(self._batch_shape) :]

    @property
    def batch_labels(self) -> list[str]:
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
        r""" """
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        r"""
        The overall shape (batch_shape + core_shape).
        """
        return self.data.shape

    def conjugate(self) -> Batch:
        r""" """
        return Batch(math.conj(self.data), self.batch_shape, self.batch_labels)

    def __array__(self):
        return self.data

    def __eq__(self, other):
        if not isinstance(other, Batch):
            return False
        return (
            math.allclose(self.data, other.data)
            and self.batch_shape == other.batch_shape
            and self.batch_labels == other.batch_labels
        )

    def __getitem__(self, idxs):
        idxs = (idxs,) if isinstance(idxs, (int, slice)) else idxs
        if len(idxs) > len(self.batch_shape):
            raise IndentationError("")
        new_data = self.data[idxs]
        new_batch_shape = (
            new_data.shape[: len(self.core_shape) - 1]
            if len(self.core_shape) < len(new_data.shape)
            else ()
        )
        new_batch_labels = self.batch_labels[: len(idxs)]
        return Batch(new_data, new_batch_shape, new_batch_labels) if new_batch_shape else new_data

    def __iter__(self):
        return iter(self.data)

    def __mul__(self, other: Scalar):
        return Batch(self.data * other, self.batch_shape, self.batch_labels)

    def __truediv__(self, other: Scalar):
        return Batch(self.data / other, self.batch_shape, self.batch_labels)

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
from functools import cached_property

import string
import random

from mrmustard import math
from mrmustard.utils.typing import (
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
)

__all__ = ["Batch"]


class Batch:
    r"""
    The class responsible for keeping track of and handling batch dimensions.

    Args:
        items: The list of items in the batch.
        batch_shape: The (optional) shape of the batch dims. Defaults to a single batch dim.
        batch_labels: The (optional) labels for the batch dims. Defaults to random characters.
    """

    def __init__(
        self,
        items: list[ComplexMatrix] | list[ComplexVector] | list[ComplexTensor],
        batch_shape: tuple[int, ...] | None = None,
        batch_labels: list[str] | None = None,
    ):
        self._items = items
        self._batch_shape = batch_shape if batch_shape else (len(self._items),)
        self._batch_labels = (
            batch_labels
            if batch_labels
            else [random.choice(string.ascii_letters) for _ in self._batch_shape]
        )  # might have to rethink a better way of generating random labels

        core_shapes = [item.shape for item in self._items if isinstance(item, Iterable)]
        self._core_shape = max(core_shapes) if core_shapes else tuple()

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
    def shape(self) -> tuple[int, ...]:
        r"""
        The overall shape (batch_shape + core_shape).
        """
        return self.batch_shape + self.core_shape

    @cached_property
    def data(self) -> ComplexMatrix | ComplexVector | ComplexTensor:  # TODO: placeholder name
        r"""
        Returns ...
        """
        return math.astensor(list(self)).reshape(self.shape)

    def concat(self, other: Batch) -> Batch:
        r"""
        Concatenate this Batch with another.

        Args:
            other: The other batch to concatenate with.
        """
        items = self._items + other._items
        batch_shape, batch_label = self._new_batch(other)
        return Batch(items, batch_shape, batch_label)

    def _new_batch(self, other: Batch) -> tuple[tuple[int, ...], list[str]]:
        r"""
        Helper method to compute the new batch shape and labels given
        the concatenation of two batches.

        Args:
            other: The other Batch.
        """
        temp = {}
        for shape, label in zip(self.batch_shape, self.batch_labels):
            temp[label] = shape
        for shape, label in zip(other.batch_shape, other.batch_labels):
            temp[label] = temp.get(label, 0) + shape
        shape = tuple(temp.values())
        labels = list(temp.keys())
        return shape, labels

    def _pad(
        self, item: ComplexMatrix | ComplexVector | ComplexTensor
    ) -> ComplexMatrix | ComplexVector | ComplexTensor:
        r"""
        Helper method to pad the given item such that it's shape
        matches that of the core shape.

        Args:
            item: The item to pad.
        """
        if item.shape < self.core_shape:
            if not item.shape:
                item = math.atleast_1d(item)
            pad_size = self.core_shape[-1] - item.shape[-1]
            temp = ((0, pad_size),) * len(item.shape)
            padded_mat = math.pad(item, temp)
            return padded_mat
        else:
            return item

    def __eq__(self, other):
        if isinstance(other, Batch):
            return (
                self._items == other._items
                and self._batch_shape == other._batch_shape
                and self._batch_labels == other._batch_labels
            )
        return False

    def __iter__(self):
        for item in self._items:
            yield self._pad(item)

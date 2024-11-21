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
        batch_shape, batch_label = self._new_batch(other, "add")
        return Batch(items, batch_shape, batch_label)

    def _new_batch(self, other: Batch, mode: str) -> tuple[tuple[int, ...], list[str]]:
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
            if mode == "add":
                temp[label] = temp.get(label, 0) + shape
            elif mode == "prod":
                temp[label] = temp.get(label, 1) * shape
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
        if isinstance(item, Iterable) and item.shape < self.core_shape:
            if not item.shape:
                item = math.atleast_1d(item)
            pad_size = self.core_shape[-1] - item.shape[-1]
            temp = ((0, pad_size),) * len(item.shape)
            padded_mat = math.pad(item, temp)
            return padded_mat
        else:
            return item

    def __array__(self):
        return self.data

    def __eq__(self, other):
        if isinstance(other, Batch):
            return (
                math.allclose(self.data, other.data)
                and self._batch_shape == other._batch_shape
                and self._batch_labels == other._batch_labels
            )
        return False

    def __getitem__(self, idxs: int | slice | tuple[int, ...] | tuple[slice, ...]) -> Batch:
        idxs = (idxs,) if isinstance(idxs, (int, slice)) else idxs

        if len(idxs) > len(self.batch_shape):
            raise IndexError("Too many indices for Batch.")

        idxs = tuple(
            (
                idx if isinstance(idx, int) else slice(idx.start or 0, idx.stop or shape, idx.step)
                for shape, idx in zip(self.batch_shape, idxs)
            )
        )

        if not all(
            (
                idx.stop < shape + 1 if isinstance(idx, slice) else idx < shape
                for shape, idx in zip(self.batch_shape, idxs)
            )
        ):
            raise IndexError("Indices are out of bounds.")

        items = self._items[idxs[0]]
        items = [items] if not isinstance(items, list) else items
        offset = self.batch_shape[0]
        new_batch_shape = [len(items)]

        for shape, idx in zip(self.batch_shape[1:], idxs[1:]):
            if isinstance(idx, slice):
                slices = (idx.start + offset, idx.stop + offset, idx.step)
                temp_idx = slice(*slices)
            else:
                temp_idx = idx + offset
            offset += shape
            temp_items = self._items[temp_idx]
            temp_items = [temp_items] if not isinstance(temp_items, list) else temp_items
            new_batch_shape.append(len(temp_items))
            items.append(*temp_items)

        return Batch(items, tuple(new_batch_shape), self._batch_labels[: len(idxs)])

    def __iter__(self):
        for item in self._items:
            yield self._pad(item)

    def __len__(self):
        return len(self.batch_shape)

    def conjugate(self):
        return Batch(
            [math.conj(item) for item in self._items], self._batch_shape, self._batch_labels
        )

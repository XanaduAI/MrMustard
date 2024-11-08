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
)

__all__ = ["Batch"]


class Batch:
    r"""
    The class responsible for keeping track of and handling batch dimensions.

    Args:
    """

    def __init__(
        self,
        items: list[ComplexMatrix | ComplexVector | ComplexTensor],
        shape: tuple[int, ...] | None = None,
        batch_label: list[str] | None = None,
    ):
        self._items = items
        self._batch_shape = shape if shape else (len(self._items),)
        self._batch_label = (
            batch_label
            if batch_label
            else [random.choice(string.ascii_letters) for _ in self._batch_shape]
        )

        core_shapes = [item.shape for item in self._items if isinstance(item, Iterable)]
        self._core_shape = max(core_shapes) if core_shapes else tuple()

    @property
    def batch_label(self):
        r""" """
        return self._batch_label

    @property
    def batch_shape(self):
        r""" """
        return self._batch_shape

    @property
    def core_shape(self):
        r""" """
        return self._core_shape

    @property
    def shape(self):
        r""" """
        return self.batch_shape + self.core_shape

    def concat(self, other: Batch) -> Batch:
        r""" """
        items = self._items + other._items
        batch_shape, batch_label = self._new_batch(other)
        return Batch(items, batch_shape, batch_label)

    def _new_batch(self, other: Batch) -> tuple[tuple[int, ...], list[str]]:
        r""" """
        temp = {}
        for shape, label in zip(self.batch_shape, self.batch_label):
            temp[label] = shape
        for shape, label in zip(other.batch_shape, other.batch_label):
            temp[label] = temp.get(label, 0) + shape
        shape = tuple(temp.values())
        labels = list(temp.keys())
        return shape, labels

    def _pad(self, item):
        if item.shape < self.core_shape:
            if not item.shape:
                item = math.atleast_1d(item)
            pad_size = self.core_shape[-1] - item.shape[-1]
            temp = ((0, pad_size),) * len(item.shape)
            padded_mat = math.pad(item, temp)
            return padded_mat
        else:
            return item

    def __iter__(self):
        for item in self._items:
            yield self._pad(item)

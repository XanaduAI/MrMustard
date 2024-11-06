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
from typing import Any, Iterable

__all__ = ["Batch"]


class Batch:
    r"""
    The class responsible for keeping track of and handling batch dimensions.

    Args:
    """

    def __init__(
        self,
        items: Any | list[Any],
        shape: tuple[int, ...] | None = None,
        batch_label: str | None = None,
    ):
        self._items = items if isinstance(items, Iterable) else [items]
        self._shape = shape if shape else (len(self._items),)
        self._batch_label = batch_label
        self._index = 0

    @property
    def shape(self):
        return self._shape

    @property
    def batch_label(self):
        return self._batch_label

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < sum(self._shape) - 1:
            self._index += 1
            return self._items[self._index]
        else:
            raise StopIteration

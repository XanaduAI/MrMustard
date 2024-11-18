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
This module contains
"""

from __future__ import annotations
from abc import ABC, abstractmethod

__all__ = ["TempName"]


class TempName(ABC):
    r"""
    A base class
    """

    def __init__(self, array) -> None:
        self._array = array

    @property
    def array(self):
        r""" """
        return self._array

    @abstractmethod
    def __and__(self, other):
        r""" """

    @abstractmethod
    def __eq__(self, other):
        r""" """

    @abstractmethod
    def __getitem__(self, idx):
        r""" """

    @abstractmethod
    def __mul__(self, other):
        r""" """

    @abstractmethod
    def __neg__(self):
        r""" """

    @abstractmethod
    def __truediv__(self, other):
        r""" """

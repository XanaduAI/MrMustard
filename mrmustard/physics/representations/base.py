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
This module contains the classes for the available representations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable

from mrmustard.utils.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    Scalar,
    Tensor,
)

__all__ = ["Representation"]


class Representation(ABC):
    r"""
    A base class for representations.

    Representations can be initialized using the ``from_ansatz`` method, which automatically equips
    them with all the functionality required to perform mathematical operations, such as equality,
    multiplication, subtraction, etc.
    """

    def __init__(self) -> None:
        self._contract_idxs: tuple[int, ...] = ()
        self._fn = None
        self._kwargs = {}

    @property
    @abstractmethod
    def data(self) -> tuple | Tensor:
        r"""
        The data of the representation.
        For now, it's the triple for Bargmann and the array for Fock.
        """

    @property
    @abstractmethod
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the representation.
        For now it's ``c`` for Bargmann and the array for Fock.
        """

    @property
    @abstractmethod
    def triple(
        self,
    ) -> tuple[Batch[ComplexMatrix], Batch[ComplexVector], Batch[ComplexTensor]]:
        r"""
        The batch of triples :math:`(A_i, b_i, c_i)`.
        """

    @abstractmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Representation:
        r"""
        Returns a representation from a function and kwargs.
        """

    @abstractmethod
    def reorder(self, order: tuple[int, ...] | list[int]) -> Representation:
        r"""
        Reorders the representation indices.
        """

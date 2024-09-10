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
This module contains the base representation class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable

from numpy.typing import ArrayLike

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
    """

    def __init__(self) -> None:
        self._contract_idxs: tuple[int, ...] = ()
        self._fn = None
        self._kwargs = {}

    @property
    @abstractmethod
    def batch_size(self) -> int:
        r"""
        The batch size of the representation.
        """

    @property
    @abstractmethod
    def conj(self) -> Representation:
        r"""
        The conjugate of the representation.
        """

    @property
    @abstractmethod
    def data(self) -> tuple | Tensor:
        r"""
        The data of the representation.
        For now, it's the triple for Bargmann and the array for Fock.
        """

    @property
    @abstractmethod
    def num_vars(self) -> int:
        r"""
        The number of variables in the representation.
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

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> Representation:
        r"""
        Deserialize a Representation.
        """

    @classmethod
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

    @abstractmethod
    def to_dict(self) -> dict[str, ArrayLike]:
        r"""
        Serialize a Representation.
        """

    @abstractmethod
    def trace(self, idxs1: tuple[int, ...], idxs2: tuple[int, ...]) -> Representation:
        r"""
        Implements the partial trace over the given index pairs.

        Args:
            idxs1: The first part of the pairs of indices to trace over.
            idxs2: The second part.

        Returns:
            The traced-over representation.
        """

    @abstractmethod
    def _generate_ansatz(self):
        r"""
        This method computes and sets data given a function
        and some kwargs.
        """

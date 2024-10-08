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
This module contains the base ansatz class.
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
    Vector,
)

__all__ = ["Ansatz"]


class Ansatz(ABC):
    r"""
    A base class for ansatz.
    """

    def __init__(self) -> None:
        self._contract_idxs: tuple[int, ...] = ()
        self._fn = None
        self._kwargs = {}

    @property
    @abstractmethod
    def batch_size(self) -> int:
        r"""
        The batch size of the ansatz.
        """

    @property
    @abstractmethod
    def conj(self) -> Ansatz:
        r"""
        The conjugate of the ansatz.
        """

    @property
    @abstractmethod
    def data(self) -> tuple | Tensor:
        r"""
        The data of the ansatz.
        For now, it's the triple for Bargmann and the array for Fock.
        """

    @property
    @abstractmethod
    def num_vars(self) -> int:
        r"""
        The number of variables in the ansatz.
        """

    @property
    @abstractmethod
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz.
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
    def from_dict(cls, data: dict[str, ArrayLike]) -> Ansatz:
        r"""
        Deserialize a Representation.
        """

    @classmethod
    @abstractmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Ansatz:
        r"""
        Returns an ansatz from a function and kwargs.
        """

    @abstractmethod
    def reorder(self, order: tuple[int, ...] | list[int]) -> Ansatz:
        r"""
        Reorders the ansatz indices.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, ArrayLike]:
        r"""
        Serialize a Representation.
        """

    @abstractmethod
    def trace(self, idxs1: tuple[int, ...], idxs2: tuple[int, ...]) -> Ansatz:
        r"""
        Implements the partial trace over the given index pairs.

        Args:
            idxs1: The first part of the pairs of indices to trace over.
            idxs2: The second part.

        Returns:
            The traced-over ansatz.
        """

    @abstractmethod
    def _generate_ansatz(self):
        r"""
        This method computes and sets data given a function
        and some kwargs.
        """

    @abstractmethod
    def __add__(self, other: Ansatz) -> Ansatz:
        r"""
        Adds this ansatz and another ansatz.

        Args:
            other: Another ansatz.

        Returns:
            The addition of this ansatz and other.
        """

    @abstractmethod
    def __and__(self, other: Ansatz) -> Ansatz:
        r"""
        Tensor product of this ansatz with another.

        Args:
            other: Another ansatz.

        Returns:
            The tensor product of this ansatz and other.
        """

    @abstractmethod
    def __call__(self, z: Batch[Vector]) -> Scalar | Ansatz:
        r"""
        Evaluates this ansatz at a given point in the domain.

        Args:
            z: point in C^n where the function is evaluated

        Returns:
            The value of the function if ``z`` has no ``None``, else it returns a new ansatz.
        """

    @abstractmethod
    def __eq__(self, other: Ansatz) -> bool:
        r"""
        Whether this ansatz is equal to another.
        """

    @abstractmethod
    def __getitem__(self, idx: int | tuple[int, ...]) -> Ansatz:
        r"""
        Returns a copy of self with the given indices marked for contraction.
        """

    @abstractmethod
    def __matmul__(self, other: Ansatz) -> Ansatz:
        r"""
        Implements the inner product of representations over the marked indices.

        Args:
            other: Another ansatz.

        Returns:
            The resulting ansatz.
        """

    @abstractmethod
    def __mul__(self, other: Scalar | Ansatz) -> Ansatz:
        r"""
        Multiplies this ansatz by a scalar or another ansatz.

        Args:
            other: A scalar or another ansatz.

        Raises:
            TypeError: If other is neither a scalar nor an ansatz.

        Returns:
            The product of this ansatz and other.
        """

    @abstractmethod
    def __neg__(self) -> Ansatz:
        r"""
        Negates the values in the ansatz.
        """

    def __rmul__(self, other: Ansatz | Scalar) -> Ansatz:
        r"""
        Multiplies this ansatz by another or by a scalar on the right.
        """
        return self.__mul__(other)

    def __sub__(self, other: Ansatz) -> Ansatz:
        r"""
        Subtracts other from this ansatz.
        """
        try:
            return self.__add__(-other)
        except AttributeError as e:
            raise TypeError(f"Cannot subtract {self.__class__} and {other.__class__}.") from e

    @abstractmethod
    def __truediv__(self, other: Scalar | Ansatz) -> Ansatz:
        r"""
        Divides this ansatz by another ansatz.

        Args:
            other: A scalar or another ansatz.

        Returns:
            The division of this ansatz and other.
        """

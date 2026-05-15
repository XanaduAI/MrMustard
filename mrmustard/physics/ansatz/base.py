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
from collections.abc import Sequence
from typing import Any

from numpy.typing import ArrayLike

from mrmustard.utils.typing import (
    Batch,
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
        self._lin_sup = False
        self._batch_shape = ()

    @property
    @abstractmethod
    def batch_dims(self) -> int:
        r"""
        The number of batch dimensions of the ansatz.
        """

    @property
    @abstractmethod
    def batch_shape(self) -> tuple[int, ...]:
        r"""
        The batch shape of the ansatz.
        """

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
    def core_dims(self) -> int:
        r"""
        The number of core dimensions of the ansatz.
        """

    @property
    @abstractmethod
    def data(self) -> tuple | Tensor:
        r"""
        The data of the ansatz.
        For now, it's the triple for PolyExpAnsatz and the array for ArrayAnsatz.
        """

    @property
    @abstractmethod
    def num_vars(self) -> int:
        r"""
        The number of variables of this ansatz.
        """

    @property
    @abstractmethod
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz.
        For now it's ``c`` for PolyExpAnsatz and the array for ArrayAnsatz.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, ArrayLike]) -> Ansatz:
        r"""
        Deserialize an Ansatz.

        Args:
            data: The data to deserialize.

        Returns:
            An Ansatz.
        """

    @abstractmethod
    def concat(self, other: Ansatz, axis: int = 0) -> Ansatz:
        r"""
        Concatenates this ansatz with another along a specified batch axis.

        All batch axes except the concatenation axis must agree in size.
        Core dimensions and other properties must match according to the
        specific ansatz type.

        Args:
            other: Another ansatz of the same type.
            axis: The batch axis along which to concatenate (default 0).

        Returns:
            A new ansatz with concatenated batch dimensions.

        Raises:
            ValueError: If the ansatze have incompatible shapes or properties.
        """

    @abstractmethod
    def contract(
        self,
        other: Ansatz,
        idxs: tuple[Sequence[int], Sequence[int]],
    ) -> Ansatz:
        r"""
        Contract two ansatz along the specified core indices, broadcasting batch dimensions.

        Args:
            other: Another ansatz.
            idxs: Tuple ``(idx_self, idx_other)`` of sequences of core-axis indices
                (0-based, relative to the core variables of each operand) to contract.
                The two sequences must have the same length. Negative indices are
                supported and are interpreted relative to the number of core variables.
        Returns:
            The resulting contracted ansatz.
        """

    @abstractmethod
    def reorder(self, order: tuple[int, ...] | list[int]) -> Ansatz:
        r"""
        Reorders the ansatz indices.

        Args:
            order: The desired order of the ansatz indices.

        Returns:
            A new Ansatz with reordered indices.
        """

    @abstractmethod
    def reorder_batch(self, order: Sequence[int]) -> Ansatz:
        r"""
        Reorders the batch dimensions of the ansatz.
        The length of ``order`` must equal the number of batch dimensions.
        This method returns a new ansatz object.

        Args:
            order: The desired order of the batch dimensions.

        Returns:
            A new Ansatz with reordered batch dimensions.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, ArrayLike]:
        r"""
        Serialize an Ansatz.

        Returns:
            A dictionary containing the serialized data.
        """

    @abstractmethod
    def trace(self, idx_z: tuple[int, ...], idx_zconj: tuple[int, ...]) -> Ansatz:
        r"""
        Implements the partial trace over the given index pairs.

        Args:
            idx_z: The first part of the pairs of indices to trace over.
            idx_zconj: The second part.

        Returns:
            The traced-over ansatz.
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
    def __getitem__(self, index: Any) -> Ansatz:
        r"""
        Batch-only indexing. Supports integers, slices, None (newaxis), and Ellipsis.
        Indexing is restricted to the first ``batch_dims`` axes.
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

    def __rmul__(self, other: Scalar | Ansatz) -> Ansatz:
        r"""
        Multiplies this ansatz by another or by a scalar on the right.
        """
        return self.__mul__(other)

    def __sub__(self, other: Scalar | Ansatz) -> Ansatz:
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

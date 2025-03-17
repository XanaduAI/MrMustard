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

from mrmustard import math

__all__ = ["Ansatz"]


class Ansatz(ABC):
    r"""
    A base class for ansatz.
    """

    def __init__(self) -> None:
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
        For now, it's the triple for PolyExpAnsatz and the array for ArrayAnsatz.
        """

    @property
    @abstractmethod
    def scalar(self) -> Scalar:
        r"""
        The scalar part of the ansatz.
        For now it's ``c`` for PolyExpAnsatz and the array for ArrayAnsatz.
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
        Deserialize an Ansatz.
        """

    @classmethod
    @abstractmethod
    def from_function(cls, fn: Callable, **kwargs: Any) -> Ansatz:
        r"""
        Returns an ansatz from a function and kwargs.
        """

    @abstractmethod
    def contract(
        self,
        other: Ansatz,
        idx1: int | tuple[int, ...] = tuple(),
        idx2: int | tuple[int, ...] = tuple(),
        batch_str: str = "",
    ) -> Ansatz:
        r"""
        Contract two ansatz together.
        Args:
            other: Another ansatz.
            idx1: The (optional) index of the first ansatz to contract.
            idx2: The (optional) index of the second ansatz to contract.
        Returns:
            The resulting contracted ansatz.
        """

    @abstractmethod
    def reorder(self, order: tuple[int, ...] | list[int]) -> Ansatz:
        r"""
        Reorders the ansatz indices.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, ArrayLike]:
        r"""
        Serialize an Ansatz.
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

    @staticmethod
    def _outer_product_batch_str(*ndims: int) -> str:
        r"""
        Creates the einsum string for the outer product of the given tuple of dimensions.
        E.g. for (2,1,3) it returns ab,c,def->abcdef
        """
        strs = []
        offset = 0
        for ndim in ndims:
            strs.append("".join([chr(97 + i + offset) for i in range(ndim)]))
            offset += ndim
        return ",".join(strs) + "->" + "".join(strs)

    @staticmethod
    def _zip_batch_strings(*ndims: int) -> str:
        r"""
        Creates a batch string for zipping over the batch dimensions.
        """
        if len(set(ndims)) != 1:
            raise ValueError(f"Arrays must have the same number of batch dimensions, got {ndims}")
        str_ = "".join([chr(97 + i) for i in range(ndims[0])])
        return ",".join([str_] * len(ndims)) + "->" + str_

    @staticmethod
    def _reshape_args_to_batch_string(
        args: list[ArrayLike], batch_string: str
    ) -> tuple[list[ArrayLike], tuple[int, ...]]:
        r"""
        Reshapes arguments to match the batch string by inserting singleton dimensions where needed
        so that they are broadcastable.
        E.g. given two arrays of shape (2,7) and (3,7) and string ab,cb->abc, it reshapes them to
        shape (2,7,1) and (1,7,3).
        """
        # Parse the batch string
        input_specs, output_spec = batch_string.split("->")
        input_specs = input_specs.split(",")
        if len(input_specs) != len(args):
            raise ValueError(
                f"Number of input specifications ({len(input_specs)}) does not match number of arguments ({len(args)})"
            )

        args = [math.astensor(arg) for arg in args]

        # Determine the size of each dimension in the output
        dim_sizes = {}
        for arg, spec in zip(args, input_specs):
            for dim, label in zip(arg.shape, spec):
                if label in dim_sizes and dim_sizes[label] != dim:
                    raise ValueError(
                        f"Dimension {label} has inconsistent sizes: got {dim_sizes[label]} and {dim}"
                    )
                dim_sizes[label] = dim

        reshaped = []
        for arg, spec in zip(args, input_specs):
            new_shape = [dim_sizes[label] if label in spec else 1 for label in output_spec]
            reshaped.append(math.reshape(arg, new_shape))
        return reshaped

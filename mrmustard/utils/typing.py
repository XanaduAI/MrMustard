# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module containing all base type annotations."""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np

__all__ = [
    "Batch",
    "ComplexMatrix",
    "ComplexTensor",
    "ComplexVector",
    "IntMatrix",
    "IntTensor",
    "IntVector",
    "Matrix",
    "RealMatrix",
    "RealTensor",
    "RealVector",
    "Scalar",
    "Tensor",
    "Trainable",
    "UIntMatrix",
    "UIntTensor",
    "UIntVector",
    "Vector",
]

R = TypeVar("R", np.float16, np.float32, np.float64)
C = TypeVar("C", np.complex64, np.complex128)
Z = TypeVar("Z", np.int16, np.int32, np.int64)
N = TypeVar("N", np.uint16, np.uint32, np.uint64)

Scalar = R | C | Z | N
Vector = np.ndarray[tuple[int], Scalar]
Matrix = np.ndarray[tuple[int, int], Scalar]
Tensor = np.ndarray[tuple[int, ...], Scalar]

RealVector = np.ndarray[tuple[int], R]
ComplexVector = np.ndarray[tuple[int], C]
IntVector = np.ndarray[tuple[int], Z]
UIntVector = np.ndarray[tuple[int], N]

RealMatrix = np.ndarray[tuple[int, int], R]
ComplexMatrix = np.ndarray[tuple[int, int], C]
IntMatrix = np.ndarray[tuple[int, int], Z]
UIntMatrix = np.ndarray[tuple[int, int], N]

RealTensor = np.ndarray[tuple[int, ...], R]
ComplexTensor = np.ndarray[tuple[int, ...], C]
IntTensor = np.ndarray[tuple[int, ...], Z]
UIntTensor = np.ndarray[tuple[int, ...], N]


# Revisit when requiring python 3.12 (see PEP 695)
T_co = TypeVar(
    "T_co",
    RealVector,
    ComplexVector,
    IntVector,
    UIntVector,
    RealMatrix,
    ComplexMatrix,
    IntMatrix,
    UIntMatrix,
    RealTensor,
    ComplexTensor,
    IntTensor,
    UIntTensor,
    covariant=True,
)

Trainable = TypeVar("Trainable")


@runtime_checkable
class Batch(Protocol[T_co]):
    r"""Anything that can iterate over objects of type T_co."""

    def __iter__(self) -> Iterator[T_co]:
        pass

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
__all__ = [
    "RealVector",
    "ComplexVector",
    "IntVector",
    "UIntVector",
    "RealMatrix",
    "ComplexMatrix",
    "IntMatrix",
    "UIntMatrix",
    "RealTensor",
    "ComplexTensor",
    "IntTensor",
    "UIntTensor",
    "Batch",
]
from typing import Tuple, TypeVar, Protocol, Iterator, runtime_checkable

import numpy as np

R = TypeVar("R", np.float16, np.float32, np.float64)
C = TypeVar("C", np.complex64, np.complex128)
Z = TypeVar("Z", np.int16, np.int32, np.int64)
N = TypeVar("N", np.uint16, np.uint32, np.uint64)

RealVector = np.ndarray[Tuple[int], R]
ComplexVector = np.ndarray[Tuple[int], C]
IntVector = np.ndarray[Tuple[int], Z]
UIntVector = np.ndarray[Tuple[int], N]

RealMatrix = np.ndarray[Tuple[int, int], R]
ComplexMatrix = np.ndarray[Tuple[int, int], C]
IntMatrix = np.ndarray[Tuple[int, int], Z]
UIntMatrix = np.ndarray[Tuple[int, int], N]

RealTensor = np.ndarray[Tuple[int, ...], R]
ComplexTensor = np.ndarray[Tuple[int, ...], C]
IntTensor = np.ndarray[Tuple[int, ...], Z]
UIntTensor = np.ndarray[Tuple[int, ...], N]

T = TypeVar(
    "T",
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


@runtime_checkable
class Batch(Protocol[T]):
    def __iter__(self) -> Iterator[T]:
        ...

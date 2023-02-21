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
from typing import Tuple, TypeVar, Protocol, Iterator

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

T = TypeVar("T", covariant=True)


class Batch(Protocol[T]):
    def __iter__(self) -> Iterator[T]:
        ...


# minitest
x: ComplexVector = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
y: RealMatrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
z: IntTensor = np.array([[[1, 2, -3], [4, 5, -6]], [[7, 8, 9], [10, 11, 12]]])
w: UIntVector = np.array([1, 2, 3])


# minitest
X: Batch[RealVector] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
Y: Batch[IntVector] = [np.array([1, 2, 3]), np.array([2, -3, -3])]

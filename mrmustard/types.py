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

# from numbers import Number
from typing import Generic, Sequence, Tuple, TypeVar, Union

import numpy as np

Number = np.number

Shape = TypeVar("Shape", bound=Sequence)
DtypeVar = TypeVar("DtypeVar", bound=np.dtype)


class Array(Generic[Shape, DtypeVar]):
    ...


class Vector(Array[Tuple[int], DtypeVar]):
    ...


class Matrix(Array[Tuple[int, int], DtypeVar]):
    ...


class Tensor(Array[Shape, DtypeVar]):
    ...


class ScalarArr(Array[Tuple[None], DtypeVar]):
    ...


Scalar = Union[Number, ScalarArr]

Numeric = TypeVar("Numeric", bound=Tensor)

Trainable = Union[Scalar, Vector, Matrix, Tensor]


class Batched(Generic[Numeric]):
    ...

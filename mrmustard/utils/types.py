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

from typing import *

# NOTE: when type-annotating with typevars, objects in a function signature with the same typevars must have the same type
# E.g. in `def f(x: Vector, y: Vector) -> Tensor: ...`
# the type of `x` and the type of `y` are assumed to be the same, even though "Vector" can mean different things.

Scalar = TypeVar("Scalar")  # rank 0
Vector = TypeVar("Vector")  # rank 1
Matrix = TypeVar("Matrix")  # rank 2
Tensor = TypeVar("Tensor")  # rank > 2
Array = TypeVar("Array")  # TODO: let mypy know that this is Vector, Matrix or Tensor
Trainable = TypeVar("Trainable")

Numeric = Union[Scalar, Vector, Matrix, Tensor]

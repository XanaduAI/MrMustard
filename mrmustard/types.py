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

# pylint: disable=unused-wildcard-import,wildcard-import

from typing import Any, Generic, TypeVar

import numpy as np

Scalar = Any
Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(np.ndarray, Generic[Shape, DType]):
    ...


# class Vector(MutableSequence, Generic[DType]):
#     ...


Vector = np.ndarray
Matrix = np.ndarray
Tensor = np.ndarray

Numeric = TypeVar("Numeric", bound=np.ndarray)


class Batch(np.ndarray, Generic[Numeric]):
    ...

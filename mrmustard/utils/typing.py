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

from typing import TypeVar

import numpy as np
from numpy._typing import _ArrayLikeInt_co
from numpy.typing import NDArray

__all__ = [
    "BoolScalar",
    "BoolScalarValue",
    "ComplexMatrix",
    "ComplexScalar",
    "ComplexScalarValue",
    "ComplexTensor",
    "ComplexVector",
    "IntArrayLike",
    "IntMatrix",
    "IntScalar",
    "IntScalarValue",
    "IntTensor",
    "IntVector",
    "Matrix",
    "RealMatrix",
    "RealScalar",
    "RealScalarValue",
    "RealTensor",
    "RealVector",
    "Scalar",
    "ScalarValue",
    "Tensor",
    "Trainable",
    "Vector",
]
# ArrayLike families
type IntArrayLike = _ArrayLikeInt_co

# Dtype families
type Array = NDArray[np.number]
type RealArray = NDArray[np.floating]
type ComplexArray = NDArray[np.complexfloating]
type IntArray = NDArray[np.signedinteger]
type BoolArray = NDArray[np.bool]

# Scalar values (true scalars, not "batched")
type ScalarValue = complex | float | int | np.number
type RealScalarValue = float | np.floating
type ComplexScalarValue = complex | np.complexfloating
type IntScalarValue = int | np.signedinteger
type BoolScalarValue = bool | np.bool

# Scalars (no core shape)
type Scalar = ScalarValue | Array
type RealScalar = RealScalarValue | RealArray
type ComplexScalar = ComplexScalarValue | ComplexArray
type IntScalar = IntScalarValue | IntArray
type BoolScalar = BoolScalarValue | BoolArray

# Vectors (core shape: (n, ))
type Vector = Array
type RealVector = RealArray
type ComplexVector = ComplexArray
type IntVector = IntArray

# Matrices (core shape: (n, m))
type Matrix = Array
type RealMatrix = RealArray
type ComplexMatrix = ComplexArray
type IntMatrix = IntArray

# Tensors (core shape: arbitrary rank)
type Tensor = Array
type RealTensor = RealArray
type ComplexTensor = ComplexArray
type IntTensor = IntArray

# Trainable
Trainable = TypeVar("Trainable", bound=NDArray[np.number])

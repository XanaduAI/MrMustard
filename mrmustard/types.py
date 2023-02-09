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

from typing import Any, Generic, List, Tuple, TypeVar

import numpy as np

# need to import all separately because otherwise
# aliased generic types assume their parameters are all Any
# from numpy.typing import NDArray as Batch
# from numpy.typing import NDArray as Matrix
from numpy.typing import _DType, _GenericAlias

Scalar = Any
Trainable = Any


Numeric = TypeVar("Numeric", bound=np.ndarray)

Vector = _GenericAlias(np.ndarray, (Tuple[int], _DType))
Matrix = _GenericAlias(np.ndarray, (Tuple[int, int], _DType))
Tensor = _GenericAlias(np.ndarray, (Tuple[int, ...], _DType))
# Batch = Union[List, np.ndarray]


class Batch(List, np.ndarray, Generic[Scalar]):
    ...

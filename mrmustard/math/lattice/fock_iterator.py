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

from typing import Tuple

import numpy as np

from mrmustard.math.lattice import recurrences, strategies
from mrmustard.types import Tensor

# class Fock_iterator:
#     def __init__(
#         self,
#         shape: Tuple[int],
#         strategy: Callable[[Vector], Generator[Vector, None, None]],
#         by_copy: bool = False,
#     ):
#         self.shape = shape
#         self.index_generator = strategy(shape)
#         self.index = np.zeros(len(shape), dtype=int)
#         self.done = False

#     def __iter__(self):
#         return self

#     def __next__(self) -> Vector:
#         if self.done:
#             raise StopIteration
#         else:
#             self.index = next(self.index_generator)
#             if np.any(self.index >= self.shape):
#                 self.done = True
#             if self.by_copy:
#                 return self.index.copy()
#             return self.index


def vanilla(shape: Tuple[int], A, b, c) -> Tensor:
    Z = np.zeros(shape, dtype=np.complex128)
    Z[(0,) * len(shape)] = c
    path = strategies.ndindex_gen(shape)
    next(path)  # skip the zero index
    for index in path:
        print(index)
        Z[index] = recurrences.vanilla_step(Z, A, b, index)
    return Z

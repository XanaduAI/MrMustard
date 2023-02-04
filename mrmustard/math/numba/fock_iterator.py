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

from typing import Callable, Generator, Tuple

import numpy as np
from numpy.types import Int1D


class Fock_iterator:
    def __init__(
        self,
        shape: Tuple[int],
        strategy: Callable[[Int1D], Generator[Int1D, None, None]],
        by_copy: bool = False,
    ):
        self.shape = shape
        self.index_generator = strategy(shape)
        self.index = np.zeros(len(shape), dtype=int)
        self.done = False

    def __iter__(self):
        return self

    def __next__(self) -> Int1D:
        if self.done:
            raise StopIteration
        else:
            self.index = next(self.index_generator)
            if np.any(self.index >= self.shape):
                self.done = True
            if self.by_copy:
                return self.index.copy()
            return self.index

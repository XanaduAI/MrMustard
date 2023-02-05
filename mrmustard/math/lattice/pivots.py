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
#

from numba import njit

from mrmustard.types import Vector


@njit
def vanilla_pivot_fn(index: Vector) -> Vector:
    r"returns the first available pivot index for the given index"
    for i, v in enumerate(index):
        if v > 0:
            index[i] -= 1
            return index
    raise ValueError("Index is zero")

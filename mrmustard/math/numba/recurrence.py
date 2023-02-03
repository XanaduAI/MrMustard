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

from typing import Callable, Tuple

import numpy as np


def fock_bargmann_recurrence(
    A, b, index: Tuple, pivot_fn: Callable, neighbors_fn: Callable
) -> complex:
    pivot = pivot_fn(index)
    A = A * np.sqrt(np.asarray(pivot))[None, :] / np.sqrt(np.asarray(pivot) + 1)[:, None]
    b = b / np.sqrt(np.asarray(pivot) + 1)

    neighbors = neighbors_fn(pivot)
    return b * pivot + A @ neighbors

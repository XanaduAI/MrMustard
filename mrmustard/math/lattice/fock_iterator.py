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


def vanilla(shape: Tuple[int], A, b, c) -> Tensor:
    print("[vanilla] vanilla called")
    Z = np.zeros(shape, dtype=np.complex128)
    Z[(0,) * len(shape)] = c
    print("[vanilla] initializing path...")
    path = strategies.ndindex_iter(np.asarray(shape))
    print("[vanilla] path initialized.")
    print("[vanilla] calling next on path...")
    skip = next(path)  # skip the zero index
    print("[vanilla] skipped index", skip)
    for index in path:
        print("[vanilla] got index for vanilla_step:", index)
        print("[vanilla] calling vanilla_step...")
        val_at_index = recurrences.vanilla_step(Z, A, b, index)
        print("[vanilla] vanilla_step returned", val_at_index)
        print(f"[vanilla] setting Z[{tuple(index)}] to", val_at_index)
        Z[tuple(index)] = val_at_index
    return Z

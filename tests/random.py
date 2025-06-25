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

import numpy as np

from mrmustard import settings


def Abc_triple(n: int, batch: tuple[int, ...] = ()):
    r"""
    Produces a random ``(A, b, c)`` triple for ``n`` modes.
    """
    min_magnitude = 1e-9
    max_magnitude = 1
    rng = settings.rng
    # complex symmetric matrix A
    A = rng.uniform(min_magnitude, max_magnitude, (*batch, n, n)) + 1.0j * rng.uniform(
        min_magnitude,
        max_magnitude,
        (*batch, n, n),
    )
    A = 0.5 * (A + np.swapaxes(A, -2, -1))  # make it symmetric

    # complex vector b
    b = rng.uniform(min_magnitude, max_magnitude, (*batch, n)) + 1.0j * rng.uniform(
        min_magnitude,
        max_magnitude,
        (*batch, n),
    )

    # complex scalar c
    c = rng.uniform(min_magnitude, max_magnitude, (*batch,)) + 1.0j * rng.uniform(
        min_magnitude,
        max_magnitude,
        (*batch,),
    )

    return A, b, c

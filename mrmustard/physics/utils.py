# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the utility functions used by the classes in ``mrmustard.physics``.
"""
from __future__ import annotations

from mrmustard import math


#  ~~~~~~~~~
#  Utilities
#  ~~~~~~~~~


def compute_batch_shape(*args) -> tuple[tuple[int, ...], int]:
    r"""
    Compute the final batch shape of the input arguments.

    Args:
        *args: The input arguments.

    Returns:
        The final batch shape and batch dimension of the input arguments.
    """
    batch_shape = None
    for arg in args:
        arg = math.astensor(arg)
        if arg.shape:
            if batch_shape is None:
                batch_shape = arg
            else:
                batch_shape = batch_shape * arg
    batch_shape = batch_shape.shape if batch_shape is not None else ()
    return batch_shape, len(batch_shape or (1,))

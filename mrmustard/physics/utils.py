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

from mrmustard.utils.typing import ComplexMatrix, ComplexVector, ComplexTensor


#  ~~~~~~~~~
#  Utilities
#  ~~~~~~~~~


def verify_batch_triple(
    A: ComplexMatrix | None, b: ComplexVector | None, c: ComplexTensor | None
) -> None:
    r"""
    Verify that the batch dimensions of the (A, b, c) triple are consistent.

    Args:
        A: The matrix of the quadratic form.
        b: The vector of the linear form.
        c: The scalar of the quadratic form.

    Raises:
        ValueError: If the batch dimensions of the (A, b, c) triple are inconsistent.
    """
    if A is None and b is None and c is None:
        return
    batch = A.shape[:-2]
    batch_dim = len(batch)

    if batch != b.shape[:batch_dim] or (len(c.shape) != 0 and batch != c.shape[:batch_dim]):
        raise ValueError(
            f"Batch dimensions of the first triple ({batch}, {b.shape[:batch_dim]}, {c.shape[:batch_dim]}) are inconsistent."
        )

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

"""Tests for indices"""

import numpy as np
import pytest

from mrmustard.math.lattice.strategies.flat_indices import first_available_pivot, shape_to_strides


def test_first_available_pivot():
    strides1 = shape_to_strides(np.array([2, 2, 2]))

    with pytest.raises(ValueError, match="zero"):
        first_available_pivot(0, strides1)
    assert first_available_pivot(1, strides1) == (2, 0)
    assert first_available_pivot(2, strides1) == (1, 0)
    assert first_available_pivot(3, strides1) == (1, 1)
    assert first_available_pivot(4, strides1) == (0, 0)
    assert first_available_pivot(5, strides1) == (0, 1)
    assert first_available_pivot(6, strides1) == (0, 2)
    assert first_available_pivot(7, strides1) == (0, 3)
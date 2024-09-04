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

"""Tests for flat indices"""

import numpy as np
import pytest

from mrmustard.math.lattice.strategies.flat_indices import (
    first_available_pivot,
    lower_neighbors,
    shape_to_strides,
)


def test_shape_to_strides():
    r"""
    Tests the ``shape_to_strides`` method.
    """
    shape1 = np.array([2])
    strides1 = np.array([1])
    assert np.allclose(shape_to_strides(shape1), strides1)

    shape2 = np.array([1, 2])
    strides2 = np.array([2, 1])
    assert np.allclose(shape_to_strides(shape2), strides2)

    shape3 = np.array([4, 5, 6])
    strides3 = np.array([30, 6, 1])
    assert np.allclose(shape_to_strides(shape3), strides3)


def test_first_available_pivot():
    r"""
    Tests the ``first_available_pivot`` method.
    """
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


def test_lower_neighbors():
    r"""
    Tests the ``lower_neighbors`` method.
    """
    strides = shape_to_strides(np.array([2, 2, 2]))

    assert list(lower_neighbors(1, strides, 0)) == [(0, -3), (1, -1), (2, 0)]
    assert list(lower_neighbors(1, strides, 1)) == [(1, -1), (2, 0)]
    assert list(lower_neighbors(1, strides, 2)) == [(2, 0)]

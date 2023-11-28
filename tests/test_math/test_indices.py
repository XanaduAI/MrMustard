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

from mrmustard.math.lattice.strategies.indices import FlatIndex

class TestFlatIndex:
    def test_init(self):
        shape = np.array([2, 2, 3])
        value = 1
        index = FlatIndex(shape, value)

        assert np.allclose(index.shape, shape)
        assert index.value == value
        assert index.range == 12
        assert np.allclose(index.strides, np.array([6, 3, 1]))

    @pytest.mark.parametrize("value", [12, 102])
    def test_init_error(self, value):
        shape = np.array([2, 2, 3])

        with pytest.raises(ValueError, match="out of range"):
            FlatIndex(shape, value)

    def test_increment(self):
        index = FlatIndex(np.array([2, 2, 2]), 0)
        for i in range(1, index.range):
            index.increment()
            assert index.value == i

    def test_increment_error(self):
        index = FlatIndex(np.array([2, 2, 2]), 7)

        with pytest.raises(ValueError, match="cannot be incremented"):
            index.increment()

    def test_first_available_pivot(self):
        index1 = FlatIndex(np.array([2, 2, 2]), 1)
        i1, pivot1 = index1.first_available_pivot() 
        assert i1 == 2 
        assert pivot1.value == 0

        index2 = FlatIndex(np.array([2, 2, 2]), 4)
        i2, pivot2 = index2.first_available_pivot() 
        assert i2 == 0
        assert pivot1.value == 0

        index3 = FlatIndex(np.array([2, 2, 2]), 7)
        i3, pivot3 = index3.first_available_pivot() 
        assert i3 == 0 
        assert pivot3.value == 3
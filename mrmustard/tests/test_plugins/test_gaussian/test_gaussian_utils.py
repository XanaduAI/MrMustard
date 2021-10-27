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
from hypothesis import given, strategies as st
from mrmustard import *
from mrmustard.plugins import gaussian as gp


def test_partition_means():
    A, B = gp.partition_means(gp.backend.astensor(np.array([1, 2, 3, 4, 5, 6])), Amodes=[0, 2])
    assert np.allclose(A, [1, 3, 4, 6])
    assert np.allclose(B, [2, 5])

    A, B = gp.partition_means(gp.backend.astensor(np.array([1, 2, 3, 4, 5, 6])), Amodes=[0])
    assert np.allclose(A, [1, 4])
    assert np.allclose(B, [2, 3, 5, 6])

    A, B = gp.partition_means(gp.backend.astensor(np.array([1, 2, 3, 4, 5, 6])), Amodes=[1])
    assert np.allclose(A, [2, 5])
    assert np.allclose(B, [1, 3, 4, 6])


def test_partition_cov_2modes():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    A, B, AB = gp.partition_cov(gp.backend.astensor(arr), Amodes=[0, 1])
    assert np.allclose(A, arr)
    assert np.allclose(B, [])
    assert np.allclose(AB, [])

    A, B, AB = gp.partition_cov(gp.backend.astensor(arr), Amodes=[0])
    assert np.allclose(A, [[1, 3], [9, 11]])
    assert np.allclose(B, [[6, 8], [14, 16]])
    assert np.allclose(AB, [[2, 4], [10, 12]])

    A, B, AB = gp.partition_cov(gp.backend.astensor(arr), Amodes=[1])
    assert np.allclose(A, [[6, 8], [14, 16]])
    assert np.allclose(B, [[1, 3], [9, 11]])
    assert np.allclose(AB, [[5, 7], [13, 15]])  # effectively BA because A is mode 1


def test_partition_cov_3modes():
    pass  # TODO

    # arr = np.array([[1,2,3,4,5,6],
    #                 [7,8,9,10,11,12],
    #                 [13,14,15,16,17,18],
    #                 [19,20,21,22,23,24],
    #                 [25,26,27,28,29,30],
    #                 [31,32,33,34,35,36]])
    # A,B,AB = gp.partition_cov(gp.backend.astensor(arr), Amodes=[0,2])

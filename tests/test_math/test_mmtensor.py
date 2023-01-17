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

"""
Unit tests for the :class:`MMTensor`.
"""
import pytest
from mrmustard.math.mmtensor import MMTensor
from mrmustard.math import Math

math = Math()
import numpy as np


def test_mmtensor_creation():
    """Test creation of MMTensor"""
    array = np.array([[1, 2, 3]])
    mmtensor = MMTensor(array)
    assert isinstance(mmtensor, MMTensor)
    assert isinstance(mmtensor.tensor, np.ndarray)
    assert mmtensor.axis_labels == ["0", "1"]


def test_mmtensor_creation_using_mmtensor():
    """Test creation of MMTensor using MMTensor"""
    array = np.array([[1, 2, 3]])
    mmtensor = MMTensor(array)
    mmtensor2 = MMTensor(mmtensor)
    assert isinstance(mmtensor2, MMTensor)
    assert mmtensor2.tensor is mmtensor.tensor
    assert mmtensor2.axis_labels == ["0", "1"]


def test_mmtensor_creation_with_axis_labels():
    """Test creation of MMTensor with axis labels"""
    array = np.array([[[1, 2, 3]]])
    mmtensor = MMTensor(array, axis_labels=["a", "b", "c"])
    assert isinstance(mmtensor, MMTensor)
    assert isinstance(mmtensor.tensor, np.ndarray)
    assert mmtensor.axis_labels == ["a", "b", "c"]


def test_mmtensor_creation_with_axis_labels_wrong_length():
    """Test creation of MMTensor with axis labels of wrong length"""
    array = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        MMTensor(array, axis_labels=["a", "b"])


def test_mmtensor_transposes_labels_too():
    """Test that MMTensor transposes axis labels"""
    array = np.array([[1, 2, 3], [4, 5, 6]])
    mmtensor = MMTensor(array, axis_labels=["a", "b"])
    mmtensor = mmtensor.transpose([1, 0])
    assert mmtensor.axis_labels == ["b", "a"]


def test_mmtensor_contract():
    """Test that MMTensor contracts correctly"""
    array = np.array([[1, 2], [3, 4]])
    trace = MMTensor(array, axis_labels=["a", "a"]).contract().tensor
    assert trace == 5


def test_mmtensor_contract_multiple_indices():
    """Test that MMTensor contracts multiple indices correctly"""
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    mmtensor = MMTensor(array, axis_labels=["a", "a", "b"])
    trace = mmtensor.contract()
    assert np.allclose(trace.tensor, np.einsum("aab", array))


def test_mmtensor_matmul():
    """Test that MMTensor objects contract correctly using the @ operator"""
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6], [7, 8]])
    T1 = MMTensor(array1, axis_labels=["a", "b"])
    T2 = MMTensor(array2, axis_labels=["b", "c"])
    contracted_tensor = T1 @ T2
    assert np.allclose(contracted_tensor.tensor, np.array([[19, 22], [43, 50]]))


def test_mmtensor_matmul_high_rank():
    """Test that higher-rank MMTensor objects contract correctly using the @ operator"""
    array1 = np.random.normal(size=(2, 3, 4, 5))
    array2 = np.random.normal(size=(5, 4, 3, 2))
    T1 = MMTensor(array1, axis_labels=["a", "b", "c", "d"])
    T2 = MMTensor(array2, axis_labels=["d", "c", "b", "e"])
    contracted_tensor = T1 @ T2
    assert np.allclose(contracted_tensor.tensor, np.einsum("abcd,dcbe->ae", array1, array2))

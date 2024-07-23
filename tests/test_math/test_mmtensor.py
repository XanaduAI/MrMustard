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
import numpy as np
import pytest

from mrmustard.math.tensor_wrappers.mmtensor import MMTensor


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


def test_mmtensor_getitem_slice():
    """Test that MMTensor slices correctly"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    sliced = mmtensor[0, :, 0]
    assert sliced.axis_labels == ["1"]
    assert np.allclose(sliced, array[0, :, 0])


def test_mmtensor_getitem_int():
    """Test that MMTensor slices correctly"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    sliced = mmtensor[0, 0, 0]
    assert sliced.axis_labels == []
    assert np.allclose(sliced, array[0, 0, 0])


def test_mmtensor_getitem_ellipsis_beginning():
    """Test that MMTensor slices correctly"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    sliced = mmtensor[..., 2]
    assert mmtensor[..., 2].axis_labels == ["0", "1"]
    assert np.allclose(sliced, array[..., 2])


def test_ufunc():
    """Test that MMTensor ufuncs work"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    assert np.allclose(np.sin(mmtensor), np.sin(array))


def test_mmtensor_algebra_add():
    """Test that MMTensor addition works"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    assert np.allclose(mmtensor + mmtensor, array + array)


def test_mmtensor_algebra_add_different_labels():
    """Test that MMTensor addition with different labels raises error"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor1 = MMTensor(array, axis_labels=["0", "1", "2"])
    mmtensor2 = MMTensor(array, axis_labels=["0", "1", "3"])
    with pytest.raises(ValueError):
        mmtensor1 + mmtensor2  # pylint: disable=pointless-statement


def test_mmtensor_algebra_subtract():
    """Test that MMTensor subtraction works"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    assert np.allclose(mmtensor - mmtensor, array - array)


def test_mmtensor_algebra_subtract_different_labels():
    """Test that MMTensor subtraction with different labels raises error"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor1 = MMTensor(array, axis_labels=["0", "1", "2"])
    mmtensor2 = MMTensor(array, axis_labels=["0", "1", "3"])
    with pytest.raises(ValueError):
        mmtensor1 - mmtensor2  # pylint: disable=pointless-statement


def test_mmtensor_algebra_multiply():
    """Test that MMTensor multiplication works"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    assert np.allclose(mmtensor * mmtensor, array * array)


def test_mmtensor_algebra_multiply_different_labels():
    """Test that MMTensor multiplication with different labels raises error"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor1 = MMTensor(array, axis_labels=["0", "1", "2"])
    mmtensor2 = MMTensor(array, axis_labels=["0", "1", "3"])
    with pytest.raises(ValueError):
        mmtensor1 * mmtensor2  # pylint: disable=pointless-statement


def test_mmtensor_algebra_divide():
    """Test that MMTensor division works"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor = MMTensor(array, axis_labels=["0", "1", "2"])
    assert np.allclose(mmtensor / mmtensor, array / array)


def test_mmtensor_algebra_divide_different_labels():
    """Test that MMTensor division with different labels raises error"""
    array = np.random.normal(size=(2, 3, 4))
    mmtensor1 = MMTensor(array, axis_labels=["0", "1", "2"])
    mmtensor2 = MMTensor(array, axis_labels=["0", "1", "3"])
    with pytest.raises(ValueError):
        mmtensor1 / mmtensor2  # pylint: disable=pointless-statement


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
    assert np.allclose(
        contracted_tensor.tensor, np.einsum("abcd,dcbe->ae", array1, array2)
    )

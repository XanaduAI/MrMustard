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

from typing import get_args, get_origin

import numpy as np

from mrmustard.utils.typing import (
    Batch,
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    IntMatrix,
    IntTensor,
    IntVector,
    RealMatrix,
    RealTensor,
    RealVector,
    UIntMatrix,
    UIntTensor,
    UIntVector,
)


def test_complexvector():
    vec: ComplexVector = np.array([1.0 + 1.0j])
    assert isinstance(vec, get_origin(ComplexVector))
    assert isinstance(vec[0], get_args(ComplexVector)[1].__constraints__)


def test_realvector():
    vec: RealVector = np.array([1.0, 2.0, 3.0])
    assert isinstance(vec, get_origin(RealVector))
    assert isinstance(vec[0], get_args(RealVector)[1].__constraints__)


def test_intvector():
    vec: IntVector = np.array([1, 2, 3])
    assert isinstance(vec, get_origin(IntVector))
    assert isinstance(vec[0], get_args(IntVector)[1].__constraints__)


def test_uintvector():
    vec: UIntVector = np.array([1, 2, 3], dtype=np.uint32)
    assert isinstance(vec, get_origin(UIntVector))
    assert isinstance(vec[0], get_args(UIntVector)[1].__constraints__)


def test_complexmatrix():
    mat: ComplexMatrix = np.array([[1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j]])
    assert isinstance(mat, get_origin(ComplexMatrix))
    assert isinstance(mat[0, 0], get_args(ComplexMatrix)[1].__constraints__)


def test_realmatrix():
    mat: RealMatrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert isinstance(mat, get_origin(RealMatrix))
    assert isinstance(mat[0, 0], get_args(RealMatrix)[1].__constraints__)


def test_intmatrix():
    mat: IntMatrix = np.array([[1, 2, 3], [4, 5, 6]])
    assert isinstance(mat, get_origin(IntMatrix))
    assert isinstance(mat[0, 0], get_args(IntMatrix)[1].__constraints__)


def test_uintmatrix():
    mat: UIntMatrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    assert isinstance(mat, get_origin(UIntMatrix))
    assert isinstance(mat[0, 0], get_args(UIntMatrix)[1].__constraints__)


def test_complextensor():
    ten: ComplexTensor = np.array(
        [[[1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j], [4.0 + 4.0j, 5.0 + 5.0j, 6.0 + 6.0j]]],
    )
    assert isinstance(ten, get_origin(ComplexTensor))
    assert isinstance(ten[0, 0, 0], get_args(ComplexTensor)[1].__constraints__)


def test_realtensor():
    ten: RealTensor = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    assert isinstance(ten, get_origin(RealTensor))
    assert isinstance(ten[0, 0, 0], get_args(RealTensor)[1].__constraints__)


def test_inttensor():
    ten: IntTensor = np.array([[[1, 2, -3], [4, 5, -6]], [[7, 8, 9], [10, 11, 12]]])
    assert isinstance(ten, get_origin(IntTensor))
    assert isinstance(ten[0, 0, 0], get_args(IntTensor)[1].__constraints__)


def test_uinttensor():
    ten: UIntTensor = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint32)
    assert isinstance(ten, get_origin(UIntTensor))
    assert isinstance(ten[0, 0, 0], get_args(UIntTensor)[1].__constraints__)


def test_batch():
    batch: Batch[RealVector] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # remember batch is a protocol, so we can't use isinstance
    assert issubclass(type(batch), Batch)
    assert isinstance(batch[0][0], get_args(get_args(Batch[RealVector])[0])[1].__constraints__)

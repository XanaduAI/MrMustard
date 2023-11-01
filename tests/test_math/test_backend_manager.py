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
from tensorflow.linalg import diag as tfdiag

import mrmustard.math as math


class TestBackendManager:
    l1 = [1.]
    l2 = [1., 2. + 2.j]
    l3 = [[1., 2.], [3., 4.]]
    l4 = [l2, l2]
    l5 = [[[1., 2., 3. + 6.j], [3., 4., 5. - 10.j]], [[1., 2. + 1.j, 3.], [3., 4., 5.]]]
    lists = [l1, l2, l3, l4, l5]


    @pytest.mark.parametrize("l", lists)
    def test_abs(self, l):
        arr = np.array(l)
        res = math.asnumpy(math.abs(np.array(l)))
        assert np.allclose(res, np.abs(arr))

    @pytest.mark.parametrize("l", lists)
    def test_any(self, l):
        arr = np.array(l)
        assert np.allclose(math.asnumpy(math.any(arr)), np.any(arr))

    def test_arange(self):
        params = (3, 20, 0.5)
        res = math.asnumpy(math.arange(*params))
        assert np.allclose(res, np.arange(*params))

    @pytest.mark.parametrize("l", lists)
    def test_asnumpy(self, l):
        arr = np.array(l)
        assert np.allclose(arr, math.asnumpy(arr))

    def test_assign(self):
        pass

    def test_atleast_1d(self):
        l1 = 1
        l2 = [1, 2, 20, -1]
        l3 = [[1, 2], [3, 4]]
        ls = [l1, l2, l3]

        for l in ls:
            res = math.asnumpy(math.atleast_1d(l))
            assert np.allclose(res, np.atleast_1d(l))

    # @pytest.mark.parametrize("t", types)
    # def test_cast(self, t):
    #     arr = np.array([1, 2, 3])
    #     assert np.allclose(arr, math.cast(arr, t))
    #     assert math.cast(arr, t).dtype is t or arr.dtype

    #     fl = 1.0
    #     assert fl == math.cast(fl, t)
    #     assert type(math.cast(fl, t)) is t or type(fl)
    #     assert not isinstance(fl, np.ndarray)

    @pytest.mark.parametrize("l", lists)
    def test_clip(self, l):
        arr = np.array(l)
        params = (arr, 0, 3)
        res = math.asnumpy(math.clip(*params))
        assert np.allclose(res, np.clip(*params))

    @pytest.mark.parametrize("a", [0, 1])
    def test_concat(self, a):
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6]]) if a != 1 else np.array([[5], [6]])
        params = ((arr1, arr2), a)
        res = math.asnumpy(math.concat(*params))
        return np.allclose(res, np.concatenate(*params))

    @pytest.mark.parametrize("l", lists)
    def test_conj(self, l):
        arr = np.array(l)
        res = math.asnumpy(math.conj(arr))
        assert np.allclose(res, np.conj(arr))

    @pytest.mark.parametrize("l", lists)
    def test_trigonometry(self, l):
        arr = np.array(l)
        assert np.allclose(math.asnumpy(math.cos(arr)), np.cos(arr))
        assert np.allclose(math.asnumpy(math.cosh(arr)), np.cosh(arr))
        assert np.allclose(math.asnumpy(math.sin(arr)), np.sin(arr))
        assert np.allclose(math.asnumpy(math.sinh(arr)), np.sinh(arr))
        # assert np.allclose(math.atan2(arr, 3), np.arctan(arr, 3))

    def test_make_complex(self):
        pass

    def test_det(self):
        pass

    @pytest.mark.parametrize("k", [-8, 1, 0, 1, 8])
    @pytest.mark.parametrize("l", lists)
    def test_diag(self, l, k):
        arr = np.array(l)

        ret_math = math.diag(arr, k)
        ret_tf = tfdiag(arr, k=k).numpy()
        assert np.allclose(ret_math, ret_tf)

    def test_diag_part(self):
        # ??
        pass

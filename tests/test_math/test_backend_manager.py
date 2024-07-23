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
Unit tests for the :class:`BackendManager`.
"""
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import tensorflow as tf

from mrmustard import math
from ..conftest import skip_np


# pylint: disable=protected-access, too-many-public-methods
class TestBackendManager:
    r"""
    Tests the BackendManager.
    """

    l1 = [1.0]
    l2 = [1.0 + 0.0j, -2.0 + 2.0j]
    l3 = [[1.0, 2.0], [-3.0, 4.0]]
    l4 = [l2, l2]
    l5 = [
        [[1.0, 2.0, 3.0 + 6], [3.0, 4.0, 5.0 - 10]],
        [[1.0, 2.0 + 1, 3.0], [3.0, 4.0, 5.0]],
    ]
    lists = [l1, l2, l3, l4, l5]

    types = ["None", "int32", "float32", "float64", "complex128"]

    def test_error(self):
        r"""
        Tests the error on `_apply`.
        """
        msg = f"Function ``ciao`` not implemented for backend ``{math.backend_name}``."
        with pytest.raises(NotImplementedError, match=msg):
            math._apply("ciao")

    def test_types(self):
        r"""
        Tests the types.
        """
        assert math.int32 is math.backend.int32
        assert math.float32 is math.backend.float32
        assert math.float64 is math.backend.float64
        assert math.complex128 is math.backend.complex128

    @pytest.mark.parametrize("l", lists)
    def test_abs(self, l):
        r"""
        Tests the ``abs`` method.
        """
        arr = np.array(l)
        res = math.asnumpy(math.abs(np.array(l)))
        assert np.allclose(res, np.abs(arr))

    def test_allclose(self):
        r"""
        Tests the ``allclose`` method.
        """
        arr1 = [1, 2, 3]
        arr2 = [1, 2, 3]
        arr3 = [1.01, 2, 3]
        arr4 = [2, 3, 1]

        assert math.allclose(arr1, arr2)
        assert not math.allclose(arr1, arr3)
        assert math.allclose(arr1, arr3, 1e-2)
        assert not math.allclose(arr1, arr4)

    def test_allclose_error(self):
        r"""
        Tests the error of ``allclose`` method.
        """
        arr1 = [1, 2, 3]
        arr2 = [[1, 2, 3]]

        with pytest.raises(ValueError, match="Cannot compare"):
            math.allclose(arr1, arr2)

    @pytest.mark.parametrize("l", lists)
    def test_any(self, l):
        r"""
        Tests the ``any`` method.
        """
        arr = np.array(l)
        assert np.allclose(math.asnumpy(math.any(arr)), np.any(arr))

    @pytest.mark.parametrize("t", ["float32", "float64"])
    def test_arange(self, t):
        r"""
        Tests the ``arange`` method.
        """
        dtype = getattr(math, t)
        params = (3, 20, 0.5, dtype)

        np_dtype = getattr(np, t)
        np_params = (3, 20, 0.5, np_dtype)

        res = math.asnumpy(math.arange(*params))
        assert np.allclose(res, np.arange(*np_params))

    @pytest.mark.parametrize("l", lists)
    def test_asnumpy(self, l):
        r"""
        Tests the ``asnumpy`` method.
        """
        arr = math.astensor(np.array(l), dtype=np.array(l).dtype)
        res = math.asnumpy(arr)
        assert np.allclose(res, np.array(l))

    def test_assign(self):
        r"""
        Tests the ``assign`` method.
        """
        arr = math.new_variable(np.eye(3), (None, None), "")
        value = math.astensor(2 * np.eye(3))
        arr = math.asnumpy(math.assign(arr, value))
        assert np.allclose(arr, value)

    @pytest.mark.parametrize("t", types)
    @pytest.mark.parametrize("l", [l1, l3])
    def test_astensor(self, t, l):
        r"""
        Tests the ``astensor`` method.
        """
        arr = np.array(l)
        dtype = getattr(math, t, None)
        res = math.astensor(arr, dtype)

        if math.backend_name == "numpy":
            assert np.allclose(res, arr.astype(dtype or np.float64))
        else:
            exp = tf.convert_to_tensor(arr, dtype=dtype or tf.float64)
            exp = exp.numpy()
            assert np.allclose(res, exp)

    @pytest.mark.parametrize("t", types)
    @pytest.mark.parametrize("l", [l1, l3])
    def test_atleast_1d(self, t, l):
        r"""
        Tests the ``atleast_1d`` method.
        """
        dtype = getattr(math, t, None)
        arr = np.array(l)

        res = math.asnumpy(math.atleast_1d(arr, dtype=dtype))

        exp = np.atleast_1d(arr)
        if dtype:
            np_dtype = getattr(np, t, None)
            exp = exp.astype(np_dtype)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("t", types)
    @pytest.mark.parametrize("l", [l1, l3])
    def test_atleast_2d(self, t, l):
        r"""
        Tests the ``atleast_2d`` method.
        """
        dtype = getattr(math, t, None)
        arr = np.array(l)

        res = math.asnumpy(math.atleast_2d(arr, dtype=dtype))

        exp = np.atleast_2d(arr)
        if dtype:
            np_dtype = getattr(np, t, None)
            exp = exp.astype(np_dtype)

        assert np.allclose(res, exp)

    @pytest.mark.parametrize("t", types)
    @pytest.mark.parametrize("l", [l1, l3, l5])
    def test_atleast_3d(self, t, l):
        r"""
        Tests the ``atleast_3d`` method.
        """
        dtype = getattr(math, t, None)
        arr = np.array(l)

        res = math.asnumpy(math.atleast_3d(arr, dtype=dtype))

        if arr.ndim == 1:
            exp_shape = (1, 1) + arr.shape
        elif arr.ndim == 2:
            exp_shape = (1,) + arr.shape
        else:
            exp_shape = arr.shape
        assert res.shape == exp_shape

    def test_boolean_mask(self):
        r"""
        Tests the ``boolean_mask`` method.
        """
        arr = np.array([1, 2, 3, 4])
        mask = [True, False, True, True]
        res = math.asnumpy(math.boolean_mask(arr, mask))
        exp = np.array([1, 3, 4])
        assert np.allclose(res, exp)

    def test_block(self):
        r"""
        Tests the ``block`` method.
        """
        I = math.ones(shape=(4, 4), dtype=math.complex128)
        O = math.zeros(shape=(4, 4), dtype=math.complex128)
        R = math.block(
            [
                [I, 1j * I, O, O],
                [O, O, I, -1j * I],
                [I, -1j * I, O, O],
                [O, O, I, 1j * I],
            ]
        )
        assert R.shape == (16, 16)

    def test_block_diag(self):
        r"""
        Tests the ``block_diag`` method.
        """
        I = math.ones(shape=(4, 4), dtype=math.complex128)
        O = math.zeros(shape=(4, 4), dtype=math.complex128)
        R = math.block_diag(I, 1j * I)
        assert R.shape == (8, 8)
        assert np.allclose(math.block([[I, O], [O, 1j * I]]), R)

    @pytest.mark.parametrize("t", types)
    def test_cast(self, t):
        r"""
        Tests the ``cast`` method.
        """
        dtype = getattr(math, t, None)
        np_dtype = getattr(np, t, None)

        arr = np.array([[1, 2], [3, 4]])
        res = math.asnumpy(math.cast(arr, dtype))
        exp = arr.astype(np_dtype or np.float64)
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("l", [l1, l3])
    def test_clip(self, l):
        r"""
        Tests the ``clip`` method.
        """
        arr = np.array(l)
        params = (arr, 0, 3)
        res = math.asnumpy(math.clip(*params))
        assert np.allclose(res, np.clip(*params))

    @pytest.mark.parametrize("axis", [0, 1])
    def test_concat(self, axis):
        r"""
        Tests the ``concat`` method.
        """
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        params = ((arr1, arr2), axis)
        res = math.asnumpy(math.concat(*params))
        assert np.allclose(res, np.concatenate(*params))

    @pytest.mark.parametrize("l", lists)
    def test_conj(self, l):
        r"""
        Tests the ``conj`` method.
        """
        arr = np.array(l)
        res = math.asnumpy(math.conj(arr))
        assert np.allclose(res, np.conj(arr))

    @pytest.mark.parametrize("l", lists)
    def test_cos(self, l):
        r"""
        Tests the ``cos`` method.
        """
        arr = np.array(l)
        assert np.allclose(math.asnumpy(math.cos(arr)), np.cos(arr))

    @pytest.mark.parametrize("l", lists)
    def test_cosh(self, l):
        r"""
        Tests the ``cosh`` method.
        """
        arr = np.array(l)
        assert np.allclose(math.asnumpy(math.cosh(arr)), np.cosh(arr))

    def test_det(self):
        r"""
        Tests the ``det`` method.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(math.det(arr), -2.0)

    def test_diag(self):
        r"""
        Tests the ``diag`` method.
        """
        d1 = math.ones(shape=(3,), dtype=math.float64)
        d2 = 2 * math.ones(shape=(2,), dtype=math.float64)
        d3 = 3 * math.ones(shape=(1,), dtype=math.float64)

        res = math.diag(d1, 0) + math.diag(d2, 1) + math.diag(d3, 2)
        res = math.asnumpy(res)
        exp = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])

        assert np.allclose(res, exp)

    def test_diag_part(self):
        r"""
        Tests the ``diag_part`` method.
        """
        arr = np.arange(9).reshape((3, 1, 3))

        dp1 = math.asnumpy(math.diag_part(arr, 0))
        exp1 = np.array([[0], [3], [6]])
        assert np.allclose(dp1, exp1)

        dp2 = math.asnumpy(math.diag_part(arr, 1))
        exp2 = np.array([[1], [4], [7]])
        assert np.allclose(dp2, exp2)

        dp3 = math.asnumpy(math.diag_part(arr, 2))
        exp3 = np.array([[2], [5], [8]])
        assert np.allclose(dp3, exp3)

    def test_eigvals(self):
        r"""
        Tests the ``eigvals`` method.
        """
        arr = np.arange(9, dtype=np.float64).reshape((3, 3))
        ev = math.asnumpy(math.eigvals(arr))
        exp = np.array([1.33484692e01, -1.34846923e00, 0.0])

        ev.sort()
        exp.sort()
        assert np.allclose(ev, exp)

    def test_eigh(self):
        r"""
        Tests the ``eigh`` method.
        """
        arr = np.eye(3)
        arr[1, 1] = 2
        arr[2, 2] = 3
        vals, vecs = math.eigh(arr)

        assert np.allclose(math.asnumpy(vals), np.array([1.0, 2.0, 3.0]))
        assert np.allclose(math.asnumpy(vecs), np.eye(3))

    def test_exp(self):
        r"""
        Tests the ``exp`` method.
        """
        arr = np.eye(3)
        arr[1, 1] = 2
        arr[2, 2] = 3
        res = math.asnumpy(math.exp(arr))
        exp = np.array(
            [
                [np.exp(0) if i != j else np.exp(i + 1) for i in range(3)]
                for j in range(3)
            ]
        )
        assert np.allclose(res, exp)

    def test_eye(self):
        r"""
        Tests the ``eye`` method.
        """
        res = math.asnumpy(math.eye(3))
        exp = np.eye(3)
        assert np.allclose(res, exp)

    def test_eye_like(self):
        r"""
        Tests the ``eye_like`` method.
        """
        res = math.asnumpy(math.eye_like(math.zeros((3, 3))))
        exp = np.eye(3)
        assert np.allclose(res, exp)

    def test_from_backend(self):
        r"""
        Tests the ``expm`` method.
        """
        v1 = [1, 2]
        assert not math.from_backend(v1)

        v2 = np.array(v1)
        v3 = tf.constant(v1)
        if math.backend_name == "numpy":
            assert math.from_backend(v2) and not math.from_backend(v3)
        else:
            assert math.from_backend(v3) and not math.from_backend(v2)

    def test_gather(self):
        r"""
        Tests the ``gather`` method.
        """
        arr = np.arange(9).reshape((3, 3))

        res1 = math.asnumpy(math.gather(arr, 2, 1))
        exp1 = np.array([2, 5, 8])
        assert np.allclose(res1, exp1)

        res2 = math.asnumpy(math.gather(arr, 2, 0))
        exp2 = np.array([6, 7, 8])
        assert np.allclose(res2, exp2)

    def test_imag(self):
        r"""
        Tests the ``imag`` method.
        """
        arr = np.eye(3) + 2j * np.eye(3)
        assert np.allclose(math.asnumpy(math.imag(arr)), 2 * np.eye(3))

        assert np.allclose(math.asnumpy(math.imag(np.eye(3))), 0 * np.eye(3))

    def test_inv(self):
        r"""
        Tests the ``inv`` method.
        """
        arr = np.array([[1.0, 0], [0, 1j]])
        inv = math.inv(arr)
        assert np.allclose(math.asnumpy(arr @ inv), np.eye(2))

    def test_is_trainable(self):
        r"""
        Tests the ``is_trainable`` method.
        """
        arr1 = np.array([1, 2])
        arr2 = tf.constant(arr1)
        arr3 = tf.Variable(arr1)

        assert not math.is_trainable(arr1)
        assert not math.is_trainable(arr2)
        assert math.is_trainable(arr3) is (math.backend_name == "tensorflow")

    def test_lgamma(self):
        r"""
        Tests the ``lgamma`` method.
        """
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(math.asnumpy(math.lgamma(arr)), math.lgamma(arr))

    def test_log(self):
        r"""
        Tests the ``log`` method.
        """
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(math.asnumpy(math.log(arr)), np.log(arr))

    def test_make_complex(self):
        r"""
        Tests the ``make_complex`` method.
        """
        r = 1.0
        i = 2.0
        assert math.asnumpy(math.make_complex(r, i)) == r + i * 1j

    def test_maximum(self):
        r"""
        Tests the ``maximum`` method.
        """
        arr1 = np.eye(3)
        arr2 = 2 * np.eye(3)
        res = math.asnumpy(math.maximum(arr1, arr2))
        assert np.allclose(res, arr2)

    def test_minimum(self):
        r"""
        Tests the ``minimum`` method.
        """
        arr1 = np.eye(3)
        arr2 = 2 * np.eye(3)
        res = math.asnumpy(math.minimum(arr1, arr2))
        assert np.allclose(res, arr1)

    @pytest.mark.parametrize("t", types)
    def test_new_variable(self, t):
        r"""
        Tests the ``new_variable`` method.
        """
        dtype = getattr(math, t, None)
        arr = np.eye(3)
        res = math.new_variable(arr, (0, 1), "my_var", dtype)

        if math.backend_name == "numpy":
            assert np.allclose(res, arr)
            assert not hasattr(res, "name")
            assert res.dtype == dtype
        else:
            assert isinstance(res, tf.Variable)
            assert np.allclose(math.asnumpy(res), arr)
            assert res.dtype == dtype or math.float64

    @pytest.mark.parametrize("t", types)
    def test_new_constant(self, t):
        r"""
        Tests the ``new_constant`` method.
        """
        dtype = getattr(math, t, None)
        arr = np.eye(3)
        res = math.new_constant(arr, "my_const", dtype)

        if math.backend_name == "numpy":
            assert np.allclose(res, arr)
            assert not hasattr(res, "name")
            assert res.dtype == dtype
        else:
            assert np.allclose(math.asnumpy(res), arr)

    def test_ones(self):
        r"""
        Tests the ``ones`` method.
        """
        arr = np.ones(3)
        res = math.asnumpy(math.ones(3))
        assert np.allclose(res, arr)

    def test_ones_like(self):
        r"""
        Tests the ``ones_like`` method.
        """
        arr = np.ones(3)
        res = math.asnumpy(math.ones_like(arr))
        assert np.allclose(res, arr)

    def test_pow(self):
        r"""
        Tests the ``pow`` method.
        """
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(math.asnumpy(math.pow(arr, 2)), math.pow(arr, 2))

    def test_kron(self):
        r"""
        Tests the ``kron`` method.
        """
        t1 = math.astensor([[0, 1], [1, 0]])
        t2 = math.eye(2)
        kron = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
        assert np.allclose(math.kron(t1, t2), kron)

    def test_prod(self):
        r"""
        Tests the ``prod`` method.
        """
        assert np.allclose(math.prod([1]), 1)
        assert np.allclose(math.prod([[2, 2], [3, 3]], axis=0), [6, 6])
        assert np.allclose(math.prod([[2, 2], [3, 3]], axis=1), [4, 9])

    def test_real(self):
        r"""
        Tests the ``real`` method.
        """
        arr = np.eye(3) + 2j * np.eye(3)
        assert np.allclose(math.asnumpy(math.real(arr)), np.eye(3))

        assert np.allclose(math.asnumpy(math.real(np.eye(3))), np.eye(3))

    def test_reshape(self):
        r"""
        Tests the ``reshape`` method.
        """
        arr = np.eye(3)
        shape = (1, 9)
        arr = math.reshape(arr, shape)
        assert arr.shape == shape

    def test_set_diag(self):
        r"""
        Tests the ``set_diag`` method.
        """
        arr = np.zeros(shape=(3, 3))
        diag = np.ones(shape=(3,))
        assert np.allclose(math.asnumpy(math.set_diag(arr, diag, 0)), np.eye(3))

    @pytest.mark.parametrize("l", lists)
    def test_sin(self, l):
        r"""
        Tests the ``sin`` method.
        """
        arr = np.array(l)
        assert np.allclose(math.asnumpy(math.sin(arr)), np.sin(arr))

    @pytest.mark.parametrize("l", lists)
    def test_sinh(self, l):
        r"""
        Tests the ``sinh`` method.
        """
        arr = np.array(l)
        assert np.allclose(math.asnumpy(math.sinh(arr)), np.sinh(arr))

    def test_solve(self):
        r"""
        Tests the ``solve`` method.
        """
        arr = np.eye(3)
        vec = np.array([1.0, 2.0, 3.0])
        res = math.asnumpy(math.solve(arr, vec))
        assert np.allclose(arr @ res, vec)

    def test_sqrt(self):
        r"""
        Tests the ``sqrt`` method.
        """
        arr = 4 * np.eye(3)
        res = math.asnumpy(math.sqrt(arr))
        assert np.allclose(res, 2 * np.eye(3))

    def test_sqrtm(self):
        r"""
        Tests the ``sqrtm`` method.
        """
        arr = 4 * np.eye(3)
        res = math.asnumpy(math.sqrtm(arr))
        assert np.allclose(res, 2 * np.eye(3))

    def test_sum(self):
        r"""
        Tests the ``sum`` method.
        """
        arr = 4 * np.eye(3)
        res = math.asnumpy(math.sum(arr))
        assert np.allclose(res, 12)

    def test_categorical(self):
        r"""
        Tests the ``Categorical`` method.
        """
        probs = np.array([1e-6 for _ in range(300)])
        results = [math.Categorical(probs, "") for _ in range(100)]
        assert len(set(results)) > 1

    @patch("importlib.metadata.distribution")
    @patch("platform.processor")
    @patch("platform.system")
    def test_euclidean_opt_warning(self, mock_system, mock_processor, mock_version):
        """Test that a warning is raised for M1/M2 Mac users with TF 2.16+."""
        skip_np()
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "arm"
        mock_version.return_value = MagicMock(version="2.16.0")

        math._euclidean_opt = None  # just in case another test set it
        try:
            with pytest.warns(
                UserWarning, match=r"Mac.*please downgrade TensorFlow to 2.15"
            ):
                opt = math.euclidean_opt
            assert isinstance(opt, tf.keras.optimizers.Adam)
        finally:
            math._euclidean_opt = None  # reset after test passes

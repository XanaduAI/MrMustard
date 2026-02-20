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
Unit tests for :class:`Constant` and :class:`Variable`.
"""

import numpy as np
import pytest

from mrmustard.parameters.parameters import (
    Constant,
    Variable,
    format_dtype,
    format_value,
)


class TestConstant:
    r"""
    Tests for Constant.
    """

    def test_init(self):
        r"""
        Tests the init.
        """
        const1 = Constant(1, "const1")
        assert const1.value == 1
        assert const1.name == "const1"

        const2 = Constant(np.array([1, 2, 3]), "const2")
        assert np.allclose(const2.value, np.array([1, 2, 3]))

        const3 = Constant(1, "const3", dtype="int64")
        assert const3.value == 1
        assert const3.name == "const3"
        assert const3.value.dtype == "int64"

    @pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
    def test_format_dtype(self, dtype):
        r"""
        Tests the ``_format_dtype`` method.
        """
        const_dtype = Constant(dtype(1.0), f"const_{dtype}")
        dtype_str = format_dtype(const_dtype)
        assert dtype_str == dtype.__name__

    def test_format_value_arrays(self):
        r"""
        Tests the ``_format_value`` method with array parameters.
        """
        # Test small array integer-like (≤3 elements)
        const_small_int = Constant([1.0, 2.0, 3.0], "const_small_int")
        value_str, shape_str = format_value(const_small_int)
        assert value_str == "[1.0, 2.0, 3.0]"
        assert shape_str == "(3,)"

        # Test small array floats (≤3 elements)
        const_small_float = Constant([1.2, 3.4, 5.6], "const_small_float")
        value_str, shape_str = format_value(const_small_float)
        assert value_str == "[1.2, 3.4, 5.6]"
        assert shape_str == "(3,)"

        # Test large array integer-like (>3 elements)
        const_large_int = Constant([1, 2, 3, 4, 5, 6], "const_large_int")
        value_str, shape_str = format_value(const_large_int)
        assert "1, 2, 3, ..." in value_str
        assert shape_str == "(6,)"

        # Test large array floats (>3 elements)
        const_large_float = Constant([1.2, 3.4, 5.6, 7.8, 9.0, 10.1], "const_large_float")
        value_str, shape_str = format_value(const_large_float)
        assert "1.2, 3.4, 5.6, ..." in value_str
        assert shape_str == "(6,)"

        # Test 2D array (gets flattened for display since it has >3 elements)
        const_2d = Constant([[1, 2], [3, 4]], "const_2d")
        value_str, shape_str = format_value(const_2d)
        assert "1, 2, 3, ..." in value_str  # Flattened array with >3 elements
        assert shape_str == "(2, 2)"

        # Test small 2D array (≤3 elements when flattened)
        const_2d_small = Constant([[1, 2]], "const_2d_small")
        value_str, shape_str = format_value(const_2d_small)
        assert value_str == "[[1, 2]]"
        assert shape_str == "(1, 2)"

        # Test empty array
        const_empty = Constant([], "const_empty")
        value_str, shape_str = format_value(const_empty)
        assert value_str == "[]"
        assert shape_str == "(0,)"

    def test_format_value_scalar(self):
        r"""
        Tests the ``_format_value`` method with scalar parameters.
        """
        # Test scalar real integer constant
        const_real_int = Constant(3, "const_real_int", dtype=np.int64)
        value_str, shape_str = format_value(const_real_int)
        assert value_str == "3"
        assert shape_str == "scalar"

        # Test scalar real float constant
        const_real_float = Constant(3.14159, "const_real_float", dtype=np.float64)
        value_str, shape_str = format_value(const_real_float)
        assert value_str == "3.14159"
        assert shape_str == "scalar"

        # Test scalar complex constant positive imaginary part
        const_complex_pos_imag = Constant(1 + 2j, "const_complex_pos_imag", dtype=np.complex128)
        value_str, shape_str = format_value(const_complex_pos_imag)
        assert value_str == "1+2j"
        assert shape_str == "scalar"

        # Test scalar complex constant negative imaginary part
        const_complex_neg_imag = Constant(1 - 2j, "const_complex_neg_imag", dtype=np.complex128)
        value_str, shape_str = format_value(const_complex_neg_imag)
        assert value_str == "1-2j"
        assert shape_str == "scalar"

    def test_is_const(self):
        r"""
        Tests that constants are immutable.
        """
        const = Constant(1, "const")

        with pytest.raises(AttributeError):
            const.value = 2

    def test_name_can_be_changed(self):
        r"""
        Tests that the name of a constant can be changed.
        """
        const = Constant(1, "const")
        assert const.name == "const"
        const.name = "new_name"
        assert const.name == "new_name"


class TestVariable:
    r"""
    Tests for Variable.
    """

    def test_init(self):
        r"""
        Tests the init.
        """
        var1 = Variable(1, "var1")
        assert var1.value == 1
        assert var1.name == "var1"
        assert var1.update_fn == "update_euclidean"

        var2 = Variable(np.array([1, 2, 3]), "var2", update_fn="update_orthogonal")
        assert np.allclose(var2.value, np.array([1, 2, 3]))
        assert var2.update_fn == "update_orthogonal"

        var3 = Variable(1, "var3", dtype="int64")
        assert var3.value == 1
        assert var3.name == "var3"
        assert var3.value.dtype == "int64"

    @pytest.mark.parametrize("dtype", [np.int64, np.float64, np.complex128])
    def test_format_dtype(self, dtype):
        r"""
        Tests the ``_format_dtype`` method.
        """
        var_dtype = Variable(dtype(1.0), f"const_{dtype}")
        dtype_str = format_dtype(var_dtype)
        assert dtype_str == dtype.__name__

    def test_format_value_arrays(self):
        r"""
        Tests the ``_format_value`` method with array parameters.
        """
        # Test small array integer-like (≤3 elements)
        const_small_int = Variable([1.0, 2.0, 3.0], "const_small_int")
        value_str, shape_str = format_value(const_small_int)
        assert value_str == "[1.0, 2.0, 3.0]"
        assert shape_str == "(3,)"

        # Test small array floats (≤3 elements)
        const_small_float = Variable([1.2, 3.4, 5.6], "const_small_float")
        value_str, shape_str = format_value(const_small_float)
        assert value_str == "[1.2, 3.4, 5.6]"
        assert shape_str == "(3,)"

        # Test large array integer-like (>3 elements)
        const_large_int = Variable([1, 2, 3, 4, 5, 6], "const_large_int")
        value_str, shape_str = format_value(const_large_int)
        assert "1, 2, 3, ..." in value_str
        assert shape_str == "(6,)"

        # Test large array floats (>3 elements)
        const_large_float = Variable([1.2, 3.4, 5.6, 7.8, 9.0, 10.1], "const_large_float")
        value_str, shape_str = format_value(const_large_float)
        assert "1.2, 3.4, 5.6, ..." in value_str
        assert shape_str == "(6,)"

        # Test 2D array (gets flattened for display since it has >3 elements)
        const_2d = Variable([[1, 2], [3, 4]], "const_2d")
        value_str, shape_str = format_value(const_2d)
        assert "1, 2, 3, ..." in value_str  # Flattened array with >3 elements
        assert shape_str == "(2, 2)"

        # Test small 2D array (≤3 elements when flattened)
        const_2d_small = Variable([[1, 2]], "const_2d_small")
        value_str, shape_str = format_value(const_2d_small)
        assert value_str == "[[1, 2]]"
        assert shape_str == "(1, 2)"

        # Test empty array
        const_empty = Variable([], "const_empty")
        value_str, shape_str = format_value(const_empty)
        assert value_str == "[]"
        assert shape_str == "(0,)"

    def test_format_value_scalar(self):
        r"""
        Tests the ``_format_value`` method with scalar parameters.
        """
        # Test scalar real integer constant
        const_real_int = Variable(3, "const_real_int")
        value_str, shape_str = format_value(const_real_int)
        assert value_str == "3"
        assert shape_str == "scalar"

        # Test scalar real float constant
        const_real_float = Variable(3.14159, "const_real_float")
        value_str, shape_str = format_value(const_real_float)
        assert value_str == "3.14159"
        assert shape_str == "scalar"

        # Test scalar complex constant positive imaginary part
        const_complex_pos_imag = Variable(1 + 2j, "const_complex_pos_imag", dtype=np.complex128)
        value_str, shape_str = format_value(const_complex_pos_imag)
        assert value_str == "1+2j"
        assert shape_str == "scalar"

        # Test scalar complex constant negative imaginary part
        const_complex_neg_imag = Variable(1 - 2j, "const_complex_neg_imag", dtype=np.complex128)
        value_str, shape_str = format_value(const_complex_neg_imag)
        assert value_str == "1-2j"
        assert shape_str == "scalar"

    def test_is_variable(self):
        r"""
        Tests that variables are mutable.
        """
        var = Variable(1, "var")

        var.value = 2
        assert var.value == 2

        var.update_fn = "update_orthogonal"
        assert var.update_fn == "update_orthogonal"

    def test_static_methods(self):
        r"""
        Tests the static methods.
        """
        var1 = Variable.symplectic(name="var1", N=1)
        assert var1.name == "var1"
        assert var1.update_fn == "update_symplectic"
        assert isinstance(var1, Variable)

        var2 = Variable.orthogonal(name="var2", N=1)
        assert var2.name == "var2"
        assert var2.update_fn == "update_orthogonal"

        var3 = Variable.unitary(name="var3", N=1)
        assert var3.name == "var3"
        assert var3.update_fn == "update_unitary"

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_orthogonal_matrix_properties(self, N):
        r"""
        Tests that Variable.orthogonal creates a proper orthogonal matrix.
        An orthogonal matrix O satisfies: O @ O.T = I
        """
        var = Variable.orthogonal(name="orthogonal_var", N=N, seed=42)

        # Check it's a Variable with correct update_fn
        assert isinstance(var, Variable)
        assert var.update_fn == "update_orthogonal"
        assert var.name == "orthogonal_var"

        # Check shape
        assert var.value.shape == (N, N)

        # Check orthogonality: O @ O.T = I
        O = var.value
        identity = np.eye(N)
        product = O @ O.T
        assert np.allclose(product, identity, atol=1e-10)

        # Check determinant is ±1
        det = np.linalg.det(O)
        assert np.isclose(np.abs(det), 1.0, atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_unitary_matrix_properties(self, N):
        r"""
        Tests that Variable.unitary creates a proper unitary matrix.
        A unitary matrix U satisfies: U @ U† = I (where U† is conjugate transpose)
        """
        var = Variable.unitary(name="unitary_var", N=N, seed=42)

        # Check it's a Variable with correct update_fn
        assert isinstance(var, Variable)
        assert var.update_fn == "update_unitary"
        assert var.name == "unitary_var"

        # Check shape
        assert var.value.shape == (N, N)

        # Check unitarity: U @ U† = I
        U = var.value
        identity = np.eye(N, dtype=complex)
        product = U @ np.conj(U.T)
        assert np.allclose(product, identity, atol=1e-10)

        # Check determinant has absolute value 1
        det = np.linalg.det(U)
        assert np.isclose(np.abs(det), 1.0, atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_symplectic_matrix_properties(self, N):
        r"""
        Tests that Variable.symplectic creates a proper symplectic matrix.
        A symplectic matrix S satisfies: S @ Omega @ S.T = Omega
        where Omega is the symplectic form matrix.
        """
        var = Variable.symplectic(name="symplectic_var", N=N, seed=42)

        # Check it's a Variable with correct update_fn
        assert isinstance(var, Variable)
        assert var.update_fn == "update_symplectic"
        assert var.name == "symplectic_var"

        # Check shape (symplectic matrices are 2N x 2N)
        assert var.value.shape == (2 * N, 2 * N)

        # Construct symplectic form matrix Omega
        # Omega = [[0, I], [-I, 0]] where I is N x N identity
        I = np.eye(N)
        O = np.zeros((N, N))
        Omega = np.block([[O, I], [-I, O]])

        # Check symplectic property: S @ Omega @ S.T = Omega
        S = var.value
        product = S @ Omega @ S.T
        assert np.allclose(product, Omega, atol=1e-10)

        # Check determinant is 1
        det = np.linalg.det(S)
        assert np.isclose(det, 1.0, atol=1e-10)

    def test_orthogonal_seed_reproducibility(self):
        r"""
        Tests that using the same seed produces the same orthogonal matrix.
        """
        var1 = Variable.orthogonal(name="var1", N=3, seed=123)
        var2 = Variable.orthogonal(name="var2", N=3, seed=123)

        assert np.allclose(var1.value, var2.value, atol=1e-15)

    def test_unitary_seed_reproducibility(self):
        r"""
        Tests that using the same seed produces the same unitary matrix.
        """
        var1 = Variable.unitary(name="var1", N=3, seed=123)
        var2 = Variable.unitary(name="var2", N=3, seed=123)

        assert np.allclose(var1.value, var2.value, atol=1e-15)

    def test_symplectic_seed_reproducibility(self):
        r"""
        Tests that using the same seed produces the same symplectic matrix.
        """
        var1 = Variable.symplectic(name="var1", N=2, seed=123)
        var2 = Variable.symplectic(name="var2", N=2, seed=123)

        assert np.allclose(var1.value, var2.value, atol=1e-15)

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3), (4, 2, 3)])
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_orthogonal_batch_shape(self, batch_shape, N):
        r"""
        Tests that Variable.orthogonal with batch_shape creates correct batches.
        """
        var = Variable.orthogonal(name="orthogonal_batch", N=N, batch_shape=batch_shape, seed=42)

        # Check shape
        expected_shape = (*batch_shape, N, N)
        assert var.value.shape == expected_shape

        # Check that each matrix in the batch is orthogonal
        batch_size = int(np.prod(batch_shape))
        matrices = var.value.reshape(batch_size, N, N)
        identity = np.eye(N)

        for i in range(batch_size):
            O = matrices[i]
            product = O @ O.T
            assert np.allclose(product, identity, atol=1e-10), f"Matrix {i} is not orthogonal"

            # Check determinant is ±1
            det = np.linalg.det(O)
            assert np.isclose(np.abs(det), 1.0, atol=1e-10), f"Matrix {i} determinant is not ±1"

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3)])
    @pytest.mark.parametrize("N", [2, 3])
    def test_orthogonal_batch_different_matrices(self, batch_shape, N):
        r"""
        Tests that different matrices are generated in the batch (not all the same).
        """
        var = Variable.orthogonal(name="orthogonal_batch", N=N, batch_shape=batch_shape, seed=42)

        batch_size = int(np.prod(batch_shape))
        matrices = var.value.reshape(batch_size, N, N)

        # Check that at least some matrices are different
        # (with high probability, random matrices should be different)
        all_same = True
        for i in range(1, batch_size):
            if not np.allclose(matrices[0], matrices[i], atol=1e-10):
                all_same = False
                break

        assert not all_same, "All matrices in batch are identical (should be different)"

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3), (4, 2, 3)])
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_unitary_batch_shape(self, batch_shape, N):
        r"""
        Tests that Variable.unitary with batch_shape creates correct batches.
        """
        var = Variable.unitary(name="unitary_batch", N=N, batch_shape=batch_shape, seed=42)

        # Check shape
        expected_shape = (*batch_shape, N, N)
        assert var.value.shape == expected_shape

        # Check that each matrix in the batch is unitary
        batch_size = int(np.prod(batch_shape))
        matrices = var.value.reshape(batch_size, N, N)
        identity = np.eye(N, dtype=complex)

        for i in range(batch_size):
            U = matrices[i]
            product = U @ np.conj(U.T)
            assert np.allclose(product, identity, atol=1e-10), f"Matrix {i} is not unitary"

            # Check determinant has absolute value 1
            det = np.linalg.det(U)
            assert np.isclose(np.abs(det), 1.0, atol=1e-10), f"Matrix {i} determinant is not 1"

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3)])
    @pytest.mark.parametrize("N", [2, 3])
    def test_unitary_batch_different_matrices(self, batch_shape, N):
        r"""
        Tests that different matrices are generated in the batch (not all the same).
        """
        var = Variable.unitary(name="unitary_batch", N=N, batch_shape=batch_shape, seed=42)

        batch_size = int(np.prod(batch_shape))
        matrices = var.value.reshape(batch_size, N, N)

        # Check that at least some matrices are different
        all_same = True
        for i in range(1, batch_size):
            if not np.allclose(matrices[0], matrices[i], atol=1e-10):
                all_same = False
                break

        assert not all_same, "All matrices in batch are identical (should be different)"

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3), (4, 2, 3)])
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_symplectic_batch_shape(self, batch_shape, N):
        r"""
        Tests that Variable.symplectic with batch_shape creates correct batches.
        """
        var = Variable.symplectic(name="symplectic_batch", N=N, batch_shape=batch_shape, seed=42)

        # Check shape (symplectic matrices are 2N x 2N)
        expected_shape = (*batch_shape, 2 * N, 2 * N)
        assert var.value.shape == expected_shape

        # Construct symplectic form matrix Omega
        I = np.eye(N)
        O = np.zeros((N, N))
        Omega = np.block([[O, I], [-I, O]])

        # Check that each matrix in the batch is symplectic
        batch_size = int(np.prod(batch_shape))
        matrices = var.value.reshape(batch_size, 2 * N, 2 * N)

        for i in range(batch_size):
            S = matrices[i]
            product = S @ Omega @ S.T
            assert np.allclose(product, Omega, atol=1e-10), f"Matrix {i} is not symplectic"

            # Check determinant is 1
            det = np.linalg.det(S)
            assert np.isclose(det, 1.0, atol=1e-10), f"Matrix {i} determinant is not 1"

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3)])
    @pytest.mark.parametrize("N", [2, 3])
    def test_symplectic_batch_different_matrices(self, batch_shape, N):
        r"""
        Tests that different matrices are generated in the batch (not all the same).
        """
        var = Variable.symplectic(name="symplectic_batch", N=N, batch_shape=batch_shape, seed=42)

        batch_size = int(np.prod(batch_shape))
        matrices = var.value.reshape(batch_size, 2 * N, 2 * N)

        # Check that at least some matrices are different
        all_same = True
        for i in range(1, batch_size):
            if not np.allclose(matrices[0], matrices[i], atol=1e-10):
                all_same = False
                break

        assert not all_same, "All matrices in batch are identical (should be different)"


class TestConstantMatrices:
    r"""
    Tests for Constant orthogonal, unitary, and symplectic matrices.
    """

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_orthogonal_matrix_properties(self, N):
        r"""
        Tests that Constant.orthogonal creates a proper orthogonal matrix.
        An orthogonal matrix O satisfies: O @ O.T = I
        """
        const = Constant.orthogonal(name="orthogonal_const", N=N, seed=42)

        # Check it's a Constant
        assert isinstance(const, Constant)
        assert const.name == "orthogonal_const"

        # Check shape
        assert const.value.shape == (N, N)

        # Check orthogonality: O @ O.T = I
        O = const.value
        identity = np.eye(N)
        product = O @ O.T
        assert np.allclose(product, identity, atol=1e-10)

        # Check determinant is ±1
        det = np.linalg.det(O)
        assert np.isclose(np.abs(det), 1.0, atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_unitary_matrix_properties(self, N):
        r"""
        Tests that Constant.unitary creates a proper unitary matrix.
        A unitary matrix U satisfies: U @ U† = I (where U† is conjugate transpose)
        """
        const = Constant.unitary(name="unitary_const", N=N, seed=42)

        # Check it's a Constant
        assert isinstance(const, Constant)
        assert const.name == "unitary_const"

        # Check shape
        assert const.value.shape == (N, N)

        # Check unitarity: U @ U† = I
        U = const.value
        identity = np.eye(N, dtype=complex)
        product = U @ np.conj(U.T)
        assert np.allclose(product, identity, atol=1e-10)

        # Check determinant has absolute value 1
        det = np.linalg.det(U)
        assert np.isclose(np.abs(det), 1.0, atol=1e-10)

    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_symplectic_matrix_properties(self, N):
        r"""
        Tests that Constant.symplectic creates a proper symplectic matrix.
        A symplectic matrix S satisfies: S @ Omega @ S.T = Omega
        where Omega is the symplectic form matrix.
        """
        const = Constant.symplectic(name="symplectic_const", N=N, seed=42)

        # Check it's a Constant
        assert isinstance(const, Constant)
        assert const.name == "symplectic_const"

        # Check shape (symplectic matrices are 2N x 2N)
        assert const.value.shape == (2 * N, 2 * N)

        # Construct symplectic form matrix Omega
        # Omega = [[0, I], [-I, 0]] where I is N x N identity
        I = np.eye(N)
        O = np.zeros((N, N))
        Omega = np.block([[O, I], [-I, O]])

        # Check symplectic property: S @ Omega @ S.T = Omega
        S = const.value
        product = S @ Omega @ S.T
        assert np.allclose(product, Omega, atol=1e-10)

        # Check determinant is 1
        det = np.linalg.det(S)
        assert np.isclose(det, 1.0, atol=1e-10)

    def test_orthogonal_seed_reproducibility(self):
        r"""
        Tests that using the same seed produces the same orthogonal matrix.
        """
        const1 = Constant.orthogonal(name="const1", N=3, seed=123)
        const2 = Constant.orthogonal(name="const2", N=3, seed=123)

        assert np.allclose(const1.value, const2.value, atol=1e-15)

    def test_unitary_seed_reproducibility(self):
        r"""
        Tests that using the same seed produces the same unitary matrix.
        """
        const1 = Constant.unitary(name="const1", N=3, seed=123)
        const2 = Constant.unitary(name="const2", N=3, seed=123)

        assert np.allclose(const1.value, const2.value, atol=1e-15)

    def test_symplectic_seed_reproducibility(self):
        r"""
        Tests that using the same seed produces the same symplectic matrix.
        """
        const1 = Constant.symplectic(name="const1", N=2, seed=123)
        const2 = Constant.symplectic(name="const2", N=2, seed=123)

        assert np.allclose(const1.value, const2.value, atol=1e-15)

    def test_constant_immutability(self):
        r"""
        Tests that Constant matrices remain immutable.
        """
        const = Constant.orthogonal(name="const", N=2, seed=42)

        # Should not be able to change value
        with pytest.raises(AttributeError):
            const.value = np.eye(2)

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3), (4, 2, 3)])
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_orthogonal_batch_shape(self, batch_shape, N):
        r"""
        Tests that Constant.orthogonal with batch_shape creates correct batches.
        """
        const = Constant.orthogonal(name="orthogonal_batch", N=N, batch_shape=batch_shape, seed=42)

        # Check shape
        expected_shape = (*batch_shape, N, N)
        assert const.value.shape == expected_shape

        # Check that each matrix in the batch is orthogonal
        batch_size = int(np.prod(batch_shape))
        matrices = const.value.reshape(batch_size, N, N)
        identity = np.eye(N)

        for i in range(batch_size):
            O = matrices[i]
            product = O @ O.T
            assert np.allclose(product, identity, atol=1e-10), f"Matrix {i} is not orthogonal"

            # Check determinant is ±1
            det = np.linalg.det(O)
            assert np.isclose(np.abs(det), 1.0, atol=1e-10), f"Matrix {i} determinant is not ±1"

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3), (4, 2, 3)])
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_unitary_batch_shape(self, batch_shape, N):
        r"""
        Tests that Constant.unitary with batch_shape creates correct batches.
        """
        const = Constant.unitary(name="unitary_batch", N=N, batch_shape=batch_shape, seed=42)

        # Check shape
        expected_shape = (*batch_shape, N, N)
        assert const.value.shape == expected_shape

        # Check that each matrix in the batch is unitary
        batch_size = int(np.prod(batch_shape))
        matrices = const.value.reshape(batch_size, N, N)
        identity = np.eye(N, dtype=complex)

        for i in range(batch_size):
            U = matrices[i]
            product = U @ np.conj(U.T)
            assert np.allclose(product, identity, atol=1e-10), f"Matrix {i} is not unitary"

            # Check determinant has absolute value 1
            det = np.linalg.det(U)
            assert np.isclose(np.abs(det), 1.0, atol=1e-10), f"Matrix {i} determinant is not 1"

    @pytest.mark.parametrize("batch_shape", [(2,), (3,), (2, 3), (4, 2, 3)])
    @pytest.mark.parametrize("N", [1, 2, 3, 5])
    def test_symplectic_batch_shape(self, batch_shape, N):
        r"""
        Tests that Constant.symplectic with batch_shape creates correct batches.
        """
        const = Constant.symplectic(name="symplectic_batch", N=N, batch_shape=batch_shape, seed=42)

        # Check shape (symplectic matrices are 2N x 2N)
        expected_shape = (*batch_shape, 2 * N, 2 * N)
        assert const.value.shape == expected_shape

        # Construct symplectic form matrix Omega
        I = np.eye(N)
        O = np.zeros((N, N))
        Omega = np.block([[O, I], [-I, O]])

        # Check that each matrix in the batch is symplectic
        batch_size = int(np.prod(batch_shape))
        matrices = const.value.reshape(batch_size, 2 * N, 2 * N)

        for i in range(batch_size):
            S = matrices[i]
            product = S @ Omega @ S.T
            assert np.allclose(product, Omega, atol=1e-10), f"Matrix {i} is not symplectic"

            # Check determinant is 1
            det = np.linalg.det(S)
            assert np.isclose(det, 1.0, atol=1e-10), f"Matrix {i} determinant is not 1"

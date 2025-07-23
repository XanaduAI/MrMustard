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

from mrmustard.math.parameters import (
    Constant,
    Variable,
    format_bounds,
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

    def test_format_bounds(self):
        r"""
        Tests the ``_format_bounds`` method with constant parameters.
        """
        # Constants should return "—"
        const = Constant(1.0, "const")
        bounds_str = format_bounds(const)
        assert bounds_str == "—"

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
        assert value_str == "[1, 2, 3]"
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

        with pytest.raises(AttributeError):
            const.name = "const2"


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
        assert var1.bounds == (None, None)
        assert var1.update_fn == "update_euclidean"

        var2 = Variable(np.array([1, 2, 3]), "var2", (0, 1), "update_orthogonal")
        assert np.allclose(var2.value, np.array([1, 2, 3]))
        assert var2.bounds == (0, 1)
        assert var2.update_fn == "update_orthogonal"

        var3 = Variable(1, "var3", dtype="int64")
        assert var3.value == 1
        assert var3.name == "var3"
        assert var3.value.dtype == "int64"

    def test_format_bounds_edge_cases(self):
        r"""
        Tests the ``_format_bounds`` method with edge cases.
        """
        # Test zero bounds
        var_zero = Variable(1.0, "var_zero", bounds=(0.0, 0.0))
        bounds_str = format_bounds(var_zero)
        assert bounds_str == "(0, 0)"

        # Test negative bounds
        var_negative = Variable(-1.0, "var_negative", bounds=(-10.0, -1.0))
        bounds_str = format_bounds(var_negative)
        assert bounds_str == "(-10, -1)"

    @pytest.mark.parametrize("bounds", [(None, None), (-2.5, 3.7), (0.0, None), (None, 10.0)])
    def test_format_bounds(self, bounds):
        r"""
        Tests the ``_format_bounds`` method with variable parameters.
        """
        var_bounds = Variable(1.0, "var_bounds", bounds=bounds)
        bounds_str = format_bounds(var_bounds)

        low = "-∞" if bounds[0] is None else f"{bounds[0]:.3g}"
        high = "+∞" if bounds[1] is None else f"{bounds[1]:.3g}"
        assert bounds_str == f"({low}, {high})"

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
        assert value_str == "[1, 2, 3]"
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

        with pytest.raises(AttributeError):
            var.name = "var2"

        with pytest.raises(AttributeError):
            var.bounds = (0, 1)

    def test_static_methods(self):
        r"""
        Tests the static methods.
        """
        va1 = Variable.symplectic(1, "var1")
        assert va1.update_fn == "update_symplectic"

        va2 = Variable.orthogonal(1, "va2")
        assert va2.update_fn == "update_orthogonal"

        var3 = Variable.unitary(1, "var3")
        assert var3.update_fn == "update_unitary"

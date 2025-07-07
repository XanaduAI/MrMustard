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
Unit tests for the :class:`ParameterSet`.
"""

import numpy as np

from mrmustard.math.parameter_set import ParameterSet
from mrmustard.math.parameters import Constant, Variable


class TestParameterSet:
    r"""
    Tests for ParameterSet.
    """

    def test_init(self):
        r"""
        Tests the init.
        """
        ps = ParameterSet()
        assert not ps.names
        assert not ps.constants
        assert not ps.variables

    def test_add_parameters(self):
        r"""
        Tests the ``add_parameter`` method.
        """
        const1 = Constant(1, "const1")
        const2 = Constant(2, "const2")
        var1 = Variable(1, "var1")

        ps = ParameterSet()
        ps.add_parameter(const1)
        ps.add_parameter(const2)
        ps.add_parameter(var1)

        assert ps.names == ["const1", "const2", "var1"]
        assert ps.constants == {"const1": const1, "const2": const2}
        assert ps.variables == {"var1": var1}

    def test_tagged_variables(self):
        r"""
        Tests the ``tagged_variables`` method.
        """
        const1 = Constant(1, "const1")
        const2 = Constant(2, "const2")
        var1 = Variable(1, "var1")

        ps = ParameterSet()
        ps.add_parameter(const1)
        ps.add_parameter(const2)
        ps.add_parameter(var1)

        variables = ps.tagged_variables("ciao")
        assert variables == {"ciao/var1": var1}

    def test_to_dict(self):
        r"""
        Tests the ``to_dict`` method.
        """
        const1 = Constant(1.2345, "const1")
        const2 = Constant(2.3456, "const2")
        var1 = Variable(3.4567, "var1")

        ps = ParameterSet()
        ps.add_parameter(const1)
        ps.add_parameter(const2)
        ps.add_parameter(var1)

        assert ps.to_dict() == {
            "const1": const1.value,
            "const2": const2.value,
            "var1": var1.value,
            "var1_trainable": True,
            "var1_bounds": (None, None),
        }

    def test_to_string(self):
        r"""
        Tests the ``to_string`` method.
        """
        const1 = Constant(1.2345, "const1")
        const2 = Constant(2.3456, "const2")
        var1 = Variable(3.4567, "var1")

        ps = ParameterSet()
        ps.add_parameter(const1)
        ps.add_parameter(const2)
        ps.add_parameter(var1)

        assert ps.to_string(1) == "1.2, 2.3, 3.5"
        assert ps.to_string(3) == "1.234, 2.346, 3.457"
        assert ps.to_string(10) == "1.2345, 2.3456, 3.4567"

    def test_eq(self):
        r"""
        Tests the ``__eq__`` method.
        """
        const1 = Constant(1, "c1")
        const2 = Constant([2, 3, 4], "c2")
        var1 = Variable(5, "v1")
        var2 = Variable([6, 7, 8], "v2")

        ps1 = ParameterSet()
        ps1.add_parameter(const1)
        ps1.add_parameter(const2)
        ps1.add_parameter(var1)
        ps1.add_parameter(var2)

        assert ps1 != 1.0

        ps2 = ParameterSet()
        ps2.add_parameter(const1)
        ps2.add_parameter(const2)
        ps2.add_parameter(var1)
        ps2.add_parameter(var2)

        assert ps1 == ps2

        ps3 = ParameterSet()
        ps3.add_parameter(const1)
        ps3.add_parameter(var1)

        assert ps1 != ps3

    def test_get_item(self):
        const1 = Constant(1, "c1")
        const2 = Constant([2, 3, 4], "c2")
        var1 = Variable(5, "v1")
        var2 = Variable([6, 7, 8], "v2")

        ps = ParameterSet()
        ps.add_parameter(const1)
        ps.add_parameter(const2)
        ps.add_parameter(var1)
        ps.add_parameter(var2)

        assert np.allclose(ps[0].constants["c1"].value, 1)
        assert np.allclose(ps[0].constants["c2"].value, 2)
        assert np.allclose(ps[0].variables["v1"].value, 5)
        assert np.allclose(ps[0].variables["v2"].value, 6)

        assert np.allclose(ps[1, 2].constants["c1"].value, 1)
        assert np.allclose(ps[1, 2].constants["c2"].value, [3, 4])
        assert np.allclose(ps[1, 2].variables["v1"].value, 5)
        assert np.allclose(ps[1, 2].variables["v2"].value, [7, 8])

    def test_format_value_scalar(self):
        r"""
        Tests the ``_format_value`` method with scalar parameters.
        """
        ps = ParameterSet()

        # Test scalar real constant
        const_real = Constant(3.14159, "const_real")
        value_str, shape_str = ps._format_value(const_real)
        assert value_str == "3.14159"
        assert shape_str == "scalar"

        # Test scalar complex constant positive imaginary part
        const_complex_pos_imag = Constant(1 + 2j, "const_complex_pos_imag", dtype=np.complex128)
        value_str, shape_str = ps._format_value(const_complex_pos_imag)
        assert value_str == "1+2j"
        assert shape_str == "scalar"

        # Test scalar complex constant negative imaginary part
        const_complex_neg_imag = Constant(1 - 2j, "const_complex_neg_imag", dtype=np.complex128)
        value_str, shape_str = ps._format_value(const_complex_neg_imag)
        assert value_str == "1-2j"
        assert shape_str == "scalar"

        # Test scalar variable
        var_scalar = Variable(2.718, "var_scalar")
        value_str, shape_str = ps._format_value(var_scalar)
        assert value_str == "2.718"
        assert shape_str == "scalar"

    def test_format_value_arrays(self):
        r"""
        Tests the ``_format_value`` method with array parameters.
        """
        ps = ParameterSet()

        # Test small array integer-like (≤3 elements)
        const_small_int = Constant([1.0, 2.0, 3.0], "const_small_int")
        value_str, shape_str = ps._format_value(const_small_int)
        assert value_str == "[1, 2, 3]"
        assert shape_str == "(3,)"

        # Test small array floats (≤3 elements)
        const_small_float = Constant([1.2, 3.4, 5.6], "const_small_float")
        value_str, shape_str = ps._format_value(const_small_float)
        assert value_str == "[1.2, 3.4, 5.6]"
        assert shape_str == "(3,)"

        # Test large array integer-like (>3 elements)
        const_large_int = Constant([1.2, 3.4, 5.6, 7.8, 9.0, 10.1], "const_large_int")
        value_str, shape_str = ps._format_value(const_large_int)
        assert "1.2, 3.4, 5.6, ..." in value_str
        assert shape_str == "(6,)"

        # Test large array floats (>3 elements)
        const_large_float = Constant([1, 2, 3, 4, 5, 6], "const_large_float")
        value_str, shape_str = ps._format_value(const_large_float)
        assert "1, 2, 3, ..." in value_str
        assert shape_str == "(6,)"

        # Test 2D array (gets flattened for display since it has >3 elements)
        const_2d = Constant([[1, 2], [3, 4]], "const_2d")
        value_str, shape_str = ps._format_value(const_2d)
        assert "1, 2, 3, ..." in value_str  # Flattened array with >3 elements
        assert shape_str == "(2, 2)"

        # Test small 2D array (≤3 elements when flattened)
        const_2d_small = Constant([[1, 2]], "const_2d_small")
        value_str, shape_str = ps._format_value(const_2d_small)
        assert value_str == "[[1, 2]]"
        assert shape_str == "(1, 2)"

        # Test empty array
        const_empty = Constant([], "const_empty")
        value_str, shape_str = ps._format_value(const_empty)
        assert value_str == "[]"
        assert shape_str == "(0,)"

    def test_format_dtype(self):
        r"""
        Tests the ``_format_dtype`` method.
        """
        ps = ParameterSet()

        # Test float64
        const_float64 = Constant(np.float64(1.0), "const_float64")
        dtype_str = ps._format_dtype(const_float64)
        assert dtype_str == "float64"

        # Test float32
        const_float32 = Constant(np.float32(1.0), "const_float32")
        dtype_str = ps._format_dtype(const_float32)
        assert dtype_str == "float32"

        # Test complex128
        const_complex128 = Constant(np.complex128(1 + 2j), "const_complex128")
        dtype_str = ps._format_dtype(const_complex128)
        assert dtype_str == "complex128"

        # Test complex64
        const_complex64 = Constant(np.complex64(1 + 2j), "const_complex64")
        dtype_str = ps._format_dtype(const_complex64)
        assert dtype_str == "complex64"

    def test_format_bounds_constant(self):
        r"""
        Tests the ``_format_bounds`` method with constant parameters.
        """
        ps = ParameterSet()

        # Constants should return "—"
        const = Constant(1.0, "const")
        bounds_str = ps._format_bounds(const)
        assert bounds_str == "—"

    def test_format_bounds_variable(self):
        r"""
        Tests the ``_format_bounds`` method with variable parameters.
        """
        ps = ParameterSet()

        # Test unbounded variable
        var_unbounded = Variable(1.0, "var_unbounded")
        bounds_str = ps._format_bounds(var_unbounded)
        assert bounds_str == "(-∞, +∞)"

        # Test bounded variable
        var_bounded = Variable(1.0, "var_bounded", bounds=(-2.5, 3.7))
        bounds_str = ps._format_bounds(var_bounded)
        assert bounds_str == "(-2.5, 3.7)"

        # Test lower bound only
        var_lower = Variable(1.0, "var_lower", bounds=(0.0, None))
        bounds_str = ps._format_bounds(var_lower)
        assert bounds_str == "(0, +∞)"

        # Test upper bound only
        var_upper = Variable(1.0, "var_upper", bounds=(None, 10.0))
        bounds_str = ps._format_bounds(var_upper)
        assert bounds_str == "(-∞, 10)"

    def test_format_bounds_edge_cases(self):
        r"""
        Tests the ``_format_bounds`` method with edge cases.
        """
        ps = ParameterSet()

        # Test zero bounds
        var_zero = Variable(1.0, "var_zero", bounds=(0.0, 0.0))
        bounds_str = ps._format_bounds(var_zero)
        assert bounds_str == "(0, 0)"

        # Test negative bounds
        var_negative = Variable(-1.0, "var_negative", bounds=(-10.0, -1.0))
        bounds_str = ps._format_bounds(var_negative)
        assert bounds_str == "(-10, -1)"

    def test_bool_and_empty_repr(self):
        r"""
        Tests the ``__bool__`` method and empty ParameterSet repr.
        """
        ps_empty = ParameterSet()
        assert not ps_empty
        assert repr(ps_empty) == "ParameterSet()"

        ps_with_param = ParameterSet()
        ps_with_param.add_parameter(Constant(1.0, "test"))
        assert bool(ps_with_param)

    def test_repr_integration(self):
        r"""
        Tests that ``__repr__`` integrates the formatting methods correctly.
        """
        ps = ParameterSet()
        ps.add_parameter(Constant(3.14, "pi"))
        ps.add_parameter(Variable(2.718, "e", bounds=(0, None)))
        ps.add_parameter(Constant([1, 2, 3], "array"))

        repr_str = repr(ps)

        # Check that the table is present and contains expected elements
        assert "ParameterSet (3 parameters)" in repr_str
        assert "pi" in repr_str
        assert "e" in repr_str
        assert "array" in repr_str
        assert "Constant" in repr_str
        assert "Variable" in repr_str
        assert "3.14" in repr_str
        assert "2.718" in repr_str
        assert "(0, +∞)" in repr_str

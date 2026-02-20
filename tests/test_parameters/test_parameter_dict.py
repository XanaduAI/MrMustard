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
Unit tests for the :class:`ParameterDict`.
"""

import numpy as np
import pytest

from mrmustard.parameters import Constant, ParameterDict, Variable


class TestParameterDict:
    r"""
    Tests for ParameterDict.
    """

    def test_init(self):
        r"""
        Tests the init.
        """
        pd = ParameterDict()
        assert not pd.names
        assert not pd.constants
        assert not pd.variables

    def test_add_parameters(self):
        r"""
        Tests adding parameters to the ParameterDict via constructor and dict-like assignment.
        """
        const1 = Constant(1, "const1")
        const2 = Constant(2, "const2")
        var1 = Variable(1, "var1")

        # Test constructor with positional args uses names as names
        pd = ParameterDict(const1, const2, var1)

        assert pd.names == ["const1", "const2", "var1"]
        assert pd.constants == {"const1": const1, "const2": const2}
        assert pd.variables == {"var1": var1}

        # Test dict-like assignment uses keys as names
        pd2 = ParameterDict()
        pd2["const1_key"] = const1
        pd2["const2_key"] = const2
        pd2["var1_key"] = var1

        assert pd2.names == ["const1_key", "const2_key", "var1_key"]
        assert pd2.constants == {"const1_key": const1, "const2_key": const2}
        assert pd2.variables == {"var1_key": var1}

    def test_to_string(self):
        r"""
        Tests the ``to_string`` method.
        """
        const1 = Constant(1.2345, "const1")
        const2 = Constant(2.3456, "const2")
        var1 = Variable(3.4567, "var1")

        pd = ParameterDict(const1, const2, var1)

        assert pd.to_string(1) == "1.2, 2.3, 3.5"
        assert pd.to_string(3) == "1.234, 2.346, 3.457"
        assert pd.to_string(10) == "1.2345, 2.3456, 3.4567"

    def test_eq(self):
        r"""
        Tests the ``__eq__`` method.
        """
        const1 = Constant(1, "c1")
        const2 = Constant([2, 3, 4], "c2")
        var1 = Variable(5, "v1")
        var2 = Variable([6, 7, 8], "v2")

        pd1 = ParameterDict(const1, const2, var1, var2)

        assert pd1 != 1.0

        pd2 = ParameterDict(const1, const2, var1, var2)

        assert pd1 == pd2

        pd3 = ParameterDict(const1, var1)

        assert pd1 != pd3

    def test_get_item(self):
        const1 = Constant(1, "c1")
        const2 = Constant([2, 3, 4], "c2")
        var1 = Variable(5, "v1")
        var2 = Variable([6, 7, 8], "v2")

        pd = ParameterDict(const1, const2, var1, var2)

        # Test access by name (dict-like access)
        assert pd["c1"] is const1
        assert pd["c2"] is const2
        assert pd["v1"] is var1
        assert pd["v2"] is var2

        # Test error case
        with pytest.raises(KeyError):
            pd["nonexistent"]  # Key not found

    def test_bool_and_empty_repr(self):
        r"""
        Tests the ``__bool__`` method and empty ParameterDict repr.
        """
        pd_empty = ParameterDict()
        assert not pd_empty
        assert repr(pd_empty) == "ParameterDict()"

        pd_with_param = ParameterDict(Constant(1.0, "test"))
        assert bool(pd_with_param)

    def test_repr_integration(self):
        r"""
        Tests that ``__repr__`` integrates the formatting methods correctly.
        """
        pd = ParameterDict(Constant(3.14, "pi"), Variable(2.718, "e"), Constant([1, 2, 3], "array"))

        repr_str = repr(pd)

        # Check that the table is present and contains expected elements
        assert "ParameterDict (3 parameters)" in repr_str
        assert "pi" in repr_str
        assert "e" in repr_str
        assert "array" in repr_str
        assert "Constant" in repr_str
        assert "Variable" in repr_str
        assert "3.14" in repr_str
        assert "2.718" in repr_str

    def test_getattr(self):
        r"""
        Tests the ``__getattr__`` method for attribute-style parameter access.
        """
        const1 = Constant(1.0, "some_constant")
        const2 = Constant([1, 2, 3], "array_param")
        var1 = Variable(3.14, "pi_variable")
        var2 = Variable(2.718, "e_variable")

        pd = ParameterDict(const1, const2, var1, var2)

        # Test accessing parameters by attribute name
        assert pd.some_constant is const1
        assert pd.array_param is const2
        assert pd.pi_variable is var1
        assert pd.e_variable is var2

        # Test that parameter values are accessible through the attribute
        assert pd.some_constant.value == 1.0
        assert np.array_equal(pd.array_param.value, [1, 2, 3])
        assert pd.pi_variable.value == 3.14
        assert pd.e_variable.value == 2.718

        # Test error case for non-existent parameter
        with pytest.raises(KeyError):
            pd.nonexistent  # noqa: B018

        # Test that normal attributes still work
        assert pd.names == ["some_constant", "array_param", "pi_variable", "e_variable"]
        assert len(pd.constants) == 2
        assert len(pd.variables) == 2

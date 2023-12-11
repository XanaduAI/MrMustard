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

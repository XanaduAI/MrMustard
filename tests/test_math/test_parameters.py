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

import pytest

from mrmustard.math import Math
from mrmustard.math.parameters import (
    Constant,
    Variable,
    update_euclidean,
    update_orthogonal,
    update_unitary,
    update_symplectic,
)
import numpy as np

math = Math()


class TestConstant:
    def test_init(self):
        const1 = Constant(1, "const1")
        assert const1.value == 1
        assert const1.name == "const1"

        math_const = math.new_constant(2, "const2")
        const2 = Constant(math_const, "const2")
        assert const2.value == math_const
        assert const2.name == "const2"

        const3 = Constant(np.array([1, 2, 3]), "const3")
        assert np.allclose(const3.value, np.array([1, 2, 3]))

    def test_is_const(self):
        const = Constant(1, "const")

        with pytest.raises(AttributeError, match="can't set attribute"):
            const.value = 2

        with pytest.raises(AttributeError, match="can't set attribute"):
            const.name = "const2"


class TestVariable:
    def test_init(self):
        var1 = Variable(1, "var1")
        assert var1.value == 1
        assert var1.name == "var1"
        assert var1.bounds == (None, None)
        assert var1.update_fn == update_euclidean

        math_var = math.new_variable(2, (0, 1), "var2")
        var2 = Variable(math_var, "var2")
        assert var2.value == math_var
        assert var2.name == "var2"
        assert var2.update_fn == update_euclidean

        var3 = Variable(np.array([1, 2, 3]), "var3", (0, 1), update_orthogonal)
        assert np.allclose(var3.value, np.array([1, 2, 3]))
        assert var3.bounds == (0, 1)
        assert var3.update_fn == update_orthogonal

    def test_is_variable(self):
        var = Variable(1, "var")

        var.value = 2
        assert var.value == 2

        var.update_fn = update_orthogonal
        assert var.update_fn == update_orthogonal

        with pytest.raises(AttributeError, match="can't set attribute"):
            var.name = "var2"

        with pytest.raises(AttributeError, match="can't set attribute"):
            var.bounds = (0, 1)

    def test_static_methods(self):
        va1 = Variable.symplectic(1, "var1")
        assert va1.update_fn == update_symplectic

        va2 = Variable.orthogonal(1, "va2")
        assert va2.update_fn == update_orthogonal

        var3 = Variable.unitary(1, "var3")
        assert var3.update_fn == update_unitary

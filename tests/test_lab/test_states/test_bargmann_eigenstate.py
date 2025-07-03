# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the BargmannEigenstate class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import BargmannEigenstate


class TestBargmannEigenstate:
    r"""
    Tests for the ``BargmannEigenstate`` class.
    """

    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_init(self, batch_shape):
        "Tests the initialization."
        alpha = math.broadcast_to(0.1 + 0.0j, batch_shape)

        be = BargmannEigenstate(0, alpha)
        assert be.name == "BargmannEigenstate"
        assert math.allclose(be.parameters.alpha.value, 0.1 + 0.0j)
        assert be.modes == (0,)
        assert math.allclose(be.ansatz.A, math.zeros((1, 1)))
        assert math.allclose(be.ansatz.b, 0.1)
        assert math.allclose(be.ansatz.c, 1.0)

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1])
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_numerial(self, alpha, batch_shape):
        "A numerical test."
        alpha = math.broadcast_to(alpha, batch_shape, dtype=math.complex128)
        be = BargmannEigenstate(0, alpha)
        assert math.allclose(
            be.contract(be.dual, "zip").ansatz.scalar,
            math.exp(alpha**2),
        )  # TODO: revisit rshift

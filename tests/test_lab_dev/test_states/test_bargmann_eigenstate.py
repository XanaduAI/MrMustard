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

# pylint: disable=unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement

import pytest

from mrmustard import math
from mrmustard.lab_dev.states import BargmannEigenstate


class TestBargmannEigenstate:
    r"""
    Tests for the ``BargmannEigenstate`` class.
    """

    def test_init(self):
        "Tests the initialization."
        be = BargmannEigenstate([0, 1], [0.1, 0.2])
        assert be.name == "BargmannEigenstate"
        assert math.allclose(be.parameters.alpha.value, [0.1, 0.2])
        assert be.modes == [0, 1]
        assert math.allclose(be.ansatz.b[0], [0.1, 0.2])
        assert math.allclose(be.ansatz.A[0], math.zeros((2, 2)))
        assert be.ansatz.c[0] == 1.0

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1])
    def test_numerial(self, alpha):
        "A numerical test."
        be = BargmannEigenstate([0], alpha)
        assert be >> be.dual == math.exp(complex(alpha**2))

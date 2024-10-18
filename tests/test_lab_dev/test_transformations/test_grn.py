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

"""Tests for the ``GRN`` class."""

# pylint: disable=protected-access, missing-function-docstring, expression-not-assigned

import numpy as np

from mrmustard import math
from mrmustard.lab_dev.transformations import GRN


class TestGRN:
    r"""
    Tests for the ``Attenuator`` class.
    """

    def test_init(self):
        "Tests the GRN initialization."
        grn = GRN([0])

    def test_grn(self):
        a = np.random.random((4,4))
        Y = a @ a.T
        phi = GRN([0,1], Y)

        _, Y_ans = phi.XY

        assert math.allclose(Y_ans, Y)
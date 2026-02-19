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

"""Tests for the ``GaussianRandomNoise`` class."""

from mrmustard import math, settings
from mrmustard.lab.states import GaussianDM
from mrmustard.lab.transformations import GaussianRandomNoise


class TestGRN:
    r"""
    Tests for the ``GaussianRandomNoise`` class.
    """

    def test_init(self):
        "Tests the GaussianRandomNoise initialization."

        a = settings.get_rng().random((2, 2))
        grn = GaussianRandomNoise((0,), a @ a.T)
        assert grn.name == "GaussianRandomNoise"
        assert grn.modes == (0,)

    def test_grn(self):
        "Tests if the A matrix of GaussianRandomNoise is computed correctly."
        a = settings.get_rng().random((4, 4))
        Y = a @ a.T
        phi = GaussianRandomNoise((0, 1), Y)

        _, Y_ans = phi.XY

        assert math.allclose(Y_ans, Y)
        assert phi.is_physical
        assert math.allclose((GaussianDM.random((0, 1)) >> phi).probability, 1.0)

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

"""Tests for the ``RealInterferometer`` class."""

from mrmustard import math
from mrmustard.lab.transformations import Identity, RealInterferometer


class TestRealInterferometer:
    r"""
    Tests the Interferometer gate (Interferometer)
    """

    def test_init(self):
        "Tests initialization of an Interferometer object"
        u_int = RealInterferometer((0, 1, 2))
        assert u_int.modes == (0, 1, 2)
        assert u_int.name == "RealInterferometer"
        assert u_int.symplectic.shape == (6, 6)

        orth = math.random_orthogonal(2)
        u_int = RealInterferometer((0, 1), orthogonal=orth)
        assert u_int.symplectic.shape == (4, 4)
        assert math.allclose(u_int.symplectic[:2, 2:], math.zeros((2, 2)))

    def test_application(self):
        "Tests the correctness of the application of a RealInterferometer gate"
        u_int = RealInterferometer((0, 1))
        assert u_int >> u_int.dual == Identity((0, 1))

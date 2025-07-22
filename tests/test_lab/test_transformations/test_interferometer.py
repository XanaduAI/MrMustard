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

"""Tests for the ``Interferometer`` class."""

from mrmustard import math
from mrmustard.lab.transformations import Identity, Interferometer


class TestInterferometer:
    r"""
    Tests the Interferometer gate (Interferometer)
    """

    def test_init(self):
        "Tests initialization of an Interferometer object"
        u_int = Interferometer((0, 1, 2))
        assert u_int.modes == (0, 1, 2)
        assert u_int.name == "Interferometer"
        assert u_int.symplectic.shape == (6, 6)

        unitary = math.random_unitary(2)
        u_int = Interferometer((0, 1), unitary=unitary)
        assert u_int.symplectic.shape == (4, 4)

    def test_application(self):
        "Tests the correctness of the application of an Interferometer gate"
        u_int = Interferometer((0, 1))
        assert u_int >> u_int.dual == Identity((0, 1))

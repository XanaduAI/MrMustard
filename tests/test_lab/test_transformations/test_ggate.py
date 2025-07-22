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

"""
Test for the Ggate class.
"""

from mrmustard import math
from mrmustard.lab import Ggate, Identity


class TestGgate:
    r"""
    Tests the Ggate class
    """

    def test_init(self):
        "Tests the init of Ggate"
        Eye = Ggate(modes=0, symplectic=math.eye(2))
        assert isinstance(Eye, Ggate)
        assert Eye.name == "Ggate"
        assert math.allclose(Eye.symplectic, math.eye(2))

    def test_ggate_is_unitary_1mode(self):
        """Test that the Ggate applied to its dual is the identity."""
        G = Ggate(modes=0, symplectic=math.random_symplectic(1))
        assert G >> G.dual == Identity(0)
        assert G.dual >> G == Identity(0)

    def test_ggate_is_unitary_2mode(self):
        """Test that the Ggate applied to its dual is the identity."""
        G = Ggate(modes=(0, 1), symplectic=math.random_symplectic(2))
        assert G >> G.dual == Identity((0, 1))
        assert G.dual >> G == Identity((0, 1))

    def test_ggate_is_unitary_3mode(self):
        """Test that the Ggate applied to its dual is the identity."""
        G = Ggate(modes=(0, 1, 2), symplectic=math.random_symplectic(3))
        assert G >> G.dual == Identity((0, 1, 2))
        assert G.dual >> G == Identity((0, 1, 2))

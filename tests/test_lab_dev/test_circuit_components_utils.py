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

""" Tests for circuit components utils. """

# pylint: disable=fixme, missing-function-docstring, protected-access, pointless-statement

import pytest

from mrmustard import math
from mrmustard.lab_dev.circuit_components_utils import DsMap

class TestDsMap:
    r"""
    Tests for the ``DsMap`` class.
    """

    modes = [[0], [1, 2], [9, 7]]
    s = [0, -1, 1]

    @pytest.mark.parametrize("modes,s", zip(modes, s))
    def test_init(self, modes, s):
        dsmap = DsMap(modes, s)

        assert dsmap.name == "DsMap"
        assert dsmap.modes == [modes] if not isinstance(modes, list) else sorted(modes)

    def test_representation(self):
        rep1 = DsMap(modes=[0], s=0).representation
        assert math.allclose(rep1.A, [[[0, 1], [1, 0]]])
        assert math.allclose(rep1.b, [[0.1 + 0.1j, -0.1 + 0.1j]])
        assert math.allclose(rep1.c, [0.990049833749168])

        rep2 = DsMap(modes=[0, 1], s=1).representation
        assert math.allclose(rep2.A, [[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]])
        assert math.allclose(rep2.b, [[0.1 + 0.1j, 0.2 + 0.1j, -0.1 + 0.1j, -0.2 + 0.1j]])
        assert math.allclose(rep2.c, [0.9656054162575665])

        rep3 = DsMap(modes=[1, 8], s=-1).representation
        assert math.allclose(rep3.A, [[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]])
        assert math.allclose(rep3.b, [[0.1, 0.2, -0.1, -0.2]])
        assert math.allclose(rep3.c, [0.9753099120283327])

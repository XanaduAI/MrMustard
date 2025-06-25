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

"""Tests for the ``Identity`` class."""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab.transformations import Identity


class TestIdentity:
    r"""
    Tests for the ``Identity`` class.
    """

    modes = [0, (1, 2), (7, 9)]

    @pytest.mark.parametrize("modes", modes)
    def test_init(
        self,
        modes,
    ):
        gate = Identity(modes)

        assert gate.name == "Identity"
        assert gate.modes == (modes,) if isinstance(modes, int) else modes

    def test_representation(self):
        rep1 = Identity(modes=0).ansatz
        assert math.allclose(
            rep1.A,
            [
                [
                    [0.0 + 0.0j, 1 + 0j],
                    [1 + 0j, 0.0 + 0.0j],
                ],
            ],
        )
        assert math.allclose(rep1.b, np.zeros((1, 2)))
        assert math.allclose(rep1.c, [1.0 + 0.0j])

        rep2 = Identity(modes=(0, 1)).ansatz
        assert math.allclose(
            rep2.A,
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ],
            ],
        )
        assert math.allclose(rep2.b, np.zeros((1, 4)))
        assert math.allclose(rep2.c, [1.0 + 0.0j])

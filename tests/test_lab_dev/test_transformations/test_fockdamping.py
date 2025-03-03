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

"""Tests for the ``FockDamping`` class."""

# pylint: disable=missing-function-docstring, expression-not-assigned

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab_dev.transformations import FockDamping, Identity


class TestFockDamping:
    r"""
    Tests for the ``FockDamping`` class.
    """

    modes = [0, 1, 7]
    damping = [0.1, 0.2, 0.3]

    @pytest.mark.parametrize("modes,damping", zip(modes, damping))
    def test_init(self, modes, damping):
        gate = FockDamping(modes, damping)

        assert gate.name == "FockDamping"
        assert gate.modes == (modes,)
        assert math.allclose(gate.parameters.damping.value, damping)

    def test_representation(self):
        rep1 = FockDamping(mode=0, damping=0.1).ansatz
        e = math.exp(-0.1)
        assert math.allclose(
            rep1.A,
            [
                [
                    [0, e],
                    [
                        e,
                        0,
                    ],
                ]
            ],
        )
        assert math.allclose(rep1.b, math.zeros((1, 2)))
        assert math.allclose(rep1.c, [1.0])

    def test_trainable_parameters(self):
        gate1 = FockDamping(0, 0.1)
        gate2 = FockDamping(0, 0.1, damping_trainable=True, damping_bounds=(0.0, 0.2))

        with pytest.raises(AttributeError):
            gate1.parameters.damping.value = 0.3

        gate2.parameters.damping.value = 0.2
        assert gate2.parameters.damping.value == 0.2

    def test_identity(self):
        rep1 = FockDamping(mode=0, damping=0.0).ansatz
        rep2 = Identity(modes=0).ansatz

        assert math.allclose(rep1.A, rep2.A)
        assert math.allclose(rep1.b, rep2.b)
        assert math.allclose(rep1.c, rep2.c)

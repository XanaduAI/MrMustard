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

"""Tests for the ``Gket`` and ``Gdm`` classes."""

# pylint: disable=unspecified-encoding, missing-function-docstring, expression-not-assigned, pointless-statement


from mrmustard import math
from mrmustard.lab_dev.states import Gdm, Gket, Vacuum
from mrmustard.lab_dev.transformations import Unitary


class TestGket:
    r"""
    Tests for the ``Gket`` class.
    """

    def test_init(self):
        "Tests initialization"
        gket = Gket([0, 1])

        assert gket.modes == [0, 1]
        assert gket.symplectic.value.shape == (4, 4)
        assert gket.name == "Gket"
        assert math.allclose(gket.probability, 1.0)

    def test_correctness(self):
        "Tests is the attributes are consistent"

        g = Gket([0])
        sym = g.symplectic.value
        u = Unitary.from_symplectic([0], sym)
        assert g == Vacuum([0]) >> u


class TestGdm:
    r"""
    Tests the ``Gdm`` class.
    """

    def test_init(self):
        "Tests the initialization"

        rho = Gdm([0, 1], [0.2, 0.3])

        assert rho.modes == [0, 1]
        assert rho.name == "Gdm"
        assert math.allclose(rho.betas.value, math.astensor([0.2, 0.3]))
        assert rho.symplectic.value.shape == (4, 4)
        assert math.allclose(rho.probability, 1.0)

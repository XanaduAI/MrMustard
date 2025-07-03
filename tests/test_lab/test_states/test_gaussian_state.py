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

from mrmustard import math
from mrmustard.lab.states import DM, GDM, GKet, Vacuum
from mrmustard.lab.transformations import Unitary


class TestGKet:
    r"""
    Tests for the ``GKet`` class.
    """

    def test_init(self):
        "Tests initialization"
        gket = GKet((0, 1))

        assert gket.modes == (0, 1)
        assert gket.parameters.symplectic.value.shape == (4, 4)
        assert gket.name == "GKet"
        assert math.allclose(gket.probability, 1.0)

    def test_correctness(self):
        "Tests is the attributes are consistent"

        g = GKet(0)
        sym = g.parameters.symplectic.value
        u = Unitary.from_symplectic((0,), sym)
        assert g == Vacuum(0) >> u

    def test_getitem(self):
        "Tests the getitem of the GKet"

        psi = GKet(0)
        assert psi == psi[0]

        phi = GKet((0, 1))
        assert isinstance(phi[0], DM)


class TestGDM:
    r"""
    Tests the ``GDM`` class.
    """

    def test_init(self):
        "Tests the initialization"

        rho = GDM((0, 1), [0.2, 0.3])

        assert rho.modes == (0, 1)
        assert rho.name == "GDM"
        assert math.allclose(rho.parameters.beta.value, math.astensor([0.2, 0.3]))
        assert rho.parameters.symplectic.value.shape == (4, 4)
        assert math.allclose(rho.probability, 1.0)

    def test_getitem(self):
        "Tests the getitem of GDM"

        rho = GDM(0, 0.2)
        assert rho == rho[0]

        sigma = GDM((0, 1), [0.5, 0.4])
        assert isinstance(sigma[0], DM)

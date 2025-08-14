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
        symplectic = math.random_symplectic(2)
        gket = GKet((0, 1), symplectic)

        assert gket.modes == (0, 1)
        assert gket.name == "GKet"
        assert math.allclose(gket.probability, 1.0)

    def test_correctness(self):
        "Tests is the attributes are consistent"

        symplectic = math.random_symplectic(1)
        g = GKet(0, symplectic)
        u = Unitary.from_symplectic((0,), symplectic)
        assert g == Vacuum(0) >> u

    def test_getitem(self):
        "Tests the getitem of the GKet"

        symplectic1 = math.random_symplectic(1)
        psi = GKet(0, symplectic1)
        assert psi == psi[0]

        symplectic2 = math.random_symplectic(2)
        phi = GKet((0, 1), symplectic2)
        assert isinstance(phi[0], DM)


class TestGDM:
    r"""
    Tests the ``GDM`` class.
    """

    def test_init(self):
        "Tests the initialization"

        symplectic = math.random_symplectic(2)
        rho = GDM((0, 1), [0.2, 0.3], symplectic)

        assert rho.modes == (0, 1)
        assert rho.name == "GDM"
        assert math.allclose(rho.probability, 1.0)

    def test_getitem(self):
        "Tests the getitem of GDM"

        symplectic1 = math.random_symplectic(1)
        rho = GDM(0, 0.2, symplectic1)
        assert rho == rho[0]

        symplectic2 = math.random_symplectic(2)
        sigma = GDM((0, 1), [0.5, 0.4], symplectic2)
        assert isinstance(sigma[0], DM)

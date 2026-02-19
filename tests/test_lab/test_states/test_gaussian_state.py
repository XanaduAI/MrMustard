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

import pytest

from mrmustard import math
from mrmustard.lab.states import DM, GaussianDM, GaussianKet, Vacuum
from mrmustard.lab.transformations import Unitary
from mrmustard.parameters import Variable


class TestGaussianKet:
    r"""
    Tests for the ``GaussianKet`` class.
    """

    def test_init(self):
        "Tests initialization"
        symplectic = math.random_symplectic(2)
        gket = GaussianKet((0, 1), symplectic)

        assert gket.modes == (0, 1)
        assert gket.parameters.symplectic.value.shape == (4, 4)
        assert gket.name == "GaussianKet"
        assert math.allclose(gket.probability, 1.0)

    def test_correctness(self):
        "Tests is the attributes are consistent"

        g = GaussianKet.random(modes=0)
        sym = g.parameters.symplectic.value
        u = Unitary.from_symplectic((0,), sym)
        assert g == Vacuum(0) >> u

    def test_get_modes(self):
        "Tests the get_modes of the GaussianKet"

        psi = GaussianKet.random(modes=0)
        assert psi == psi.get_modes(0)

        phi = GaussianKet.random(modes=(0, 1))
        assert isinstance(phi.get_modes(0), DM)

    def test_random(self):
        "Tests the random method of the GaussianKet"
        psi = GaussianKet.random(modes=0, seed=1)
        assert isinstance(psi, GaussianKet)
        assert psi.modes == (0,)
        assert psi.parameters.symplectic.value.shape == (2, 2)
        assert math.allclose(psi.probability, 1.0)
        assert math.allclose(psi.parameters.symplectic.value, math.random_symplectic(1, seed=1))


class TestGaussianDM:
    r"""
    Tests the ``GaussianDM`` class.
    """

    def test_init(self):
        "Tests the initialization"
        symplectic = math.random_symplectic(2)
        rho = GaussianDM((0, 1), [0.2, 0.3], symplectic)

        assert rho.modes == (0, 1)
        assert rho.name == "GaussianDM"
        assert math.allclose(rho.parameters.beta.value, math.astensor([0.2, 0.3]))
        assert rho.parameters.symplectic.value.shape == (4, 4)
        assert math.allclose(rho.probability, 1.0)

        beta_var = Variable(math.astensor([0.2, 0.3]), "beta")
        rho = GaussianDM((0, 1), beta_var, symplectic)
        assert math.allclose(rho.parameters.beta.value, beta_var.value)

    def test_get_modes(self):
        "Tests the get_modes of GaussianDM"

        rho = GaussianDM.random(modes=0)
        assert rho == rho.get_modes(0)

        sigma = GaussianDM.random(modes=(0, 1))
        assert isinstance(sigma.get_modes(0), DM)

    def test_random(self):
        "Tests the random method of GaussianDM"
        rho = GaussianDM.random(modes=0, min_beta=0.2, seed=1)
        assert isinstance(rho, GaussianDM)
        assert rho.modes == (0,)
        assert rho.parameters.beta.value.shape == (1,)
        assert rho.parameters.symplectic.value.shape == (2, 2)
        assert math.allclose(rho.probability, 1.0)
        assert math.allclose(rho.parameters.symplectic.value, math.random_symplectic(1, seed=1))

    def test_random_max_beta_equal_to_min_beta_but_not_less(self):
        rho = GaussianDM.random(modes=0, min_beta=0.3, max_beta=0.3)
        assert isinstance(rho, GaussianDM)
        assert rho.modes == (0,)
        assert rho.parameters.beta.value.shape == (1,)
        with pytest.raises(ValueError, match="high - low < 0"):
            GaussianDM.random(modes=0, min_beta=0.3, max_beta=0.2)

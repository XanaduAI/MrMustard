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

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab.states import DM, GaussianDM, GaussianKet, Vacuum
from mrmustard.lab.states.gaussian_state import _random_alphas_in_disk
from mrmustard.lab.transformations import Dgate, Unitary
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

    def test_random_displaced_preserves_normalization(self):
        """Displaced random kets must remain normalized."""
        psi = GaussianKet.random(modes=(0, 1), max_disp=2.0, seed=42)
        assert np.isclose(psi.probability, 1.0, atol=1e-8)

    def test_random_displaced_is_reproducible(self):
        """Same seed must produce the same displaced state."""
        psi1 = GaussianKet.random(modes=0, max_disp=1.5, seed=7)
        psi2 = GaussianKet.random(modes=0, max_disp=1.5, seed=7)
        assert psi1 == psi2

    def test_random_displaced_matches_manual_construction(self):
        """Displaced random ket must equal the undisplaced ket >> Dgate(alpha)."""
        seed, modes, max_disp = 42, (0, 1), 1.5

        psi = GaussianKet.random(modes=modes, max_disp=max_disp, seed=seed)

        psi_manual = GaussianKet.random(modes=modes, seed=seed)
        rng = settings.get_rng(seed)
        for mode, alpha in zip(modes, _random_alphas_in_disk(len(modes), max_disp, rng)):
            psi_manual = psi_manual >> Dgate(mode, alpha)

        assert psi == psi_manual


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

    def test_random_displaced_preserves_normalization(self):
        """Displaced random density matrices must remain normalized."""
        rho = GaussianDM.random(modes=(0, 1), max_disp=2.0, seed=42)
        assert np.isclose(rho.probability, 1.0, atol=1e-8)

    def test_random_displaced_is_reproducible(self):
        """Same seed must produce the same displaced state."""
        rho1 = GaussianDM.random(modes=0, max_disp=1.5, seed=7)
        rho2 = GaussianDM.random(modes=0, max_disp=1.5, seed=7)
        assert rho1 == rho2

    def test_random_displaced_matches_manual_construction(self):
        """Displaced random DM must equal the undisplaced DM >> Dgate(alpha)."""
        seed, modes, max_disp = 42, (0, 1), 1.5

        rho = GaussianDM.random(modes=modes, max_disp=max_disp, seed=seed)

        rho_manual = GaussianDM.random(modes=modes, seed=seed)
        rng = settings.get_rng(seed)
        for mode, alpha in zip(modes, _random_alphas_in_disk(len(modes), max_disp, rng)):
            rho_manual = rho_manual >> Dgate(mode, alpha)

        assert rho == rho_manual


class TestRandomAlphasInDisk:
    """Tests for the disk-uniform sampling helper."""

    def test_magnitudes_bounded(self):
        """All sampled magnitudes must be at most max_disp."""
        rng = np.random.default_rng(0)
        magnitudes = np.abs(_random_alphas_in_disk(1000, 2.0, rng))
        assert np.all(magnitudes <= 2.0 + 1e-15)

    def test_zero_radius_gives_zero_displacement(self):
        """max_disp=0 must produce alpha=0 for every mode."""
        rng = np.random.default_rng(0)
        alphas = _random_alphas_in_disk(5, 0.0, rng)
        assert np.allclose(alphas, 0.0)

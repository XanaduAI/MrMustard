import pytest

from mrmustard import *
import numpy as np
import tensorflow as tf
from thewalrus.random import random_covariance
from thewalrus.quantum import real_to_complex_displacements
from mrmustard.physics import gaussian as gp, fock as fp

from mrmustard.math import Math

math = Math()


class TestGaussianStates:
    @pytest.mark.parametrize("hbar", [1 / 2, 1.0, 2.0, 1.6])
    @pytest.mark.parametrize("num_modes", np.arange(5, 10))
    @pytest.mark.parametrize("pure", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_fidelity_is_symmetric(self, num_modes, hbar, pure, block_diag):
        """Test that the fidelity is symmetric"""
        cov1 = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
        means1 = np.sqrt(2 * hbar) * np.random.rand(2 * num_modes)
        cov2 = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
        means2 = np.sqrt(2 * hbar) * np.random.rand(2 * num_modes)
        f12 = gp.fidelity(means1, cov1, means2, cov2, hbar)
        f21 = gp.fidelity(means2, cov2, means1, cov1, hbar)
        assert np.allclose(f12, f21)

    @pytest.mark.parametrize("hbar", [1 / 2, 1.0, 2.0, 1.6])
    @pytest.mark.parametrize("num_modes", np.arange(5, 10))
    @pytest.mark.parametrize("pure", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_fidelity_is_leq_one(self, num_modes, hbar, pure, block_diag):
        """Test that the fidelity is between 0 and 1"""
        cov1 = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
        means1 = np.sqrt(2 * hbar) * np.random.rand(2 * num_modes)
        cov2 = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
        means2 = np.sqrt(2 * hbar) * np.random.rand(2 * num_modes)
        f12 = gp.fidelity(means1, cov1, means2, cov2, hbar)
        assert 0 <= np.real_if_close(f12) < 1.0

    @pytest.mark.parametrize("hbar", [1 / 2, 1.0, 2.0, 1.6])
    @pytest.mark.parametrize("num_modes", np.arange(5, 10))
    @pytest.mark.parametrize("pure", [True, False])
    @pytest.mark.parametrize("block_diag", [True, False])
    def test_fidelity_with_self(self, num_modes, hbar, pure, block_diag):
        """Test that the fidelity of two identical quantum states is 1"""
        cov = random_covariance(num_modes, hbar=hbar, pure=pure, block_diag=block_diag)
        means = np.random.rand(2 * num_modes)
        assert np.allclose(gp.fidelity(means, cov, means, cov, hbar=hbar), 1, atol=1e-4)

    @pytest.mark.parametrize("num_modes", np.arange(5, 10))
    @pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0, 1.6])
    def test_fidelity_coherent_state(self, num_modes, hbar):
        """Test the fidelity of two multimode coherent states"""
        beta1 = np.random.rand(num_modes) + 1j * np.random.rand(num_modes)
        beta2 = np.random.rand(num_modes) + 1j * np.random.rand(num_modes)
        means1 = real_to_complex_displacements(np.concatenate([beta1, beta1.conj()]), hbar=hbar)
        means2 = real_to_complex_displacements(np.concatenate([beta2, beta2.conj()]), hbar=hbar)
        cov1 = hbar * np.identity(2 * num_modes) / 2
        cov2 = hbar * np.identity(2 * num_modes) / 2
        fid = gp.fidelity(means1, cov1, means2, cov2, hbar=hbar)
        expected = np.exp(-np.linalg.norm(beta1 - beta2) ** 2)
        assert np.allclose(expected, fid)

    @pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0, 1.6])
    @pytest.mark.parametrize("r1", np.random.rand(3))
    @pytest.mark.parametrize("r2", np.random.rand(3))
    def test_fidelity_squeezed_vacuum(self, r1, r2, hbar):
        """Tests fidelity between two squeezed states"""
        cov1 = np.diag([np.exp(2 * r1), np.exp(-2 * r1)]) * hbar / 2
        cov2 = np.diag([np.exp(2 * r2), np.exp(-2 * r2)]) * hbar / 2
        mu = np.zeros([2])
        assert np.allclose(1 / np.cosh(r1 - r2), gp.fidelity(mu, cov1, mu, cov2, hbar=hbar))

    @pytest.mark.parametrize("n1", [0.5, 1.0, 2.0, 1.6])
    @pytest.mark.parametrize("n2", [0.5, 1.0, 2.0, 1.6])
    @pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0, 1.6])
    def test_fidelity_thermal(self, n1, n2, hbar):
        """Test fidelity between two thermal states"""
        expected = 1 / (1 + n1 + n2 + 2 * n1 * n2 - 2 * np.sqrt(n1 * n2 * (n1 + 1) * (n2 + 1)))
        cov1 = hbar * (n1 + 0.5) * np.identity(2)
        cov2 = hbar * (n2 + 0.5) * np.identity(2)
        mu1 = np.zeros([2])
        mu2 = np.zeros([2])
        assert np.allclose(expected, gp.fidelity(mu1, cov1, mu2, cov2, hbar=hbar))

    @pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0, 1.6])
    @pytest.mark.parametrize("r", [-2.0, 0.0, 2.0])
    @pytest.mark.parametrize("alpha", np.random.rand(10) + 1j * np.random.rand(10))
    def test_fidelity_vac_to_displaced_squeezed(self, r, alpha, hbar):
        """Calculates the fidelity between a coherent squeezed state and vacuum"""
        cov1 = np.diag([np.exp(2 * r), np.exp(-2 * r)]) * hbar / 2
        means1 = real_to_complex_displacements(np.array([alpha, np.conj(alpha)]), hbar=hbar)
        means2 = np.zeros([2])
        cov2 = np.identity(2) * hbar / 2
        expected = (
            np.exp(-np.abs(alpha) ** 2)
            * np.abs(np.exp(np.tanh(r) * np.conj(alpha) ** 2))
            / np.cosh(r)
        )
        assert np.allclose(expected, gp.fidelity(means1, cov1, means2, cov2, hbar=hbar))


class TestMixedStates:

    state1 = 1 / 2 * np.eye(2)

    state2 = 1 / 3 * np.ones((2, 2))
    state2[1, 1] = 2 / 3

    def test_fidelity_with_self(self):
        """Test that the fidelity of two identical quantum states is 1"""
        assert np.allclose(fp.fidelity(self.state1, self.state1, False, False), 1, atol=1e-4)

    def test_fidelity_is_symmetric(self):
        """Test that the fidelity is symmetric and between 0 and 1"""
        f12 = fp.fidelity(self.state1, self.state2, False, False)
        f21 = fp.fidelity(self.state2, self.state1, False, False)
        assert np.allclose(f12, f21)

    def test_fidelity_leq_one(self):
        """Test that the fidelity is symmetric and between 0 and 1"""
        f12 = fp.fidelity(self.state1, self.state2, False, False)
        assert 0 <= np.real_if_close(f12) < 1.0

    def test_fidelity_formula(self):
        """Test fidelity of known mixed states."""
        expected = 5 / 6
        assert np.allclose(expected, fp.fidelity(self.state1, self.state2, False, False))

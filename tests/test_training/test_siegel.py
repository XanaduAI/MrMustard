# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Siegel-disk Riemannian optimization (Bergman metric)."""

import numpy as np
import optax
import pytest

from mrmustard import math
from mrmustard.parameters import Variable
from mrmustard.training import Optimizer
from mrmustard.training.parameter_update import siegel_retract


class TestSiegelBackend:
    r"""Backend-level Siegel primitives.

    Tolerance convention used throughout this file:

    * ``atol=1e-12`` — algebraic identities that involve only symmetrization or
      sampling (e.g. ``Z == Z.T`` after :func:`random_siegel`, or after the
      ``sym`` projector in :func:`euclidean_to_siegel`). Only float64 roundoff.
    * ``atol=1e-10`` — single short numerical chain (e.g. zero-tangent retraction
      goes through one ``eigh`` + ``inv`` but the inputs are exact).
    * ``atol=1e-9`` — operator-norm slack from the boundary after a Bergman walk.
    * ``atol=1e-8`` — singular-value comparisons via ``np.linalg.svd`` of an
      ``eigh``-composed iterate; the eigendecomposition condition number
      dominates.
    * ``atol=1e-3`` — optimizer convergence; depends on lr × steps, not roundoff.
    """

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 4)])
    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_random_siegel_lives_in_open_disk(self, n, batch_shape):
        Z = math.asnumpy(math.random_siegel(n, max_r=0.9, seed=42, batch_shape=batch_shape))
        assert Z.shape == (*batch_shape, n, n)
        assert np.allclose(Z, np.swapaxes(Z, -1, -2))
        assert np.linalg.svd(Z, compute_uv=False).max() < 0.9

    def test_random_siegel_rejects_non_strict_radius(self):
        with pytest.raises(ValueError, match="max_r"):
            math.random_siegel(3, max_r=1.0, seed=0)

    def test_euclidean_to_siegel_returns_symmetric(self):
        Z = math.astensor(math.random_siegel(4, max_r=0.7, seed=2))
        rng = np.random.default_rng(3)
        G = math.astensor(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
        grad = math.asnumpy(math.euclidean_to_siegel(Z, G))
        assert np.allclose(grad, grad.T, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 4])
    def test_siegel_retract_stays_in_disk(self, n):
        r"""Walking with Bergman-projected tangents keeps the iterate strictly
        inside :math:`\mathcal{D}_g`: the metric shrinks tangent magnitudes by
        :math:`(I - ZZ^*)(I - Z^*Z)` so ``tanh`` does not saturate in finite
        precision.
        """
        Z = math.astensor(math.random_siegel(n, max_r=0.5, seed=4))
        rng = np.random.default_rng(5)
        for _ in range(30):
            raw = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            eta = math.euclidean_to_siegel(Z, math.astensor(raw + raw.T))
            Z = siegel_retract(Z, eta)
            assert np.linalg.svd(math.asnumpy(Z), compute_uv=False).max() < 1.0 - 1e-9

    def test_siegel_retract_zero_tangent_is_identity(self):
        Z_np = math.random_siegel(3, max_r=0.6, seed=6)
        Z_new = math.asnumpy(
            siegel_retract(math.astensor(Z_np), math.astensor(np.zeros_like(Z_np)))
        )
        assert np.allclose(Z_new, Z_np, atol=1e-10)

    def test_siegel_retract_at_origin_matches_tanh_of_takagi_values(self):
        r"""At :math:`Z = 0` the retraction reduces to the origin geodesic
        :math:`V = U\,\tanh(S)\,U^T` where :math:`\eta = U\,\mathrm{diag}(\sigma)\,U^T`
        is the Takagi factorization. Two rotation-invariant checks:

        * For non-degenerate :math:`\eta`, the singular values of :math:`V` must
          equal :math:`\tanh` applied to the singular values of :math:`\eta`, and
          :math:`V` must be complex symmetric.
        * For :math:`\eta = c\,I` (fully-degenerate Takagi values, where the
          Takagi unitary is *arbitrary*), the closed form
          :math:`V = (c/|c|)\tanh(|c|)\,I` must hold exactly --- this is the
          case any Takagi-unitary-dependent implementation would get wrong.
        """
        n = 4
        rng = np.random.default_rng(11)
        raw = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        eta_np = 0.3 * (raw + raw.T)
        Z0 = math.astensor(np.zeros((n, n), dtype=np.complex128))
        V_new = math.asnumpy(siegel_retract(Z0, math.astensor(eta_np)))
        sigma_eta = np.linalg.svd(eta_np, compute_uv=False)
        sigma_V = np.linalg.svd(V_new, compute_uv=False)
        # SVD: 1e-8 tolerance class.
        assert np.allclose(np.sort(sigma_V), np.sort(np.tanh(sigma_eta)), atol=1e-8)
        assert np.allclose(V_new, V_new.T, atol=1e-10)

        c = 0.4 + 0.2j
        eta_deg = math.astensor(c * np.eye(n, dtype=np.complex128))
        V_deg = math.asnumpy(siegel_retract(Z0, eta_deg))
        V_deg_ref = (c / abs(c)) * np.tanh(abs(c)) * np.eye(n)
        assert np.allclose(V_deg, V_deg_ref, atol=1e-10)


class TestSiegelParameter:
    def test_variable_siegel_tags_update_fn(self):
        v = Variable.siegel(n=3, max_r=0.8, seed=7)
        assert v.update_fn == "update_siegel"
        assert v.shape == (3, 3)
        v_np = math.asnumpy(v.value)
        assert np.allclose(v_np, v_np.T, atol=1e-12)
        assert np.linalg.svd(v_np, compute_uv=False).max() < 0.8


@pytest.mark.requires_backend("jax")
class TestSiegelOptimizer:
    def test_optimizer_converges_to_siegel_target(self):
        r"""Minimising :math:`\|Z - Z_\mathrm{target}\|_F^2` for a target inside
        :math:`\mathcal{D}_g` recovers the target.
        """
        n = 3
        Z_target = math.astensor(math.random_siegel(n, max_r=0.5, seed=8))
        Z_var = Variable.siegel(n=n, max_r=0.1, seed=9)

        def cost_fn(Z_var):
            diff = Z_var.value - Z_target
            return math.real(math.sum(diff * math.conj(diff)))

        opt = Optimizer(siegel_lr=0.1)
        (Z_opt,) = opt.minimize(cost_fn, by_optimizing=[Z_var], max_steps=400)
        assert math.allclose(Z_opt.value, Z_target, atol=1e-3)

    def test_optimizer_confines_z_under_outward_cost(self):
        r"""The ill-posed cost :math:`-\|Z\|_F^2` would diverge in Euclidean
        optimization, but the Bergman gradient vanishes at the boundary and the
        geodesic retraction keeps ``Z`` inside :math:`\mathcal{D}_g`. Plain SGD
        avoids AdaBelief's adaptive rescaling blowing up on the unbounded cost.
        """
        Z_var = Variable.siegel(n=3, max_r=0.3, seed=10)

        def cost_fn(Z_var):
            return -math.real(math.sum(Z_var.value * math.conj(Z_var.value)))

        opt = Optimizer(siegel_lr=0.5, siegel_optimizer=optax.sgd)
        (Z_opt,) = opt.minimize(cost_fn, by_optimizing=[Z_var], max_steps=200)
        assert np.linalg.svd(math.asnumpy(Z_opt.value), compute_uv=False).max() < 1.0
